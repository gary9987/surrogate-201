import os
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.util import SeedStream
from tensorflow_probability.python import random as tfp_random

tfpl = tfp.layers
tfk = tf.keras
tfkl = tfk.layers
tfkc = tfk.callbacks
K = tfk.backend


class ForceSTDDenseFlipout(tfpl.DenseFlipout):
    def __init__(self, std, **kwargs):
        super(ForceSTDDenseFlipout, self).__init__(**kwargs)
        self.std = tfp.math.softplus_inverse(std)

    def _apply_variational_kernel(self, inputs):
        if (not isinstance(self.kernel_posterior, independent_lib.Independent) or
                not isinstance(self.kernel_posterior.distribution, normal_lib.Normal)):
            raise TypeError(
                '`DenseFlipout` requires '
                '`kernel_posterior_fn` produce an instance of '
                '`tfd.Independent(tfd.Normal)` '
                '(saw: \"{}\").'.format(self.kernel_posterior.name))
        self.kernel_posterior_affine = tfp.distributions.Normal(
            loc=tf.zeros_like(self.kernel_posterior.distribution.loc),
            scale=tf.fill(tf.shape(self.kernel_posterior.distribution.scale), self.std))
        self.kernel_posterior_affine_tensor = (
            self.kernel_posterior_tensor_fn(self.kernel_posterior_affine))
        self.kernel_posterior_tensor = None

        input_shape = tf.shape(inputs)
        batch_shape = input_shape[:-1]

        seed_stream = SeedStream(self.seed, salt='DenseFlipout')

        sign_input = tfp_random.rademacher(
            input_shape,
            dtype=inputs.dtype,
            seed=seed_stream())
        sign_output = tfp_random.rademacher(
            tf.concat([batch_shape,
                       tf.expand_dims(self.units, 0)], 0),
            dtype=inputs.dtype,
            seed=seed_stream())
        perturbed_inputs = tf.matmul(
            inputs * sign_input, self.kernel_posterior_affine_tensor) * sign_output

        outputs = tf.matmul(inputs, self.kernel_posterior.distribution.loc)
        outputs += perturbed_inputs
        return outputs


class NN(tfkl.Layer):
    '''
    Reused from https://github.com/MokkeMeguru/glow-realnvp-tutorial
    '''
    def __init__(self, n_dim, n_layer=3, n_hid=512, activation='relu', name='fc_layer', use_bias=True):
        super(NN, self).__init__(name=name)
        self.n_dim = n_dim
        self.n_layer = n_layer
        self.n_hid = n_hid
        self.layer_list = []
        self.std = 0.6
        for _ in range(n_layer):
            self.layer_list.append(ForceSTDDenseFlipout(std=self.std,
                                                         units=n_hid,
                                                         activation=activation,
                                                         bias_posterior_fn=tfp.layers.default_mean_field_normal_fn()))

        self.log_s_layer = ForceSTDDenseFlipout(std=self.std, units=n_dim // 2, activation='tanh', name='log_s_layer', bias_posterior_fn=tfp.layers.default_mean_field_normal_fn())
        self.t_layer = ForceSTDDenseFlipout(std=self.std, units=n_dim // 2, activation='linear', name='t_layer', bias_posterior_fn=tfp.layers.default_mean_field_normal_fn())

    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        log_s = self.log_s_layer(x)
        t = self.t_layer(x)
        return log_s, t


class TwoNVPCouplingLayers(tfkl.Layer):
    def __init__(self, inp_dim, n_hid_layer, n_hid_dim, name, shuffle_type, use_bias=True):
        super(TwoNVPCouplingLayers, self).__init__(name=name)
        '''Implementation of Coupling layers in Ardizzone et al (2018)

        # Forward
        y1 = x1 * exp(s2(x2)) + t2(x2)
        y2 = x2 * exp(s1(x1)) + t1(x1)
        # Inverse
        x2 = (y2 - t1(y1)) * exp(-s1(y1))
        x1 = (y1 - t2(y2)) * exp(-s2(y2))
        '''
        self.inp_dim = inp_dim
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.shuffle_type = shuffle_type
        self.nn1 = NN(inp_dim, n_hid_layer, n_hid_dim, use_bias=use_bias)
        self.nn2 = NN(inp_dim, n_hid_layer, n_hid_dim, use_bias=use_bias)
        self.idx = tf.Variable(list(range(self.inp_dim)),
                               shape=(self.inp_dim,),
                               trainable=False,
                               name='index',
                               dtype='int64')
        if self.shuffle_type == 'random':
            self.idx.assign(tf.random.shuffle(self.idx))
        elif self.shuffle_type == 'reverse':
            self.idx.assign(tf.reverse(self.idx, axis=[0]))

    def call(self, x):
        x = self.shuffle(x, isInverse=False)
        x1, x2 = self.split(x)
        log_s2, t2 = self.nn2(x2)
        y1 = x1 * tf.math.exp(log_s2) + t2
        log_s1, t1 = self.nn1(y1)
        y2 = x2 * tf.math.exp(log_s1) + t1
        y = tf.concat([y1, y2], axis=-1)
        # Add loss
        self.log_det_J = log_s1 + log_s2
        self.add_loss(- tf.math.reduce_sum(self.log_det_J))
        return y

    def inverse(self, y):
        y1, y2 = self.split(y)
        log_s1, t1 = self.nn1(y1)
        x2 = (y2 - t1) * tf.math.exp(-log_s1)
        log_s2, t2 = self.nn2(x2)
        x1 = (y1 - t2) * tf.math.exp(-log_s2)
        x = tf.concat([x1, x2], axis=-1)
        x = self.shuffle(x, isInverse=True)
        return x

    def shuffle(self, x, isInverse=False):
        if not isInverse:
            # Forward
            idx = self.idx
        else:
            # Inverse
            idx = tf.map_fn(tf.math.invert_permutation,
                            tf.expand_dims(self.idx, 0))
            idx = tf.squeeze(idx)
        x = tf.transpose(x)
        x = tf.gather(x, idx)
        x = tf.transpose(x)
        return x

    def split(self, x):
        dim = self.inp_dim
        x = tf.reshape(x, [-1, dim])
        return x[:, :dim//2], x[:, dim//2:]


class NVP_BNN(tfk.Model):
    def __init__(self, inp_dim, n_couple_layer, n_hid_layer, n_hid_dim, name, shuffle_type='reverse', use_bias=True):
        super(NVP_BNN, self).__init__(name=name)
        self.inp_dim = inp_dim
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.shuffle_type = shuffle_type
        self.AffineLayers = []
        for i in range(n_couple_layer):
            layer = TwoNVPCouplingLayers(
                inp_dim, n_hid_layer, n_hid_dim,
                name=f'Layer{i}', shuffle_type=shuffle_type, use_bias=use_bias)
            self.AffineLayers.append(layer)

    def call(self, x):
        '''Forward: data (x) --> latent (z); inference'''
        z = x
        for i in range(self.n_couple_layer):
            z = self.AffineLayers[i](z)
        return z

    def inverse(self, z):
        '''Inverse: latent (z) --> data (y); sampling'''
        x = z
        for i in reversed(range(self.n_couple_layer)):
            x = self.AffineLayers[i].inverse(x)
        return x


if __name__ == "__main__":
    inp_dim = 128
    n_couple_layer = 2
    n_hid_layer = 4
    n_hid_dim = 64
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    model = NVP_BNN(inp_dim, n_couple_layer, n_hid_layer, n_hid_dim, name='NVP')
    import numpy as np
    x = np.random.randn(1, inp_dim)
    a = model(x)
    model.summary()
    inv = model.inverse(a)
    print(a)
    print(x)
    print(inv)
