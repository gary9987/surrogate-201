import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from spektral.layers import GINConvBatch, GlobalSumPool, GlobalMaxPool, GlobalAvgPool, DiffPool
import tensorflow as tf
from invertible_neural_networks.flow import NVP
from models.TransformerAE import Decoder
from models.base import PositionalEmbedding


class GINEncoder(Model):
    def __init__(self, n_hidden, mlp_hidden, activation: str, dropout=0.):
        super(GINEncoder, self).__init__()
        self.graph_conv = GINConvBatch(n_hidden, mlp_hidden=mlp_hidden, mlp_activation=activation, mlp_batchnorm=True,
                                       activation=activation)
        self.bn = tensorflow.keras.layers.BatchNormalization()
        self.dropout = tensorflow.keras.layers.Dropout(dropout)
        self.mean = Dense(n_hidden)
        self.var = Dense(n_hidden)

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.bn(out)
        out = self.dropout(out)
        mean = self.mean(out)
        var = self.var(out)
        return mean, var


class TransformerDecoder(Decoder):
    def __init__(self, num_layers, d_model, num_heads, dff, input_length, num_ops, num_nodes, num_adjs,
                 dropout_rate=0.0):
        super(TransformerDecoder, self).__init__(num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs,
                                                 dropout_rate)
        self.pos_embedding = PositionalEmbedding(d_model=d_model, input_length=input_length)

        self.adj_cls = [
            Dense(2, activation='softmax')
            for _ in range(num_adjs)
        ]

        self.ops_cls = tf.keras.layers.Dense(num_ops, activation='softmax')
        self.adj_weight = tf.keras.layers.Dense(num_nodes, activation='relu')
        self.adj_cls = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x)

        ops_cls = self.ops_cls(x)

        x = self.adj_weight(x)  # (8, 16) -> (8, 8)
        x = tf.reshape(x, (tf.shape(x)[0], -1, 1))  # (8, 8, 1)
        adj_cls = self.adj_cls(x)
        # ops_cls = tf.stack([self.ops_cls[i](flatten_x) for i in range(self.num_nodes)], axis=-1)
        # ops_cls = tf.transpose(ops_cls, (0, 2, 1))

        # adj_cls = tf.stack([self.adj_cls[i](flatten_x) for i in range(self.num_adjs)], axis=-1)
        # adj_cls = tf.transpose(adj_cls, (0, 2, 1))

        return ops_cls, adj_cls


class MLPDecoder(tf.keras.Model):
    def __init__(self, num_layers, hidden_dim, num_ops, num_nodes):
        super(MLPDecoder, self).__init__()
        self.num_layers = num_layers
        self.num_ops = num_ops
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.ops_cls = [
            Dense(hidden_dim, activation='relu')
            for _ in range(num_layers-1)
        ]
        self.ops_cls.append(Dense(num_ops, activation='softmax'))

        self.adj_transform = [
            Dense(hidden_dim, activation='relu')
            for _ in range(num_layers-1)
        ]
        self.adj_transform.append(Dense(num_nodes, activation='relu'))
        self.adj_cls = Dense(2, activation='softmax')

    def call(self, x):
        ops = x
        for i in self.ops_cls:
            ops = i(ops)

        adj = x
        for i in self.adj_transform:
            adj = i(adj)
        adj = tf.reshape(adj, (tf.shape(adj)[0], -1, 1))
        adj = self.adj_cls(adj)

        return ops, adj


class GraphAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim, num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs, eps_scale=0.01,
                 dropout_rate=0.0):
        super(GraphAutoencoder, self).__init__()
        self.ckpt_weights = None
        self.d_model = d_model
        self.num_ops = num_ops
        self.num_adjs = num_adjs
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.eps_scale = eps_scale
        self.encoder = GINEncoder(self.latent_dim, [128, 128, 128, 128], 'relu', dropout_rate)

        self.decoder = MLPDecoder(num_layers, dff, num_ops, num_nodes)
        '''
        self.decoder = TransformerDecoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                                          dff=dff, input_length=num_nodes, num_ops=num_ops, num_nodes=num_nodes,
                                          num_adjs=num_adjs,
                                          dropout_rate=dropout_rate)
        '''

    def sample(self, mean, log_var, eps_scale=0.01):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(log_var * 0.5) * eps * eps_scale

    def call(self, inputs, kl_reduction='mean'):
        latent_mean, latent_var = self.encoder(inputs)  # (batch_size, context_len, d_model)
        c = self.sample(latent_mean, latent_var, self.eps_scale)
        kl_loss = -0.5 * tf.reduce_sum(1 + latent_var - tf.square(latent_mean) - tf.exp(latent_var), axis=-1)

        if kl_reduction == 'mean':
            # (1)
            kl_loss = tf.reduce_mean(kl_loss)
        elif kl_reduction == 'none':
            # (batch_size)
            kl_loss = tf.reduce_mean(kl_loss, axis=-1)

        ops_cls, adj_cls = self.decoder(c)  # (batch_size, target_len, d_model)

        # Return the final output
        return ops_cls, adj_cls, kl_loss, latent_mean

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, inputs):
        ops_cls, adj_cls = self.decoder(inputs)
        ops = []
        for i in range(len(ops_cls)):
            ops.append(tf.argmax(ops_cls[i], axis=-1))
        adj = tf.cast(tf.argmax(adj_cls, axis=-1), tf.float32)
        return ops, adj, ops_cls, adj_cls

    def get_weights_to_self_ckpt(self):
        self.ckpt_weights = self.get_weights()

    def set_weights_from_self_ckpt(self):
        if self.ckpt_weights is None:
            raise ValueError('No weights to set')
        self.set_weights(self.ckpt_weights)


class GraphAutoencoderNVP(GraphAutoencoder):
    def __init__(self, nvp_config, latent_dim, num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs,
                 eps_scale=0.01, dropout_rate=0.0):
        super(GraphAutoencoderNVP, self).__init__(latent_dim, num_layers, d_model, num_heads, dff, num_ops,
                                                  num_nodes, num_adjs, eps_scale, dropout_rate)
        if nvp_config['inp_dim'] is None:
            nvp_config['inp_dim'] = latent_dim

        self.pad_dim = nvp_config['inp_dim'] - latent_dim * num_nodes
        self.nvp = NVP(**nvp_config)

    def call(self, inputs, kl_reduction='mean'):
        ops_cls, adj_cls, kl_loss, latent_mean = super().call(inputs, kl_reduction)
        latent_mean = tf.reshape(latent_mean, (tf.shape(latent_mean)[0], -1))
        latent_mean = tf.concat([latent_mean, tf.zeros((tf.shape(latent_mean)[0], self.pad_dim))], axis=-1)
        reg = self.nvp(latent_mean)
        return ops_cls, adj_cls, kl_loss, reg, latent_mean

    def inverse(self, z):
        return self.nvp.inverse(z)


class GraphAutoencoderEnsembleNVP(GraphAutoencoder):
    def __init__(self, num_nvp, nvp_config, latent_dim, num_layers, d_model, num_heads, dff, num_ops, num_nodes,
                 num_adjs,
                 eps_scale=0.01, dropout_rate=0.0):
        super(GraphAutoencoderEnsembleNVP, self).__init__(latent_dim, num_layers, d_model, num_heads, dff, num_ops,
                                                          num_nodes, num_adjs, eps_scale, dropout_rate)
        if nvp_config['inp_dim'] is None:
            nvp_config['inp_dim'] = latent_dim

        self.num_nvp = num_nvp
        self.pad_dim = nvp_config['inp_dim'] - latent_dim * num_nodes
        self.nvp_list = [NVP(**nvp_config) for _ in range(num_nvp)]

    def call(self, inputs, kl_reduction='mean'):
        ops_cls, adj_cls, kl_loss, latent_mean = super().call(inputs, kl_reduction)
        latent_mean = tf.reshape(latent_mean, (tf.shape(latent_mean)[0], -1))
        latent_mean = tf.concat([latent_mean, tf.zeros((tf.shape(latent_mean)[0], self.pad_dim))], axis=-1)
        reg = tf.transpose(tf.stack([nvp(latent_mean) for nvp in self.nvp_list]), (1, 0, 2))
        return ops_cls, adj_cls, kl_loss, reg, latent_mean

    def inverse(self, z):
        return tf.transpose(tf.stack([nvp.inverse(z) for nvp in self.nvp_list]), (1, 0, 2))


def bpr_loss(y_true, y_pred):
    N = tf.shape(y_true)[0]  # y_true.shape[0] = batch size
    lc_length = tf.shape(y_true)[1]

    total_loss = tf.constant([])

    for i in range(lc_length):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(total_loss, tf.TensorShape([None]))]
        )
        loss_value = 0.0
        for j in range(N):
            loss_value += tf.reduce_sum(tf.keras.backend.switch(y_true[:, i] > y_true[j, i],
                                                                -tf.math.log(tf.sigmoid(y_pred[:, i] - y_pred[j, i])),
                                                                0))
        total_loss = tf.concat([total_loss, tf.expand_dims(loss_value, 0)], 0)

    return total_loss / tf.cast(N, tf.float32) ** 2


def get_rank_weight(y_true):
    N = tf.shape(y_true)[0]  # y_true.shape[0] = batch size
    rank = tf.subtract(y_true, tf.transpose(y_true))
    rank = tf.where(rank < 0, 1., 0.)
    rank = tf.reduce_sum(rank, axis=1)
    weight = tf.math.reciprocal(rank + tf.cast(N, tf.float32) * 10e-3)
    return weight


def weighted_mse(y_true, y_pred):
    mse = tf.keras.losses.mse(y_true, y_pred)
    weight = get_rank_weight(y_true)
    '''
    mse = tf.keras.losses.mse(y_true, y_pred)
    weight = []
    for i in range(N):
        rank = tf.cast(tf.reduce_sum(tf.where(y_true > y_true[i], 1, 0)), tf.float32)
        weight.append(1. / (tf.cast(N, tf.float32) * 10e-3 + rank))
    '''
    return tf.reduce_sum(tf.multiply(mse, weight))
