import tensorflow.keras.layers
from spektral.data import BatchLoader
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from spektral.layers import GINConvBatch, GlobalSumPool, GlobalMaxPool, GlobalAvgPool, DiffPool
import tensorflow as tf

from invertible_neural_networks.flow import NVP
from models.TransformerAE import Decoder, positional_encoding


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, input_length):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Dense(d_model, use_bias=False)
        self.pos_encoding = positional_encoding(length=input_length, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(tf.reshape(x, [tf.shape(x)[0], length, -1]))
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class Graph_Model(Model):

    def __init__(self, n_hidden, mlp_hidden, activation: str, epochs, dropout=0., is_only_validation_data=False):
        super(Graph_Model, self).__init__()
        self.graph_conv = GINConvBatch(n_hidden, mlp_hidden=mlp_hidden, mlp_activation=activation, mlp_batchnorm=True,
                                       activation=activation)
        self.bn = tensorflow.keras.layers.BatchNormalization()
        self.pool = GlobalMaxPool()
        self.dropout = tensorflow.keras.layers.Dropout(dropout)
        if is_only_validation_data:
            self.dense = Dense(epochs)
        elif epochs == 12:
            self.dense = Dense(3 * epochs)  # (train_acc, valid_acc, test_acc) * 12 epochs
        elif epochs == 200:
            self.dense = Dense(2 * epochs)  # (train_acc, valid_acc) * 200 epochs
        else:
            raise NotImplementedError('epochs')

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.bn(out)
        out = self.pool(out)
        out = self.dropout(out)
        out = self.dense(out)
        return out


class GINEncoder(Model):
    def __init__(self, n_hidden, mlp_hidden, activation: str, dropout=0.):
        super(GINEncoder, self).__init__()
        self.graph_conv = GINConvBatch(n_hidden, mlp_hidden=mlp_hidden, mlp_activation=activation, mlp_batchnorm=True,
                                       activation=activation)
        self.bn = tensorflow.keras.layers.BatchNormalization()
        self.pool = GlobalMaxPool()
        self.dropout = tensorflow.keras.layers.Dropout(dropout)
        self.mean = Dense(n_hidden)
        self.var = Dense(n_hidden)

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.bn(out)
        out = self.pool(out)
        out = self.dropout(out)
        mean = self.mean(out)
        var = self.var(out)
        return mean, var


class TransformerDecoder(Decoder):
    def __init__(self, num_layers, d_model, num_heads, dff, input_length, num_ops, num_nodes, num_adjs, dropout_rate=0.0):
        super(TransformerDecoder, self).__init__(num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs,
                                                 dropout_rate)
        self.pos_embedding = PositionalEmbedding(d_model=d_model, input_length=input_length)
        self.adj_cls = [
            Dense(2, activation='softmax')
            for _ in range(num_adjs)
        ]

    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x)

        flatten_x = tf.reshape(x, (tf.shape(x)[0], -1))
        ops_cls = tf.stack([self.ops_cls[i](flatten_x) for i in range(self.num_nodes)], axis=-1)
        ops_cls = tf.transpose(ops_cls, (0, 2, 1))

        adj_cls = tf.stack([self.adj_cls[i](flatten_x) for i in range(self.num_adjs)], axis=-1)
        adj_cls = tf.transpose(adj_cls, (0, 2, 1))

        return ops_cls, adj_cls


class GraphAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim, num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs, eps_scale=0.01, dropout_rate=0.0):
        super(GraphAutoencoder, self).__init__()
        self.d_model = d_model
        self.num_ops = num_ops
        self.num_adjs = num_adjs
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.eps_scale = eps_scale
        self.encoder = GINEncoder(self.latent_dim, [128, 128, 128, 128], 'relu', dropout_rate)

        self.decoder = TransformerDecoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                               dff=dff, input_length=self.latent_dim, num_ops=num_ops, num_nodes=num_nodes, num_adjs=num_adjs,
                               dropout_rate=dropout_rate)

    def sample(self, mean, log_var, eps_scale=0.01):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(log_var * 0.5) * eps * eps_scale

    def call(self, inputs):
        latent_mean, latent_var = self.encoder(inputs)  # (batch_size, context_len, d_model)
        c = self.sample(latent_mean, latent_var, self.eps_scale)
        kl_loss = tf.reduce_sum(-0.5 * tf.reduce_sum(1 + latent_var - tf.square(latent_mean) - tf.exp(latent_var), axis=-1))

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


class GraphAutoencoderNVP(GraphAutoencoder):
    def __init__(self, nvp_config, latent_dim, num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs,
                 eps_scale=0.01, dropout_rate=0.0):
        super(GraphAutoencoderNVP, self).__init__(latent_dim, num_layers, d_model, num_heads, dff, num_ops,
                                                  num_nodes, num_adjs, eps_scale, dropout_rate)
        if nvp_config['inp_dim'] is None:
            nvp_config['inp_dim'] = latent_dim

        self.pad_dim = nvp_config['inp_dim'] - latent_dim
        self.nvp = NVP(**nvp_config)

    def call(self, inputs):
        ops_cls, adj_cls, kl_loss, latent_mean = super().call(inputs)
        latent_mean = tf.concat([latent_mean, tf.zeros((tf.shape(latent_mean)[0], self.pad_dim))], axis=-1)
        reg = self.nvp(latent_mean)
        return ops_cls, adj_cls, kl_loss, reg, latent_mean

    def inverse(self, z):
        return self.nvp.inverse(z)


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
