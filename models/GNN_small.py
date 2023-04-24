from spektral.layers import GlobalMaxPool
from invertible_neural_networks.flow import NVP
from models.GNN import GINEncoder, GraphAutoencoder
from models.TransformerAE import Decoder
from models.base import PositionalEmbedding
import tensorflow as tf


class GINEncoderSmall(GINEncoder):
    def __init__(self, n_hidden, mlp_hidden, activation: str, dropout=0.):
        super(GINEncoderSmall, self).__init__(n_hidden, mlp_hidden, activation, dropout)
        self.pool = GlobalMaxPool()

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.bn(out)
        out = self.pool(out)
        out = self.dropout(out)
        mean = self.mean(out)
        var = self.var(out)
        return mean, var


class TransformerDecoderSmall(Decoder):
    def __init__(self, num_layers, d_model, num_heads, dff, input_length, num_ops, num_nodes, num_adjs, dropout_rate=0.0):
        super(TransformerDecoderSmall, self).__init__(num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs,
                                                 dropout_rate)
        self.pos_embedding = PositionalEmbedding(d_model=d_model, input_length=input_length)
        self.adj_cls = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(num_adjs, 1)),
            tf.keras.layers.Dense(2, activation='softmax'),
        ])

    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x)

        flatten_x = tf.reshape(x, (tf.shape(x)[0], -1))
        ops_cls = tf.stack([self.ops_cls[i](flatten_x) for i in range(self.num_nodes)], axis=-1)
        ops_cls = tf.transpose(ops_cls, (0, 2, 1))

        adj_cls = self.adj_cls(flatten_x)
        return ops_cls, adj_cls


class GraphAutoEncoderSmall(GraphAutoencoder):
    def __init__(self, latent_dim, num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs, eps_scale=0.01, dropout_rate=0.0):
        super(GraphAutoEncoderSmall, self).__init__(latent_dim, num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs, eps_scale, dropout_rate)
        self.encoder = GINEncoderSmall(self.latent_dim, [128] * 4, 'relu', dropout_rate)
        self.decoder = TransformerDecoderSmall(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                               input_length=self.latent_dim, num_ops=num_ops, num_nodes=num_nodes, num_adjs=num_adjs, dropout_rate=dropout_rate)

    def call(self, inputs):
        latent_mean, latent_var = self.encoder(inputs)  # (batch_size, context_len, d_model)
        c = self.sample(latent_mean, latent_var, self.eps_scale)
        kl_loss = tf.reduce_mean(
            -0.5 * tf.reduce_sum(1 + latent_var - tf.square(latent_mean) - tf.exp(latent_var), axis=-1))

        ops_cls, adj_cls = self.decoder(c)  # (batch_size, target_len, d_model)

        # Return the final output
        return ops_cls, adj_cls, kl_loss, latent_mean


class GraphAutoencoderNVPSmall(GraphAutoEncoderSmall):
    def __init__(self, nvp_config, latent_dim, num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs,
                 eps_scale=0.01, dropout_rate=0.0):
        super(GraphAutoencoderNVPSmall, self).__init__(latent_dim, num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs,
                 eps_scale, dropout_rate)
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