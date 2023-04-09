import numpy.random
import tensorflow as tf
import numpy as np
from invertible_neural_networks.flow import NVP


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, input_length):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(2, d_model, mask_zero=False, input_length=input_length)
        self.pos_encoding = positional_encoding(length=input_length, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, input_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(d_model=d_model, input_length=input_size)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.mean_emb = tf.keras.layers.Dense(d_model)
        self.var_emb = tf.keras.layers.Dense(d_model)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        x_mean = self.mean_emb(x)
        x_var = self.var_emb(x)
        return x_mean, x_var  # Shape `(batch_size, seq_len, d_model)`.


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, num_ops, num_nodes, num_adjs, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_ops = num_ops
        self.num_nodes = num_nodes
        self.num_adjs = num_adjs

        self.dec_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.ops_cls = [
            tf.keras.layers.Dense(num_ops, activation='softmax')
            for _ in range(num_nodes)
        ]
        self.adj_cls = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x)
        # Shape `(batch_size, seq_len, d_model)`.

        flatten_x = tf.reshape(x, (tf.shape(x)[0], -1))
        ops_cls = tf.stack([self.ops_cls[i](flatten_x) for i in range(self.num_nodes)], axis=-1)
        ops_cls = tf.transpose(ops_cls, (0, 2, 1))  # Shape `(batch_size, num_nodes, num_ops)`
        adj_cls = self.adj_cls(x[:, -self.num_adjs:, :])  # Shape `(batch_size, num_adjs, 2)`
        return ops_cls, adj_cls


class TransformerAutoencoder(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_size, num_ops, num_nodes, num_adjs, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_ops = num_ops
        self.num_adjs = num_adjs
        self.num_nodes = num_nodes
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                               dff=dff, input_size=input_size, dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                               dff=dff, num_ops=num_ops, num_nodes=num_nodes, num_adjs=num_adjs,
                               dropout_rate=dropout_rate)

    def sample(self, mean, log_var, eps_scale=0.01):
        eps = tf.random.normal(shape=mean.shape)
        return mean + tf.exp(log_var * 0.5) * eps * eps_scale

    def call(self, inputs):
        latent_mean, latent_var = self.encoder(inputs)  # (batch_size, context_len, d_model)
        c = self.sample(latent_mean, latent_var)
        kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + latent_var - tf.square(latent_mean) - tf.exp(latent_var), axis=-1))

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


class TransformerAutoencoderReg(TransformerAutoencoder):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_size, dropout_rate=0.1):
        super(TransformerAutoencoderReg, self).__init__(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, input_size=input_size, dropout_rate=dropout_rate)
        self.reg_mlp = RegMLP(h_dim=dff, target_dim=1, dropout_rate=dropout_rate)

    def call(self, inputs):
        ops_cls, adj_cls, kl_loss, latent_mean = super().call(inputs)
        # Regression
        reg = self.reg_mlp(tf.reshape(latent_mean, (tf.shape(latent_mean)[0], -1)))  # (batch_size, 1)

        # Return the final output and the attention weights.
        return ops_cls, adj_cls, kl_loss, reg, latent_mean


class TransformerAutoencoderNVP(TransformerAutoencoder):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_size, num_ops, num_nodes, num_adjs, nvp_config, dropout_rate=0.1):
        super(TransformerAutoencoderNVP, self).__init__(num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                                                        dff=dff, input_size=input_size, num_ops=num_ops,
                                                        num_nodes=num_nodes,num_adjs=num_adjs,
                                                        dropout_rate=dropout_rate)
        self.nvp = NVP(inp_dim=d_model * input_size, **nvp_config)

    def call(self, inputs):
        ops_cls, adj_cls, kl_loss, latent_mean = super().call(inputs)
        flat_encoding = tf.reshape(latent_mean, (tf.shape(latent_mean)[0], -1))
        # Regression
        reg = self.nvp(flat_encoding)  # (batch_size, 1)

        # Return the final output and the attention weights.
        return ops_cls, adj_cls, kl_loss, reg, flat_encoding

    def encode(self, inputs, training=True):
        encoding = super().encode(inputs)
        flat_encoding = tf.reshape(encoding, (tf.shape(encoding)[0], -1))
        return flat_encoding

    def inverse(self, z):
        return self.nvp.inverse(z)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class RegMLP(tf.keras.layers.Layer):
    def __init__(self, h_dim, target_dim=1, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(h_dim, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(h_dim // 2, activation='relu'),
            tf.keras.layers.Dense(target_dim)
        ])

    def call(self, x):
        x = self.seq(x)
        return x


if __name__ == '__main__':
    model = TransformerAutoencoder(num_layers=3, d_model=16, num_heads=3, dff=128, input_size=28)
    model.compile('adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))
    train = np.random.randint(0, 2, (1024, 28))

    train_data = tf.data.Dataset.from_tensor_slices((train, train))
    train_data = train_data.shuffle(1024).batch(5)

    model.fit(train_data, epochs=10)

    test = np.random.randint(0, 2, (1, 28))
    print(test)
    latent = model.encode(test)

    pred = model.decode(latent)
    print(tf.argmax(pred, axis=-1))
