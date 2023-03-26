import numpy.random
import tensorflow as tf
import numpy as np


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

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, target_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dec_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.final_layer = tf.keras.layers.Dense(2)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x)
        # Shape `(batch_size, seq_len, d_model)`.

        x = self.final_layer(x)
        x = tf.nn.softmax(x, axis=-1)
        return x


class TransformerAutoencoder(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                               dff=dff, input_size=input_size, dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                               dff=dff, target_size=input_size, dropout_rate=dropout_rate)

    def call(self, inputs):
        context = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)
        x = self.decoder(context)  # (batch_size, target_len, d_model)

        # Return the final output and the attention weights.
        return x

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, inputs):
        return self.decoder(inputs)

if __name__ == '__main__':
    model = TransformerAutoencoder(num_layers=3, d_model=64, num_heads=3, dff=128, input_size=28)
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
