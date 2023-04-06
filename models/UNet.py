import tensorflow as tf


class DoubleConv(tf.keras.layers.Layer):
    def __init__(self, out_c):
        super().__init__()
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(out_c, kernel_size=3, padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(out_c, kernel_size=3, padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Down(tf.keras.layers.Layer):
    def __init__(self, out_c):
        super().__init__()
        self.down = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            DoubleConv(out_c, first_residual=True)
        ])

        self.emb_layer = tf.keras.Sequential([
            tf.keras.layers.Activation(tf.nn.silu),
            tf.keras.layers.Dense(out_c)
        ])

    def call(self, x, t):
        x = self.down(x)
        t_emb =  self.emb(t)[:, None, None, :]
        return x + t_emb


class Up(tf.keras.layers.Layer):
    def __init__(self, out_c):
        super().__init__()
        self.up = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv = DoubleConv(out_c)
        self.emb_layer = tf.keras.Sequential([
            tf.keras.layers.Activation(tf.nn.silu),
            tf.keras.layers.Dense(out_c)
        ])

    def call(self, x, skip_x, t):
        x = self.up(x)
        x = tf.concat([skip_x, x], axis=-1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None]
        #emb = tf.broadcast_to(emb, shape=[emb.shape[0], emb.shape[1], x.shape[2], x.shape[3]])
        return x + emb


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d_model)
        self.ln = tf.keras.layers.LayerNormalization()
        self.ff_self = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Activation(tf.nn.gelu),
            tf.keras.layers.Dense(d_model)
        ])

    def call(self, x):
        batch_size, w, h =tf.shape(x)[0],  tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, [batch_size, w * h, -1])
        x_ln = self.ln(x)
        attention_value = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        attention_value = tf.reshape(attention_value, [batch_size, w, h, -1])
        return attention_value


class UNet(tf.keras.Model):
    def __init__(self, c_out, time_dim=128):
        super().__init__()

        self.time_dim = time_dim
        self.inc = DoubleConv(64)

        self.down1 = Down(128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(512)
        self.bot2 = DoubleConv(512)
        self.bot3 = DoubleConv(256)

        self.up1 = Up(128)
        self.sa4 = SelfAttention(16)
        self.up2 = Up(64)
        self.sa5 = SelfAttention(32)
        self.up3 = Up(64)
        self.sa6 = SelfAttention(64)

        self.outc = tf.keras.layers.Conv2D(c_out, kernel_size=1)


    def call(self, x, t):
        # initial conv
        x1 = self.inc(x)

        # Down
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # Bottle neck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # Up
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        # Output
        output = self.outc(x)
        return output

