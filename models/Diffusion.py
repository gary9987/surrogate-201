import tensorflow as tf


def linear_schedule(timesteps=500, start=0.0001, end=0.02):
    '''
    return a tensor of a linear schedule
    '''
    return tf.linspace(start, end, timesteps)


def forward_diffusion_process(x_0, t):
    # precalculations
    betas = linear_schedule()
    alphas = 1 - betas

    alphas_cumprod = tf.math.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = tf.math.sqrt(alphas_cumprod)
    sqrt_oneminus_alphas_cumprod = tf.math.sqrt(1 - alphas_cumprod)

    noise = tf.random.normal(shape=x_0.shape)  # 回傳與X_0相同size的noise tensor，也就是reparameterization的epsilon

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]
    sqrt_oneminus_alphas_cumprod_t = sqrt_oneminus_alphas_cumprod[t]

    return sqrt_alphas_cumprod_t * x_0 + sqrt_oneminus_alphas_cumprod_t * noise, noise



class Down(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(Down, self).__init__()
        self.dense = tf.keras.layers.Dense(output_dim, activation='relu')
        #  For project position encoding
        self.emb_layer = tf.keras.layers.Dense(output_dim, activation='silu')

    def call(self, inputs, t):
        x = self.dense(inputs)
        t_emb = self.emb_layer(t)
        t_emb = tf.tile(t_emb, [tf.shape(x)[0], 1])
        x = x + t_emb
        return x


class Up(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(Up, self).__init__()
        self.dense = tf.keras.layers.Dense(output_dim, activation='relu')
        #  For project position encoding
        self.emb_layer = tf.keras.layers.Dense(output_dim, activation='silu')

    def call(self, inputs, t):
        x = self.dense(inputs)
        t_emb = self.emb_layer(t)
        x = x + t_emb
        return x


class DiffusionModel(tf.keras.Model):
    def __init__(self, input_dim, diffusion_steps_t):
        super(DiffusionModel, self).__init__()
        self.latent_dim = input_dim
        self.diffusion_steps_t = diffusion_steps_t
        self.t_dim = diffusion_steps_t // 2 + 1

        self.down_list = [
            Down(input_dim//2),
            Down(input_dim // 4),
            Down(input_dim // 6)
        ]

        self.up_list = [
            Up(input_dim // 4),
            Up(input_dim // 2),
            Up(input_dim)
        ]

    def call(self, x, t):
        t = tf.constant([t], dtype=tf.float32)
        t = tf.expand_dims(self.pos_encoding(t, self.t_dim), axis=0)
        for i in self.down_list:
            x = i(x, t)
        for i in self.up_list:
            x = i(x, t)
        return x

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000 ** (tf.range(0, channels, 2, dtype=tf.float32) / channels)
        )
        pos_enc_a = tf.math.sin(tf.transpose(t) * inv_freq)
        pos_enc_b = tf.math.cos(tf.transpose(t) * inv_freq)
        pos_enc = tf.concat([pos_enc_a, pos_enc_b], axis=-1)
        return pos_enc


if __name__ == '__main__':
    model = DiffusionModel(input_dim=20, diffusion_steps_t=20)
    inp = tf.random.normal(shape=(2, 20))

    a = model(inp, 1)
    print(a)