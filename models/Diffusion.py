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

        # the max diffusion_steps_t should be < t_dim / 2
        # if diffusion_steps_t=20, then the range of diffusion_steps_t is [0, 19]
        self.t_dim = diffusion_steps_t * 2

        self.down_list = [
            Down(input_dim // 2),
            Down(input_dim // 4),
            Down(input_dim // 6)
        ]

        self.up_list = [
            Up(input_dim // 4),
            Up(input_dim // 2),
            Up(input_dim)
        ]

    def unet_forward(self, x, t):
        for i in self.down_list:
            x = i(x, t)
        for i in self.up_list:
            x = i(x, t)
        return x

    def call(self, x, t):
        t = tf.constant(t, dtype=tf.float32)
        t = self.pos_encoding(t, self.t_dim)
        x = self.unet_forward(x, t)
        return x

    def pos_encoding(self, t, channels):
        t = tf.reshape(t, [-1, 1])
        inv_freq = 1.0 / (
                10000.0 ** (tf.range(0, channels, 2, dtype=tf.float32) / channels)
        )
        pos_enc_a = tf.sin(t * inv_freq)
        pos_enc_b = tf.cos(t * inv_freq)
        pos_enc = tf.concat([pos_enc_a, pos_enc_b], axis=-1)
        return pos_enc


class ConditionalDiffusionModel(DiffusionModel):
    def __init__(self, input_dim, diffusion_steps_t):
        super(ConditionalDiffusionModel, self).__init__(input_dim, diffusion_steps_t)
        self.acc_emb = tf.keras.layers.Dense(self.t_dim)

    def call(self, x, acc, t):
        t = tf.constant(t, dtype=tf.float32)
        t = self.pos_encoding(t, self.t_dim)
        acc = self.acc_emb(acc)
        x = self.unet_forward(x, t + acc)
        return x


if __name__ == '__main__':
    model = ConditionalDiffusionModel(input_dim=20, diffusion_steps_t=20)
    inp = tf.random.normal(shape=(2, 20))

    a = model(inp, tf.constant([[90.], [92.]]), tf.constant([[0.], [1.]]))
    print(a)