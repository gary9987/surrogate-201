import numpy as np
import tensorflow as tf
from models.TransformerAE import TransformerAutoencoder


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
        t = tf.cast(t, dtype=tf.float32)
        t = self.pos_encoding(t, self.t_dim)
        acc = self.acc_emb(acc)
        x = self.unet_forward(x, t + acc)
        return x


class TransformerAutoencoderDiffusion(TransformerAutoencoder):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_size, diffusion_steps, dropout_rate=0.0):
        super(TransformerAutoencoderDiffusion, self).__init__(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, input_size=input_size, dropout_rate=dropout_rate)
        self.diffusion_steps = diffusion_steps
        self.beta = self.prepare_noise_schedule()
        self.alpha = 1 - self.beta

        self.MSE = tf.keras.losses.MeanSquaredError()
        self.alpha_hat = tf.math.cumprod(self.alpha, axis=0)
        self.diffusion_model = ConditionalDiffusionModel(d_model * input_size, diffusion_steps)

    def prepare_noise_schedule(self):
        return linear_schedule(timesteps=self.diffusion_steps)

    def sample_t(self, batch_size):
        return tf.random.uniform(shape=[batch_size], minval=1, maxval=self.diffusion_steps, dtype=tf.int32)

    def add_noise(self, x, t):
        # Add noise to latent at instant t
        alpha_hat = tf.gather(self.alpha_hat, t)
        sqrt_alpha_hat = tf.math.sqrt(alpha_hat)
        sqrt_one_minus_alpha_hat  = tf.math.sqrt(1 - alpha_hat)
        sqrt_alpha_hat = tf.reshape(sqrt_alpha_hat, (tf.shape(x)[0], 1))
        sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, (tf.shape(x)[0], 1))
        noise = tf.random.normal(shape=x.shape, dtype=x.dtype)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def call(self, inputs, y):

        latent = self.encoder(inputs)  # (batch_size, context_len, d_model)
        flat_encoding = tf.reshape(latent, (tf.shape(latent)[0], -1))

        # Noise Loss
        t = self.sample_t(batch_size=tf.shape(inputs)[0])
        x_t, noise = self.add_noise(flat_encoding, t)
        pred_noise = self.diffusion_model(x_t, y, t)
        noise_loss = self.MSE(noise, pred_noise)

        # Reconstruction
        rec = self.decoder(latent)  # (batch_size, target_len, d_model)

        # Return the final output and the attention weights.
        return rec, noise_loss, flat_encoding

    def encode(self, inputs, training=True):
        encoding = self.encoder(inputs, training=training)
        flat_encoding = tf.reshape(encoding, (tf.shape(encoding)[0], -1))
        return flat_encoding

    @property
    def sqrt_recip_alphas(self):
        return tf.sqrt(1.0 / self.alpha)
    @property
    def sqrt_alpha_hat(self):
        return tf.sqrt(self.alpha_hat)
    @property
    def sqrt_one_minus_alphas_hat(self):
        return tf.sqrt(1.0 - self.alpha_hat)

    @property
    def posterior_variance(self):
        alpha_hat_prev = tf.pad(self.alpha_hat[:-1], [[1, 0]], constant_values=1.0)
        posterior_variance = self.beta * (1. - alpha_hat_prev) / (1. - self.alpha_hat)
        return posterior_variance

    def denoise(self, x, acc, t):
        """
        Denoise the noise latent at time t
        :param x: noise latent
        :param acc: conditional accuracy
        :param t: timestamp
        :return: denoised latent
        """
        beta = tf.gather(self.beta, t)
        sqrt_one_minus_alpha_hat = tf.gather(self.sqrt_one_minus_alphas_hat, t)
        sqrt_recip_alpha = tf.gather(self.sqrt_recip_alphas, t)
        posterior_variance = tf.gather(self.posterior_variance, t)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alpha * (x - beta * self.diffusion_model(x, acc, t) / sqrt_one_minus_alpha_hat)
        noise = tf.sqrt(posterior_variance) * tf.random.normal(shape=x.shape, dtype=x.dtype)
        return model_mean + noise


if __name__ == '__main__':
    '''
    model = ConditionalDiffusionModel(input_dim=20, diffusion_steps_t=20)
    inp = tf.random.normal(shape=(2, 20))

    a = model(inp, tf.constant([[90.], [92.]]), tf.constant([[0.], [1.]]))
    print(a)
    '''
    model = TransformerAutoencoderDiffusion(num_layers=3,
                                            d_model=4,
                                            num_heads=3,
                                            dff=128,
                                            input_size=5,
                                            diffusion_steps=1000)
    inp = tf.random.normal(shape=(2, 5))

    a = model(inp, tf.constant([[90.], [92.]]))

    x = model.denoise(tf.random.normal(shape=(2, 20)), tf.constant([[90.], [92.]]), tf.constant([[0], [1]]))
    print(a, x)