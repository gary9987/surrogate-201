import numpy as np
import tensorflow as tf
from models.GNN import GraphAutoencoder
from models.base import PositionalEmbedding
from models.TransformerAE import TransformerAutoencoder, EncoderLayer
from models.UNet import UNet


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


class NoisePredictor(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, latent_dim,
                 dff, dropout_rate=0.0):
        super(NoisePredictor, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(d_model=d_model, input_length=latent_dim)
        self.dec_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dense = tf.keras.layers.Dense(1)
        self.emb_layer = tf.keras.Sequential([
            tf.keras.layers.Activation(tf.nn.silu),
            tf.keras.layers.Dense(d_model)
        ])

    def call(self, x, t):
        t = self.emb_layer(t)[:, None, :]
        x = self.pos_embedding(x) + t
        for i in range(self.num_layers):
            x = self.dec_layers[i](x)
        x = self.dense(x)
        return tf.squeeze(x)


class DiffusionModel(tf.keras.layers.Layer):
    def __init__(self, input_dim, diffusion_steps_t):
        super(DiffusionModel, self).__init__()
        self.latent_dim = input_dim
        self.diffusion_steps_t = diffusion_steps_t

        # the max diffusion_steps_t should be < t_dim / 2
        # if diffusion_steps_t=20, then the range of diffusion_steps_t is [0, 19]
        self.t_dim = diffusion_steps_t * 2
        #self.unet = UNet(input_dim, self.t_dim)
        self.unet = NoisePredictor(num_layers=3, d_model=32, num_heads=3, dff=256, latent_dim=input_dim)

    def unet_forward(self, x, t):
        x = self.unet(x, t)
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


class GraphAutoencoderDiffusion(GraphAutoencoder):
    def __init__(self, latent_dim, num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs, diffusion_steps,
                 eps_scale=0.01, dropout_rate=0.0):
        super(GraphAutoencoderDiffusion, self).__init__(latent_dim, num_layers, d_model, num_heads, dff, num_ops,
                                                  num_nodes, num_adjs, eps_scale, dropout_rate)

        self.diffusion = ConditionalDiffusionModel(latent_dim, diffusion_steps)
        self.diffusion_steps = diffusion_steps
        self.beta = linear_schedule(timesteps=self.diffusion_steps)
        self.alpha = 1 - self.beta
        self.alpha_hat = tf.math.cumprod(self.alpha, axis=0)

    def add_noise(self, x, t):
        # Add noise to latent at instant t
        alpha_hat = tf.gather(self.alpha_hat, t)
        sqrt_alpha_hat = tf.math.sqrt(alpha_hat)
        sqrt_one_minus_alpha_hat  = tf.math.sqrt(1 - alpha_hat)
        sqrt_alpha_hat = tf.reshape(sqrt_alpha_hat, (tf.shape(x)[0], 1))
        sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, (tf.shape(x)[0], 1))
        noise = tf.random.normal(shape=tf.shape(x), dtype=x.dtype)
        # reshape to [batch_size, 1, 1, 1, ...]
        sqrt_alpha_hat = tf.reshape(sqrt_alpha_hat, [tf.shape(x)[0]] + (len(tf.shape(x)) - 1) * [1])
        sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, [tf.shape(x)[0]] + (len(tf.shape(x)) - 1) * [1])
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_t(self, batch_size):
        return tf.random.uniform(shape=[batch_size], minval=1, maxval=self.diffusion_steps, dtype=tf.int32)

    def call(self, inputs, y):
        ops_cls, adj_cls, kl_loss, latent_mean = super().call(inputs)
        # Noise Loss
        t = self.sample_t(batch_size=tf.shape(y)[0])
        x_t, noise = self.add_noise(latent_mean, t)
        pred_noise = self.diffusion(x_t, y, t)

        return ops_cls, adj_cls, kl_loss, latent_mean, pred_noise, noise

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
        shape = [tf.shape(x)[0]] + (len(tf.shape(x)) - 1) * [1]

        beta = tf.reshape(beta, shape)
        sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, shape)
        sqrt_recip_alpha = tf.reshape(sqrt_recip_alpha, shape)
        posterior_variance = tf.reshape(posterior_variance, shape)
        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alpha * (x - beta * self.diffusion_model(x, acc, t) / sqrt_one_minus_alpha_hat)
        noise = tf.sqrt(posterior_variance) * tf.random.normal(shape=x.shape, dtype=x.dtype)
        return model_mean + noise


class TransformerAutoencoderDiffusion(TransformerAutoencoder):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_size,  num_ops, num_nodes, num_adjs, diffusion_steps, dropout_rate=0.0):
        super(TransformerAutoencoderDiffusion, self).__init__(num_layers=num_layers, d_model=d_model,
                                                              num_heads=num_heads, dff=dff, input_size=input_size,
                                                              num_ops=num_ops, num_nodes=num_nodes, num_adjs=num_adjs,
                                                              dropout_rate=dropout_rate)
        self.diffusion_steps = diffusion_steps
        self.beta = self.prepare_noise_schedule()
        self.alpha = 1 - self.beta

        self.alpha_hat = tf.math.cumprod(self.alpha, axis=0)
        self.diffusion_model = ConditionalDiffusionModel(d_model, diffusion_steps)

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
        # reshape to [batch_size, 1, 1, 1, ...]
        sqrt_alpha_hat = tf.reshape(sqrt_alpha_hat, [tf.shape(x)[0]] + (len(tf.shape(x)) - 1) * [1])
        sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, [tf.shape(x)[0]] + (len(tf.shape(x)) - 1) * [1])
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def call(self, inputs, y):
        ops_cls, adj_cls, kl_loss, latent_mean = super().call(inputs)
        latent_img = tf.reshape(latent_mean, [tf.shape(latent_mean)[0], int(int(tf.shape(latent_mean)[1]) ** 0.5), -1, self.d_model])

        # Noise Loss
        t = self.sample_t(batch_size=tf.shape(inputs)[0])
        x_t, noise = self.add_noise(latent_img, t)
        pred_noise = self.diffusion_model(x_t, y, t)

        return ops_cls, adj_cls, pred_noise, noise, kl_loss

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
        shape = [tf.shape(x)[0]] + (len(tf.shape(x)) - 1) * [1]

        beta = tf.reshape(beta, shape)
        sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, shape)
        sqrt_recip_alpha = tf.reshape(sqrt_recip_alpha, shape)
        posterior_variance = tf.reshape(posterior_variance, shape)
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
                                            input_size=120,
                                            diffusion_steps=1000)

    inp = tf.random.normal(shape=(2, 120))
    a, pred_noise, _ = model(inp, tf.constant([[90.], [92.]]))

    x = model.denoise(tf.random.normal(shape=(2, 10, 12, 4)), tf.constant([[90.], [92.]]), tf.constant([[0], [1]]))
    print(a.shape, pred_noise.shape, x.shape)