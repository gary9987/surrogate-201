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



class DiffusionModel(tf.keras.Model):
    def __init__(self, input_dim, diffusion_steps):
        super(DiffusionModel, self).__init__()
        self.latent_dim = input_dim
        self.diffusion_steps = diffusion_steps

        self.alpha = self.add_weight(shape=(1,), initializer='random_normal')
        self.beta = self.add_weight(shape=(1,), initializer='random_normal')

        self.down = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(input_dim//2, activation='relu'),
            tf.keras.layers.Dense(input_dim//2, activation='relu'),
            tf.keras.layers.Dense(input_dim//4),
        ])

        self.up = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim//4,)),
            tf.keras.layers.Dense(input_dim//2, activation='relu'),
            tf.keras.layers.Dense(input_dim//2, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='sigmoid'),
        ])

    def call(self, x):
        z = self.down(x)
        for t in range(self.diffusion_steps):
            epsilon = tf.random.normal(tf.shape(z))
            z = (1 - self.alpha) * z + self.beta * epsilon
            z = z + self.alpha * self.up(z)
        x_recon = self.up(z)
        return x_recon
