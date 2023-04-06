import argparse
from pathlib import Path
from tensorflow.python.keras.callbacks import CSVLogger, EarlyStopping
from tqdm import tqdm
from datasets.transformation import ReshapeYTransform, OnlyValidAccTransform
from invertible_neural_networks.flow import MSE, MMD_multiscale
from models.TransformerAE import CustomSchedule
from models.Diffusion import TransformerAutoencoderDiffusion
import tensorflow as tf
import logging
import sys, os, datetime
from datasets.nb201_dataset import NasBench201Dataset
from datasets.utils import train_valid_test_split_dataset, to_latent_feature_data
import numpy as np


now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join("logs", Path(__file__).stem, now_time)
os.makedirs(logdir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logdir, f'train.log'), level=logging.INFO, force=True, filemode='w')
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


parser = argparse.ArgumentParser(description='train TAE+Diffusion')
parser.add_argument('--train_sample_amount', type=int, default=1000, help='Number of samples to train (default: 900)')
parser.add_argument('--valid_sample_amount', type=int, default=100, help='Number of samples to train (default: 100)')
args = parser.parse_args()


random_seed = 0
tf.random.set_seed(random_seed)


class Trainer(tf.keras.Model):
    def __init__(self, model):
        super(Trainer, self).__init__()
        self.model = model
        self.rec_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.noise_loss_fn = tf.keras.losses.MeanSquaredError()

    def train_step(self, data):
        x_batch_train, y_batch_train = data

        non_nan_idx = tf.reshape(tf.where(~tf.math.is_nan(tf.reduce_sum(y_batch_train, axis=-1))), -1)

        with tf.GradientTape() as tape:
            rec_logits, pred_noise, label_noise = self.model(x_batch_train, y_batch_train)  # Logits for this minibatch
            noise_loss = self.noise_loss_fn(label_noise, pred_noise)
            rec_loss = self.rec_loss_fn(x_batch_train, rec_logits)
            loss = noise_loss + rec_loss
            #rec_loss = self.rec_loss_fn(x_batch_train, rec_logits)
            # To avoid nan loss when batch size is small
            # if tf.shape(non_nan_idx)[0] != 0:

        grads = tape.gradient(loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return {'loss': loss, 'rec_loss': rec_loss, 'noise_loss': noise_loss}

    def test_step(self, data):
        x_batch_train, y_batch_train = data

        rec_logits, pred_noise, label_noise = self.model(x_batch_train, y_batch_train)  # Logits for this minibatch
        noise_loss = self.noise_loss_fn(label_noise, pred_noise)
        rec_loss = self.rec_loss_fn(x_batch_train, rec_logits)
        loss = noise_loss + rec_loss
        return {'loss': loss, 'rec_loss': rec_loss, 'noise_loss': noise_loss}

if __name__ == '__main__':
    is_only_validation_data = True
    label_epochs = 200

    d_model = 4
    dropout_rate = 0.0
    dff = 512
    num_layers = 3
    num_heads = 3
    diffusion_steps = 500
    input_size = 120

    batch_size = 256
    train_epochs = 1000
    patience = 50

    # 15624
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(label_epochs), seed=777),
                                              ratio=[0.9, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    #datasets['train'] = datasets['train'][:args.train_sample_amount]
    #datasets['valid'] = datasets['valid'][:args.valid_sample_amount]

    for key in datasets:
        if is_only_validation_data:
            datasets[key].apply(OnlyValidAccTransform())
        else:
            datasets[key].apply(ReshapeYTransform())


    x_train, y_train = to_latent_feature_data(datasets['train'], -1)
    x_valid, y_valid = to_latent_feature_data(datasets['valid'], -1)

    #learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-5)

    model = TransformerAutoencoderDiffusion(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                            input_size=input_size, diffusion_steps=diffusion_steps,
                                            dropout_rate=dropout_rate)

    loader = {'train': tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(batch_size=batch_size).repeat(),
              'valid': tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size=batch_size)}

    #model.encoder.trainable = False
    #model.decoder.trainable = False
    trainer = Trainer(model)
    trainer.compile(optimizer=optimizer, run_eagerly=True)

    print(len(datasets['train']))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    trainer.fit(loader['train'],
                validation_data=loader['valid'],
                batch_size=batch_size,
                epochs=train_epochs,
                steps_per_epoch=len(datasets['train']) // batch_size,
                callbacks=[CSVLogger(os.path.join(logdir, "learning_curve.log")),
                           tensorboard_callback,
                           #EarlyStopping(monitor='val_noise_loss', patience=patience, restore_best_weights=True)
                           ]
                )

    model.save_weights(os.path.join(logdir, 'modelTAE_weights'))