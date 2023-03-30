import argparse
from tensorflow.python.keras.callbacks import CSVLogger, EarlyStopping
from tqdm import tqdm
from datasets.transformation import ReshapeYTransform, OnlyValidAccTransform
from invertible_neural_networks.flow import MSE, MMD_multiscale
from models.TransformerAE import TransformerAutoencoderNVP, CustomSchedule
import tensorflow as tf
import logging
import sys, os, datetime
from datasets.nb201_dataset import NasBench201Dataset
from datasets.utils import train_valid_test_split_dataset
import numpy as np
from evalTAE import inverse_from_acc


logging.basicConfig(filename='train.log', level=logging.INFO, force=True, filemode='w')
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


parser = argparse.ArgumentParser(description='train TAE')
parser.add_argument('--train_sample_amount', type=int, default=1000, help='Number of samples to train (default: 900)')
parser.add_argument('--valid_sample_amount', type=int, default=100, help='Number of samples to train (default: 100)')
args = parser.parse_args()


random_seed = 0
tf.random.set_seed(random_seed)


def to_NVP_data(graph_dataset, z_dim, reg_size):
    features = []
    y_list = []
    if reg_size == -1:
        nan_size = 0
    else:
        nan_size = len(graph_dataset) - reg_size

    to_nan_idx = np.random.choice(range(len(graph_dataset)), nan_size, replace=False)

    for data in graph_dataset:
        x = np.reshape(data.x, -1)
        a = np.reshape(data.a, -1)
        features.append(np.concatenate([x, a]))

        #z = np.reshape(np.random.multivariate_normal([0.]*z_dim, np.eye(z_dim), 1), -1)
        y = np.array([data.y[-1] / 100.0])
        y_list.append(y)

    y_list = np.array(y_list)
    z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim), y_list.shape[0])
    y_list = np.concatenate([z, y_list], axis=-1)
    y_list[to_nan_idx, :] = np.nan

    return np.array(features).astype(np.float32), np.array(y_list).astype(np.float32)


class Trainer(tf.keras.Model):
    def __init__(self, model, rec_loss_fn, reg_loss_fn,
          x_dim, y_dim, z_dim):
        super(Trainer, self).__init__()
        self.model = model
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        # For rec loss weight
        self.w0 = 1.
        # For reg loss weight
        self.w1 = 1
        # For latent loss weight
        self.w2 = 1.
        # For rev loss weight
        self.w3 = 1.

        self.rec_loss_fn = rec_loss_fn
        self.reg_loss_fn = reg_loss_fn
        self.loss_latent = MMD_multiscale
        self.loss_backward = MSE

    def train_step(self, data):
        x_batch_train, y_batch_train = data
        y = y_batch_train[:, -y_dim:]
        z = y_batch_train[:, :z_dim]
        y_short = tf.concat([z, y], axis=-1)
        non_nan_idx = tf.reshape(tf.where(~tf.math.is_nan(tf.reduce_sum(y_batch_train, axis=-1))), -1)

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        # Forward loss and AE Reconstruct loss
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            rec_logits, y_out, x_encoding = self.model(x_batch_train, training=True)  # Logits for this minibatch

            rec_loss = self.rec_loss_fn(x_batch_train, rec_logits)
            # To avoid nan loss when batch size is small
            if tf.shape(non_nan_idx)[0] != 0:
                reg_loss = self.reg_loss_fn(tf.gather(y_batch_train[:, z_dim:], non_nan_idx),
                                            tf.gather(y_out[:, z_dim:], non_nan_idx))
                latent_loss = self.loss_latent(tf.gather(y_short, non_nan_idx),
                                            tf.gather(tf.concat([y_out[:, :z_dim], y_out[:, -y_dim:]], axis=-1), non_nan_idx))  # * x_batch_train.shape[0]
            else:
                reg_loss = 0.
                latent_loss = 0.

            forward_loss = self.w0 * rec_loss + self.w1 * reg_loss + self.w2 * latent_loss

        grads = tape.gradient(forward_loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        '''
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(forward_loss, model.trainable_weights)
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        '''

        # To avoid nan loss when batch size is small
        if tf.shape(non_nan_idx)[0] == 0:
            return {'total_loss': forward_loss, 'rec_loss': rec_loss, 'reg_loss': reg_loss, 'latent_loss': latent_loss, 'rev_loss': 0}

        # Backward loss
        with tf.GradientTape() as tape:
            self.model.decoder.trainable = False
            _, _, x_encoding = self.model(x_batch_train, training=True)  # Logits for this minibatch
            x_rev = self.model.inverse(tf.gather(y_batch_train, non_nan_idx))
            rev_loss = self.loss_backward(x_rev, tf.gather(x_encoding, non_nan_idx))  # * x_batch_train.shape[0]
            loss = self.w3 * rev_loss

        grads = tape.gradient(loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.model.decoder.trainable = True

        return {'total_loss': forward_loss + loss,
                'rec_loss': rec_loss,
                'reg_loss': reg_loss,
                'latent_loss': latent_loss,
                'rev_loss': rev_loss}

    def test_step(self, data):
        x_batch_train, y_batch_train = data
        y = y_batch_train[:, -y_dim:]
        z = y_batch_train[:, :z_dim]
        y_short = tf.concat([z, y], axis=-1)

        rec_logits, y_out, x_encoding = self.model(x_batch_train, training=False)  # Logits for this minibatch

        rec_loss = self.rec_loss_fn(x_batch_train, rec_logits)
        reg_loss = self.reg_loss_fn(y_batch_train[:, z_dim:], y_out[:, z_dim:])
        latent_loss = self.loss_latent(y_short, tf.concat([y_out[:, :z_dim], y_out[:, -y_dim:]], axis=-1))  # * x_batch_train.shape[0]
        x_rev = self.model.inverse(y_batch_train)
        rev_loss = self.loss_backward(x_rev, x_encoding)  # * x_batch_train.shape[0]

        return {'total_loss': self.w0 * rec_loss + self.w1 * reg_loss + self.w2 * latent_loss + self.w3 * rev_loss,
                'rec_loss': rec_loss,
                'reg_loss': reg_loss,
                'latent_loss': latent_loss,
                'rev_loss': rev_loss}


if __name__ == '__main__':
    is_only_validation_data = True
    label_epochs = 200

    d_model = 4
    dropout_rate = 0.0
    dff = 512
    num_layers = 3
    num_heads = 3
    nvp_config = {
        'n_couple_layer': 3,
        'n_hid_layer': 3,
        'n_hid_dim': 128,
        'name': 'NVP'
    }

    batch_size = 512
    train_epochs = 1000
    patience = 100

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

    x_dim = (np.reshape(datasets['train'][0].x, -1).shape[-1] + np.reshape(datasets['train'][0].a, -1).shape[-1]) * d_model  # 120 * 4
    y_dim = 1  # 1
    z_dim = x_dim - y_dim  # 479

    x_train, y_train = to_NVP_data(datasets['train'], z_dim, args.train_sample_amount)
    x_valid, y_valid = to_NVP_data(datasets['valid'], z_dim, -1)

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    #tf.keras.losses.Reduction.SUM
    rec_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    reg_loss_fn = tf.keras.losses.MeanSquaredError()

    model = TransformerAutoencoderNVP(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, input_size=x_train.shape[-1], nvp_config=nvp_config, dropout_rate=dropout_rate)

    loader = {'train': tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(batch_size=batch_size).repeat(),
              'valid': tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size=batch_size)}

    trainer = Trainer(model, rec_loss_fn, reg_loss_fn, x_dim, y_dim, z_dim)
    trainer.compile(optimizer='adam', run_eagerly=True)
    print(len(datasets['train']))

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    trainer.fit(loader['train'],
                validation_data=loader['valid'],
                batch_size=batch_size,
                epochs=train_epochs,
                steps_per_epoch=len(datasets['train']) // batch_size,
                callbacks=[CSVLogger(f"learning_curve.log"),
                           tensorboard_callback,
                           EarlyStopping(monitor='val_total_loss', patience=patience, restore_best_weights=True)]
                )

    model.save_weights('modelTAE_weights')

    x, y = np.array([x_valid[0]]), np.array([y_valid[0]])
    rec, reg, flat_encoding = model(x)
    rec = tf.argmax(rec, axis=-1)
    z = np.random.multivariate_normal([1.]*z_dim, np.eye(z_dim), 1)
    y_new = np.concatenate([z, y[:, -y_dim:]], axis=-1)
    print(x)
    print(rec.numpy())
    print(y[:, -y_dim:], reg[:, -y_dim:])
    rev_x = model.inverse(y_new.astype(np.float32))
    print(MSE(rev_x, flat_encoding))

    decode_x = model.decode(tf.reshape(rev_x, (1, -1, d_model)))
    print(decode_x)

    print(inverse_from_acc(model, num_sample_z=100, z_dim=z_dim, to_inv_acc=1.0))