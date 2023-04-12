import argparse
from tensorflow.python.keras.callbacks import CSVLogger, EarlyStopping
from datasets.transformation import ReshapeYTransform, OnlyValidAccTransform, OnlyFinalAcc, LabelScale_NasBench101
from invertible_neural_networks.flow import MSE, MMD_multiscale
from models.GNN import GraphAutoencoder, GraphAutoencoderNVP
from models.TransformerAE import TransformerAutoencoderNVP, CustomSchedule
import tensorflow as tf
import logging
import sys, os, datetime
from datasets.nb201_dataset import NasBench201Dataset
from datasets.utils import train_valid_test_split_dataset
from spektral.data import BatchLoader

now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join("logs", now_time)
os.makedirs(logdir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logdir, f'train.log'), level=logging.INFO, force=True, filemode='w')
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

num_ops = 7
num_nodes = 8
num_adjs = 64


def cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls):
    ops_label, adj_label = x_batch_train
    adj_label = tf.reshape(adj_label, [tf.shape(adj_label)[0], -1])
    ops_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(ops_label, ops_cls)
    adj_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(adj_label, adj_cls)
    return ops_loss, adj_loss


def to_undiredted_adj(adj):
    undirected_adj = tf.cast(tf.cast(adj, tf.int32) | tf.cast(tf.transpose(adj, perm=[0, 2, 1]), tf.int32), tf.float32)
    return undirected_adj


class Trainer1(tf.keras.Model):
    def __init__(self, model: GraphAutoencoder):
        super(Trainer1, self).__init__()
        self.model = model
        self.ce_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.sce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.ops_weight = 1
        self.adj_weight = 1
        self.kl_weight = 0.1


    def train_step(self, data):
        x_batch_train, _ = data
        undirected_x_batch_train = (x_batch_train[0], to_undiredted_adj(x_batch_train[1]))

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        # Forward loss and AE Reconstruct loss
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            ops_cls, adj_cls, kl_loss, _ = self.model(undirected_x_batch_train,
                                                      training=True)  # Logits for this minibatch
            ops_loss, adj_loss = cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls)
            rec_loss = self.ops_weight * ops_loss + self.adj_weight * adj_loss + self.kl_weight * kl_loss

        grads = tape.gradient(rec_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return {'rec_loss': rec_loss, 'ops_loss': ops_loss, 'adj_loss': adj_loss, 'kl_loss': kl_loss}

    def test_step(self, data):
        x_batch_train, _ = data
        undirected_x_batch_train = (x_batch_train[0], to_undiredted_adj(x_batch_train[1]))
        ops_cls, adj_cls, kl_loss, _ = self.model(undirected_x_batch_train, training=True)  # Logits for this minibatch
        ops_loss, adj_loss = cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls)
        rec_loss = self.ops_weight * ops_loss + self.adj_weight * adj_loss + self.kl_weight * kl_loss
        return {'rec_loss': rec_loss, 'ops_loss': ops_loss, 'adj_loss': adj_loss, 'kl_loss': kl_loss}


class Trainer2(tf.keras.Model):
    def __init__(self, model: TransformerAutoencoderNVP, x_dim, y_dim, z_dim, finetune=False):
        super(Trainer2, self).__init__()
        self.model = model
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.finetune = finetune

        # For rec loss weight
        self.w0 = 10.
        # For reg loss weight
        self.w1 = 5.
        # For latent loss weight
        self.w2 = 1.
        # For rev loss weight
        self.w3 = 10.

        self.reg_loss_fn = tf.keras.losses.MeanSquaredError()
        self.loss_latent = MMD_multiscale
        self.loss_backward = MSE

    def train_step(self, data):
        if not self.finetune:
            self.model.encoder.trainable = False
            self.model.decoder.trainable = False

        x_batch_train, y_batch_train = data
        undirected_x_batch_train = (x_batch_train[0], to_undiredted_adj(x_batch_train[1]))
        y = y_batch_train[:, -y_dim:]
        z = tf.random.normal([tf.shape(y_batch_train)[0], z_dim])
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
            ops_cls, adj_cls, kl_loss, y_out, x_encoding = self.model(undirected_x_batch_train,
                                                                      training=True)  # Logits for this minibatch

            # To avoid nan loss when batch size is small
            if tf.shape(non_nan_idx)[0] != 0:
                reg_loss = self.reg_loss_fn(tf.gather(y, non_nan_idx),
                                            tf.gather(y_out[:, z_dim:], non_nan_idx))
                latent_loss = self.loss_latent(tf.gather(y_short, non_nan_idx),
                                               tf.gather(tf.concat([y_out[:, :z_dim], y_out[:, -y_dim:]], axis=-1),
                                                         non_nan_idx))  # * x_batch_train.shape[0]
            else:
                reg_loss = 0.
                latent_loss = 0.

            forward_loss = self.w1 * reg_loss + self.w2 * latent_loss

            if self.finetune:
                ops_loss, adj_loss = cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls)
                rec_loss = ops_loss + adj_loss + kl_loss
                forward_loss += self.w0 * rec_loss

        grads = tape.gradient(forward_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

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
            if self.finetune:
                return {'total_loss': forward_loss, 'reg_loss': reg_loss, 'latent_loss': latent_loss, 'rev_loss': 0,
                        'rec_loss': rec_loss}
            return {'total_loss': forward_loss, 'reg_loss': reg_loss, 'latent_loss': latent_loss, 'rev_loss': 0}

        # Backward loss
        with tf.GradientTape() as tape:
            self.model.encoder.trainable = False
            self.model.decoder.trainable = False
            y = tf.gather(y, non_nan_idx)
            z = tf.gather(z, non_nan_idx)
            y = y + 0.0001 * tf.random.normal(shape=y.shape)
            _, _, _, _, x_encoding = self.model(undirected_x_batch_train, training=True)  # Logits for this minibatch
            x_rev = self.model.inverse(tf.concat([z, y], axis=-1))
            rev_loss = self.loss_backward(x_rev, tf.gather(x_encoding, non_nan_idx))  # * x_batch_train.shape[0]
            loss = self.w3 * rev_loss

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.model.encoder.trainable = True
        self.model.decoder.trainable = True

        if self.finetune:
            return {'total_loss': forward_loss + loss,
                    'reg_loss': reg_loss,
                    'latent_loss': latent_loss,
                    'rev_loss': rev_loss,
                    'rec_loss': rec_loss}

        return {'total_loss': forward_loss + loss,
                'reg_loss': reg_loss,
                'latent_loss': latent_loss,
                'rev_loss': rev_loss}

    def test_step(self, data):
        x_batch_train, y_batch_train = data
        undirected_x_batch_train = (x_batch_train[0], to_undiredted_adj(x_batch_train[1]))
        y = y_batch_train[:, -y_dim:]
        z = tf.random.normal(shape=[tf.shape(y_batch_train)[0], z_dim])
        y_short = tf.concat([z, y], axis=-1)

        ops_cls, adj_cls, kl_loss, y_out, x_encoding = self.model(undirected_x_batch_train,
                                                                  training=False)  # Logits for this minibatch

        reg_loss = self.reg_loss_fn(y, y_out[:, z_dim:])
        latent_loss = self.loss_latent(y_short, tf.concat([y_out[:, :z_dim], y_out[:, -y_dim:]],
                                                          axis=-1))  # * x_batch_train.shape[0]
        x_rev = self.model.inverse(y_short)
        rev_loss = self.loss_backward(x_rev, x_encoding)  # * x_batch_train.shape[0]
        if self.finetune:
            ops_loss, adj_loss = cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls)
            rec_loss = ops_loss + adj_loss + kl_loss
            return {'total_loss': self.w1 * reg_loss + self.w2 * latent_loss + self.w3 * rev_loss + self.w0 * rec_loss,
                    'reg_loss': reg_loss,
                    'latent_loss': latent_loss,
                    'rev_loss': rev_loss,
                    'rec_loss': rec_loss}

        return {'total_loss': self.w1 * reg_loss + self.w2 * latent_loss + self.w3 * rev_loss,
                'reg_loss': reg_loss,
                'latent_loss': latent_loss,
                'rev_loss': rev_loss}


def train(phase: int, model, loader, batch_size, train_epochs, steps_per_epoch, callbacks=None, x_dim=None, y_dim=None,
          z_dim=None, finetune=False):
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    if phase == 1:
        trainer = Trainer1(model)
    elif phase == 2:
        trainer = Trainer2(model, x_dim, y_dim, z_dim, finetune)

    try:
        kw = {'validation_steps': loader['valid'].steps_per_epoch}
    except:
        kw = {}

    trainer.compile(optimizer=optimizer, run_eagerly=True)
    trainer.fit(loader['train'].load(),
                validation_data=loader['valid'].load(),
                batch_size=batch_size,
                epochs=train_epochs,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                **kw)
    model.save_weights(os.path.join(logdir, f'modelGAE_weights_phase{phase}'))
    return trainer


if __name__ == '__main__':
    is_only_validation_data = True
    label_epochs = 200

    train_phase = [0, 1]  # 0 not train, 1 train
    pretrained_phase1_weight = 'logs/20230412-110343_GAE_ok/modelGAE_weights_phase1'

    repeat = 1
    eps_scale = 0.05
    d_model = 32
    dropout_rate = 0.0
    dff = 256
    num_layers = 3
    num_heads = 3
    finetune = False
    latent_dim = 16

    batch_size = 256
    train_epochs = 1000
    patience = 100

    # 15624
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(label_epochs), seed=777),
                                              ratio=[0.8, 0.1, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    # datasets['train'] = datasets['train'][:args.train_sample_amount]
    # datasets['valid'] = datasets['valid'][:args.valid_sample_amount]

    for key in datasets:
        if is_only_validation_data:
            datasets[key].apply(OnlyValidAccTransform())
            datasets[key].apply(OnlyFinalAcc())
            datasets[key].apply(LabelScale_NasBench101(scale=0.01))
        else:
            datasets[key].apply(ReshapeYTransform())

    x_dim = latent_dim
    y_dim = 1  # 1
    z_dim = x_dim - y_dim  # 15

    pretrained_model = GraphAutoencoder(latent_dim=latent_dim, num_layers=num_layers,
                                         d_model=d_model, num_heads=num_heads,
                                         dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                         num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=eps_scale)
    pretrained_model((tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))))

    nvp_config = {
        'n_couple_layer': 4,
        'n_hid_layer': 4,
        'n_hid_dim': 256,
        'name': 'NVP'
    }
    model = GraphAutoencoderNVP(nvp_config=nvp_config, latent_dim=latent_dim, num_layers=num_layers,
                                            d_model=d_model, num_heads=num_heads,
                                            dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                            num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=eps_scale)

    model((tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))))
    model.summary(print_fn=logger.info)

    loader = {'train': BatchLoader(datasets['train'], batch_size=batch_size, shuffle=True, epochs=train_epochs),
              'valid': BatchLoader(datasets['valid'], batch_size=batch_size, shuffle=False, epochs=train_epochs),
              'test': BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)}

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    if train_phase[0]:
        logger.info('Train phase 1')


        class SaveModelCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 50 == 0:
                    self.model.save_weights(os.path.join(logdir, 'model_{:04d}.ckpt'.format(epoch + 1)))

        callbacks = [CSVLogger(os.path.join(logdir, "learning_curve_phase1.log")),
                     SaveModelCallback(),
                     tensorboard_callback,
                     EarlyStopping(monitor='val_rec_loss', patience=patience, restore_best_weights=True)]
        trainer = train(1, model, loader, batch_size, train_epochs, loader['train'].steps_per_epoch, callbacks)
        results = trainer.evaluate(loader['test'].load(), steps=loader['test'].steps_per_epoch, batch_size=256)
        logger.info(f'{results}')
    else:
        pretrained_model.load_weights(pretrained_phase1_weight)
        model.encoder.set_weights(pretrained_model.encoder.get_weights())
        model.decoder.set_weights(pretrained_model.decoder.get_weights())

    if train_phase[1]:
        logger.info('Train phase 2')
        callbacks = [CSVLogger(os.path.join(logdir, f"learning_curve_phase2.log")),
                     tensorboard_callback,
                     EarlyStopping(monitor='val_total_loss', patience=patience, restore_best_weights=True)]
        trainer = train(2, model, loader, batch_size, train_epochs, loader['train'].steps_per_epoch, callbacks,
                        x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, finetune=finetune)
        results = trainer.evaluate(loader['test'], batch_size=256)
        logger.info(f'{results}')
    else:
        exit()
