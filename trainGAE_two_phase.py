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

from utils.tf_utils import SaveModelCallback

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
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

random_seed = args.seed
tf.random.set_seed(random_seed)

num_ops = 7
num_nodes = 8
num_adjs = 64


def cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls):
    ops_label, adj_label = x_batch_train
    adj_label = tf.reshape(adj_label, [tf.shape(adj_label)[0], -1])
    #ops_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(ops_label, ops_cls)
    ops_loss = tf.keras.losses.KLDivergence()(ops_label, ops_cls)
    adj_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(adj_label, adj_cls)
    return ops_loss, adj_loss


def to_undiredted_adj(adj):
    undirected_adj = tf.cast(tf.cast(adj, tf.int32) | tf.cast(tf.transpose(adj, perm=[0, 2, 1]), tf.int32), tf.float32)
    return undirected_adj


class Trainer1(tf.keras.Model):
    def __init__(self, model: GraphAutoencoder):
        super(Trainer1, self).__init__()
        self.model = model
        self.ops_weight = 10
        self.adj_weight = 1
        self.kl_weight = 0.005

        self.loss_tracker = {
            'rec_loss': tf.keras.metrics.Mean(name="rec_loss"),
            'ops_loss': tf.keras.metrics.Mean(name="ops_loss"),
            'adj_loss': tf.keras.metrics.Mean(name="adj_loss"),
            'kl_loss': tf.keras.metrics.Mean(name="kl_loss")
        }

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

        self.loss_tracker['rec_loss'].update_state(rec_loss)
        self.loss_tracker['ops_loss'].update_state(ops_loss)
        self.loss_tracker['adj_loss'].update_state(adj_loss)
        self.loss_tracker['kl_loss'].update_state(kl_loss)
        return {key: value.result() for key, value in self.loss_tracker.items()}

    def test_step(self, data):
        x_batch_train, _ = data
        undirected_x_batch_train = (x_batch_train[0], to_undiredted_adj(x_batch_train[1]))
        ops_cls, adj_cls, kl_loss, _ = self.model(undirected_x_batch_train, training=True)  # Logits for this minibatch
        ops_loss, adj_loss = cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls)
        rec_loss = self.ops_weight * ops_loss + self.adj_weight * adj_loss + self.kl_weight * kl_loss

        self.loss_tracker['rec_loss'].update_state(rec_loss)
        self.loss_tracker['ops_loss'].update_state(ops_loss)
        self.loss_tracker['adj_loss'].update_state(adj_loss)
        self.loss_tracker['kl_loss'].update_state(kl_loss)
        return {key: value.result() for key, value in self.loss_tracker.items()}

    @property
    def metrics(self):
        return [value for _, value in self.loss_tracker.items()]


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
        self.loss_tracker = {
            'total_loss': tf.keras.metrics.Mean(name="total_loss"),
            'reg_loss': tf.keras.metrics.Mean(name="reg_loss"),
            'latent_loss': tf.keras.metrics.Mean(name="latent_loss"),
            'rev_loss': tf.keras.metrics.Mean(name="rev_loss")
        }
        if self.finetune:
            self.loss_tracker['rec_loss'] = tf.keras.metrics.Mean(name="rec_loss")


    def cal_reg_and_latent_loss(self, y, z, y_out, nan_mask):
        reg_loss = self.reg_loss_fn(tf.dynamic_partition(y, nan_mask, 2)[0],
                                    tf.dynamic_partition(y_out[:, self.z_dim:], nan_mask, 2)[0])
        latent_loss = self.loss_latent(tf.dynamic_partition(tf.concat([z, y], axis=-1), nan_mask, 2)[0],
                                       tf.dynamic_partition(tf.concat([y_out[:, :self.z_dim], y_out[:, -self.y_dim:]], axis=-1),
                                                 nan_mask, 2)[0])  # * x_batch_train.shape[0]
        return reg_loss, latent_loss

    def cal_rev_loss(self, undirected_x_batch_train, y, z, nan_mask, noise_scale):
        self.model.encoder.trainable = False
        self.model.decoder.trainable = False
        y = tf.dynamic_partition(y, nan_mask, 2)[0]
        z = tf.dynamic_partition(z, nan_mask, 2)[0]
        y = y + noise_scale * tf.random.normal(shape=tf.shape(y), dtype=tf.float32)
        _, _, _, _, x_encoding = self.model(undirected_x_batch_train, training=True)  # Logits for this minibatch
        x_rev = self.model.inverse(tf.concat([z, y], axis=-1))
        rev_loss = self.loss_backward(x_rev, tf.dynamic_partition(x_encoding, nan_mask, 2)[0])  # * x_batch_train.shape[0]
        return rev_loss

    def train_step(self, data):
        if not self.finetune:
            self.model.encoder.trainable = False
            self.model.decoder.trainable = False

        x_batch_train, y_batch_train = data
        undirected_x_batch_train = (x_batch_train[0], to_undiredted_adj(x_batch_train[1]))
        y = y_batch_train[:, -self.y_dim:]
        z = tf.random.normal([tf.shape(y_batch_train)[0], self.z_dim])
        nan_mask = tf.squeeze(tf.where(~tf.math.is_nan(tf.reduce_sum(y_batch_train, axis=-1)), x=0, y=1))

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        # Forward loss and AE Reconstruct loss
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            ops_cls, adj_cls, kl_loss, y_out, x_encoding = self.model(undirected_x_batch_train, training=True)

            # To avoid nan loss when batch size is small
            reg_loss, latent_loss = tf.cond(tf.shape(nan_mask)[0] != 0,
                                            lambda: self.cal_reg_and_latent_loss(y, z, y_out, nan_mask),
                                            lambda: (0., 0.))
            ''' Original code for Eargly execution
            # To avoid nan loss when batch size is small
            if tf.shape(nan_mask)[0] != 0):
                reg_loss = self.reg_loss_fn(tf.gather(y, nan_mask),
                                            tf.gather(y_out[:, self.z_dim:], nan_mask))
                latent_loss = self.loss_latent(tf.gather(y_short, nan_mask),
                                               tf.gather(tf.concat([y_out[:, :self.z_dim], y_out[:, -self.y_dim:]], axis=-1),
                                                         nan_mask))  # * x_batch_train.shape[0]
            else:
                reg_loss = 0.
                latent_loss = 0.
            '''

            forward_loss = self.w1 * reg_loss + self.w2 * latent_loss
            rec_loss = 0.
            if self.finetune:
                ops_loss, adj_loss = cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls)
                rec_loss = ops_loss + adj_loss + kl_loss
                forward_loss += self.w0 * rec_loss

        grads = tape.gradient(forward_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # Backward loss
        with tf.GradientTape() as tape:
            # To avoid nan loss when batch size is small
            rev_loss = tf.cond(tf.shape(nan_mask)[0] != 0,
                               lambda: self.cal_rev_loss(undirected_x_batch_train, y, z, nan_mask, 0.0001),
                               lambda: 0.)
            backward_loss = self.w3 * rev_loss

        grads = tape.gradient(backward_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.model.encoder.trainable = True
        self.model.decoder.trainable = True

        self.loss_tracker['total_loss'].update_state(forward_loss + backward_loss)
        self.loss_tracker['reg_loss'].update_state(reg_loss)
        self.loss_tracker['latent_loss'].update_state(latent_loss)
        self.loss_tracker['rev_loss'].update_state(rev_loss)
        if self.finetune:
            self.loss_tracker['rec_loss'].update_state(rec_loss)

        return {key: value.result() for key, value in self.loss_tracker.items()}

    def test_step(self, data):
        x_batch_train, y_batch_train = data
        undirected_x_batch_train = (x_batch_train[0], to_undiredted_adj(x_batch_train[1]))
        y = y_batch_train[:, -self.y_dim:]
        z = tf.random.normal(shape=[tf.shape(y_batch_train)[0], self.z_dim])

        ops_cls, adj_cls, kl_loss, y_out, x_encoding = self.model(undirected_x_batch_train, training=False)
        reg_loss, latent_loss = self.cal_reg_and_latent_loss(y, z, y_out, tf.zeros(tf.shape(y_batch_train)[0], dtype=tf.int32))
        forward_loss = self.w1 * reg_loss + self.w2 * latent_loss
        rev_loss = self.cal_rev_loss(undirected_x_batch_train, y, z, tf.zeros(tf.shape(y_batch_train)[0], dtype=tf.int32), 0.)
        backward_loss = self.w3 * rev_loss
        if self.finetune:
            ops_loss, adj_loss = cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls)
            rec_loss = ops_loss + adj_loss + kl_loss
            forward_loss += self.w0 * rec_loss

        self.loss_tracker['total_loss'].update_state(forward_loss + backward_loss)
        self.loss_tracker['reg_loss'].update_state(reg_loss)
        self.loss_tracker['latent_loss'].update_state(latent_loss)
        self.loss_tracker['rev_loss'].update_state(rev_loss)
        if self.finetune:
            self.loss_tracker['rec_loss'].update_state(rec_loss)

        return {key: value.result() for key, value in self.loss_tracker.items()}

    @property
    def metrics(self):
        return [value for _, value in self.loss_tracker.items()]

def train(phase: int, model, loader, train_epochs, callbacks=None, x_dim=None, y_dim=None,
          z_dim=None, finetune=False):
    #optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    if phase == 1:
        trainer = Trainer1(model)
    elif phase == 2:
        trainer = Trainer2(model, x_dim, y_dim, z_dim, finetune)

    try:
        kw = {'validation_steps': loader['valid'].steps_per_epoch,
              'steps_per_epoch': loader['train'].steps_per_epoch}
    except:
        kw = {}

    trainer.compile(optimizer='adam', run_eagerly=False)
    trainer.fit(loader['train'].load(),
                validation_data=loader['valid'].load(),
                epochs=train_epochs,
                callbacks=callbacks,
                **kw)
    model.save_weights(os.path.join(logdir, f'modelGAE_weights_phase{phase}'))
    return trainer


if __name__ == '__main__':
    is_only_validation_data = True
    label_epochs = 200

    train_phase = [1, 1]  # 0 not train, 1 train
    pretrained_phase1_weight = 'logs/20230414-202845_GAE_opskl/modelGAE_weights_phase1'

    #repeat = 1
    eps_scale = 0.5
    d_model = 32
    dropout_rate = 0.0
    dff = 256
    num_layers = 3
    num_heads = 3
    finetune = False
    latent_dim = 14

    train_epochs = 1000
    patience = 100

    # 15624
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(label_epochs), seed=False),
                                              ratio=[0.8, 0.1, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    for key in datasets:
        if is_only_validation_data:
            datasets[key].apply(OnlyValidAccTransform())
            datasets[key].apply(OnlyFinalAcc())
            datasets[key].apply(LabelScale_NasBench101(scale=0.01))
        else:
            datasets[key].apply(ReshapeYTransform())

    x_dim = latent_dim
    y_dim = 1  # 1
    z_dim = latent_dim * 2 - 1  # 27
    tot_dim = y_dim + z_dim  # 28
    pad_dim = tot_dim - x_dim  # 14

    pretrained_model = GraphAutoencoder(latent_dim=latent_dim, num_layers=num_layers,
                             d_model=d_model, num_heads=num_heads,
                             dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                             num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=eps_scale)
    pretrained_model((tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))))

    nvp_config = {
        'n_couple_layer': 4,
        'n_hid_layer': 4,
        'n_hid_dim': 256,
        'name': 'NVP',
        'inp_dim': tot_dim
    }
    model = GraphAutoencoderNVP(nvp_config=nvp_config, latent_dim=latent_dim, num_layers=num_layers,
                                d_model=d_model, num_heads=num_heads,
                                dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=0.0)
    model((tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))))
    model.summary(print_fn=logger.info)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    if train_phase[0]:
        logger.info('Train phase 1')
        batch_size = 32
        loader = {'train': BatchLoader(datasets['train'], batch_size=batch_size, shuffle=True, epochs=train_epochs*2),
                  'valid': BatchLoader(datasets['valid'], batch_size=batch_size, shuffle=False, epochs=train_epochs*2),
                  'test': BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)}
        callbacks = [CSVLogger(os.path.join(logdir, "learning_curve_phase1.csv")),
                     SaveModelCallback(save_dir=logdir, every_epoch=100),
                     tensorboard_callback,
                     EarlyStopping(monitor='val_rec_loss', patience=patience, restore_best_weights=True)]
        trainer = train(1, pretrained_model, loader, train_epochs, callbacks)
        results = trainer.evaluate(loader['test'].load(), steps=loader['test'].steps_per_epoch)
        logger.info(f'{dict(zip(trainer.metrics_names, results))}')
    else:
        pretrained_model.load_weights(pretrained_phase1_weight)

    # Load AE weights from pretrained model
    model.encoder.set_weights(pretrained_model.encoder.get_weights())
    model.decoder.set_weights(pretrained_model.decoder.get_weights())

    if train_phase[1]:
        batch_size = 256
        logger.info('Train phase 2')

        tmp_train = datasets['train'][:args.train_sample_amount]
        tmp_valid = datasets['valid'][:args.valid_sample_amount]
        datasets['train'] = tmp_train
        datasets['valid'] = tmp_valid
        for i in range(20):
            datasets['train'] += tmp_train
            datasets['valid'] += tmp_valid

        #datasets['train'] = datasets['train'][:args.train_sample_amount]
        #datasets['valid'] = datasets['valid'][:args.valid_sample_amount]
        loader = {'train': BatchLoader(datasets['train'], batch_size=batch_size, shuffle=True, epochs=train_epochs),
                  'valid': BatchLoader(datasets['valid'], batch_size=batch_size, shuffle=False, epochs=train_epochs),
                  'test': BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)}

        callbacks = [CSVLogger(os.path.join(logdir, f"learning_curve_phase2.csv")),
                     tensorboard_callback,
                     EarlyStopping(monitor='val_total_loss', patience=patience, restore_best_weights=True)]
        trainer = train(2, model, loader, train_epochs, callbacks,
                        x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, finetune=finetune)
        results = trainer.evaluate(loader['test'], steps=loader['test'].steps_per_epoch)
        logger.info(str(dict(zip(trainer.metrics_names, results))))
    else:
        exit()
