import argparse
import random
from tensorflow.python.keras.callbacks import CSVLogger, EarlyStopping
from datasets.transformation import ReshapeYTransform, OnlyValidAccTransform, OnlyFinalAcc, LabelScale
from invertible_neural_networks.flow import MSE, MMD_multiscale
from models.GNN import GraphAutoencoder, GraphAutoencoderNVP, weighted_mse
from models.TransformerAE import TransformerAutoencoderNVP
import tensorflow as tf
import os
from datasets.nb201_dataset import NasBench201Dataset
from datasets.utils import train_valid_test_split_dataset, mask_graph_dataset
from spektral.data import BatchLoader
from evalGAE import eval_query_best
from utils.tf_utils import SaveModelCallback
from utils.py_utils import get_logdir_and_logger
from models.Diffusion import GraphAutoencoderDiffusion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sample_amount', type=int, default=350, help='Number of samples to train (default: 350)')
    parser.add_argument('--valid_sample_amount', type=int, default=50, help='Number of samples to train (default: 50)')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


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
    def __init__(self, model: TransformerAutoencoderNVP, finetune=False):
        super(Trainer2, self).__init__()
        self.model = model
        self.finetune = finetune

        self.noise_loss_fn = tf.keras.losses.MeanSquaredError()
        self.loss_latent = MMD_multiscale
        self.loss_backward = MSE
        self.loss_tracker = {
            'total_loss': tf.keras.metrics.Mean(name="total_loss"),
            'noise': tf.keras.metrics.Mean(name="noise"),
        }
        if self.finetune:
            self.loss_tracker['rec_loss'] = tf.keras.metrics.Mean(name="rec_loss")
            self.ops_weight = 10
            self.adj_weight = 1
            self.kl_weight = 0.005

    def train_step(self, data):
        if not self.finetune:
            self.model.encoder.trainable = False
            self.model.decoder.trainable = False

        x_batch_train, y_batch_train = data
        undirected_x_batch_train = (x_batch_train[0], to_undiredted_adj(x_batch_train[1]))
        y = y_batch_train[:, -1:]
        nan_mask = tf.squeeze(tf.where(~tf.math.is_nan(tf.reduce_sum(y_batch_train, axis=-1)), x=0, y=1))

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        # Forward loss and AE Reconstruct loss
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            ops_cls, adj_cls, kl_loss, latent_mean, pred_noise, noise = self.model(undirected_x_batch_train, y, training=True)
            noise_loss = self.noise_loss_fn(noise, pred_noise)

        grads = tape.gradient(noise_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        total_loss = noise_loss

        self.loss_tracker['noise'].update_state(noise_loss)
        self.loss_tracker['total_loss'].update_state(total_loss)
        return {key: value.result() for key, value in self.loss_tracker.items()}

    def test_step(self, data):
        x_batch_train, y_batch_train = data
        undirected_x_batch_train = (x_batch_train[0], to_undiredted_adj(x_batch_train[1]))
        y = y_batch_train[:, -1:]
        nan_mask = tf.squeeze(tf.where(~tf.math.is_nan(tf.reduce_sum(y_batch_train, axis=-1)), x=0, y=1))

        ops_cls, adj_cls, kl_loss, latent_mean, pred_noise, noise = self.model(undirected_x_batch_train, y, training=False)
        noise_loss = self.noise_loss_fn(noise, pred_noise)
        total_loss = noise_loss
        self.loss_tracker['noise'].update_state(noise_loss)
        self.loss_tracker['total_loss'].update_state(total_loss)
        return {key: value.result() for key, value in self.loss_tracker.items()}

    @property
    def metrics(self):
        return [value for _, value in self.loss_tracker.items()]


def train(phase: int, model, loader, train_epochs, logdir, callbacks=None, finetune=False, learning_rate=1e-3):
    #optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    if phase == 1:
        trainer = Trainer1(model)
    elif phase == 2:
        trainer = Trainer2(model, finetune)
    else:
        raise ValueError('phase should be 1 or 2')

    try:
        kw = {'validation_steps': loader['valid'].steps_per_epoch,
              'steps_per_epoch': loader['train'].steps_per_epoch}
    except:
        kw = {}

    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), run_eagerly=False)
    trainer.fit(loader['train'].load(),
                validation_data=loader['valid'].load(),
                epochs=train_epochs,
                callbacks=callbacks,
                **kw)
    model.save_weights(os.path.join(logdir, f'modelGAE_weights_phase{phase}'))
    return trainer


def main(seed, train_sample_amount, valid_sample_amount):
    logdir, logger = get_logdir_and_logger(f'trainGAE_two_phase_{seed}.log')
    random_seed = seed
    tf.random.set_seed(random_seed)
    random.seed(random_seed)

    num_ops = 7
    num_nodes = 8
    num_adjs = 64
    is_only_validation_data = True
    label_epochs = 200

    train_phase = [0, 1]  # 0 not train, 1 train
    pretrained_weight = 'logs/20230415-161729_GAE_eps0.5_b32_kl0.005_ops10/modelGAE_weights_phase1'

    # repeat = 1
    eps_scale = 0.5
    d_model = 32
    dropout_rate = 0.0
    dff = 256
    num_layers = 3
    num_heads = 3
    finetune = False
    latent_dim = 14

    train_epochs = 500
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
            datasets[key].apply(LabelScale(scale=0.01))
        else:
            datasets[key].apply(ReshapeYTransform())

    pretrained_model = GraphAutoencoder(latent_dim=latent_dim, num_layers=num_layers,
                                        d_model=d_model, num_heads=num_heads,
                                        dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                        num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=eps_scale)
    pretrained_model(
        (tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))))


    model = GraphAutoencoderDiffusion(latent_dim=latent_dim, num_layers=num_layers,
                                d_model=d_model, num_heads=num_heads,
                                dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=eps_scale, diffusion_steps=500)
    model((tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))), tf.zeros(shape=(1, 1)))
    model.summary(print_fn=logger.info)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    if train_phase[0]:
        logger.info('Train phase 1')
        batch_size = 32
        loader = {'train': BatchLoader(datasets['train'], batch_size=batch_size, shuffle=True, epochs=train_epochs * 2),
                  'valid': BatchLoader(datasets['valid'], batch_size=batch_size, shuffle=False,
                                       epochs=train_epochs * 2),
                  'test': BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)}
        callbacks = [CSVLogger(os.path.join(logdir, "learning_curve_phase1.csv")),
                     SaveModelCallback(save_dir=logdir, every_epoch=100),
                     tensorboard_callback,
                     EarlyStopping(monitor='val_rec_loss', patience=100, restore_best_weights=True)]
        trainer = train(1, pretrained_model, loader, train_epochs, logdir, callbacks)
        results = trainer.evaluate(loader['test'].load(), steps=loader['test'].steps_per_epoch)
        logger.info(f'{dict(zip(trainer.metrics_names, results))}')
    else:
        pretrained_model.load_weights(pretrained_weight)

    # Load AE weights from pretrained model
    model.encoder.set_weights(pretrained_model.encoder.get_weights())
    model.decoder.set_weights(pretrained_model.decoder.get_weights())

    if train_phase[1]:
        batch_size = 32
        logger.info('Train phase 2')
        random.shuffle(datasets['train'])
        random.shuffle(datasets['valid'])

        tmp_train = datasets['train'][:train_sample_amount]
        tmp_valid = datasets['valid'][:valid_sample_amount]
        datasets['train'] = tmp_train
        datasets['valid'] = tmp_valid
        for i in range(20):
            datasets['train'] += tmp_train
            datasets['valid'] += tmp_valid
        '''
        datasets['train'] = mask_graph_dataset(datasets['train'], train_sample_amount, 20)
        datasets['valid'] = mask_graph_dataset(datasets['valid'], valid_sample_amount, 20)
        '''
        # datasets['train'] = datasets['train'][:args.train_sample_amount]
        # datasets['valid'] = datasets['valid'][:args.valid_sample_amount]
        loader = {'train': BatchLoader(datasets['train'], batch_size=batch_size, shuffle=True, epochs=train_epochs),
                  'valid': BatchLoader(datasets['valid'], batch_size=batch_size, shuffle=False, epochs=train_epochs),
                  'test': BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)}

        callbacks = [CSVLogger(os.path.join(logdir, f"learning_curve_phase2.csv")),
                     tensorboard_callback,
                     EarlyStopping(monitor='val_total_loss', patience=patience, restore_best_weights=True)]
        trainer = train(2, model, loader, train_epochs, logdir, callbacks, finetune=finetune)
        results = trainer.evaluate(loader['test'], steps=loader['test'].steps_per_epoch)
        logger.info(str(dict(zip(trainer.metrics_names, results))))
    else:
        model.load_weights(pretrained_weight)

    #invalid, avg_acc, best_acc = eval_query_best(model, x_dim, z_dim)
    #logger.info(f'Number of invalid decode {invalid}')
    #logger.info(f'Avg found acc {avg_acc}')
    #logger.info(f'Best found acc {best_acc}')
    #return invalid, avg_acc, best_acc


if __name__ == '__main__':
    args = parse_args()
    main(args.seed, args.train_sample_amount, args.valid_sample_amount),