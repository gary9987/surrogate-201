import argparse
import copy
import pickle
import random
import numpy as np
from tensorflow.python.keras.callbacks import CSVLogger, EarlyStopping
from datasets.transformation import ReshapeYTransform, OnlyValidAccTransform, OnlyFinalAcc, LabelScale
from invertible_neural_networks.flow import MMD_multiscale
from models.GNN import GraphAutoencoder, GraphAutoencoderNVP, get_rank_weight
from models.TransformerAE import TransformerAutoencoderNVP
import tensorflow as tf
import os
from datasets.nb201_dataset import NasBench201Dataset, OP_PRIMITIVES_NB201
from datasets.nb101_dataset import NasBench101Dataset, OP_PRIMITIVES_NB101, mask_for_model, mask_padding_vertex_for_spec
from datasets.utils import train_valid_test_split_dataset, mask_graph_dataset, arch_list_to_set, graph_to_str, \
    repeat_graph_dataset_element
from spektral.data import PackedBatchLoader
from evalGAE import eval_query_best, query_tabular, nb101_dataset
from utils.py_utils import get_logdir_and_logger
from spektral.data import Graph
from utils.tf_utils import to_undiredted_adj, set_global_determinism
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sample_amount', type=int, default=50, help='Number of samples to train (default: 50)')
    parser.add_argument('--valid_sample_amount', type=int, default=50, help='Number of samples to train (default: 50)')
    parser.add_argument('--query_budget', type=int, default=192)
    parser.add_argument('--dataset', type=str, default='cifar10-valid',
                        help='Could be nb101, cifar10-valid, cifar100, ImageNet16-120')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--no_finetune', dest='finetune', action='store_false')
    parser.set_defaults(finetune=True)
    parser.add_argument('--retrain_finetune', action='store_true')
    parser.add_argument('--no_retrain_finetune', dest='retrain_finetune', action='store_false')
    parser.set_defaults(retrain_finetune=True)
    parser.add_argument('--rank_weight', action='store_true')
    parser.add_argument('--no_rank_weight', dest='rank_weight', action='store_false')
    parser.set_defaults(rank_weight=True)
    parser.add_argument('--random_sample', action='store_true')
    parser.set_defaults(random_sample=False)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls, reduction='auto', rank_weight=None):
    ops_label, adj_label = x_batch_train
    # adj_label = tf.reshape(adj_label, [tf.shape(adj_label)[0], -1])
    ops_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=reduction)(ops_label, ops_cls)
    #ops_loss = tf.keras.losses.KLDivergence()(ops_label, ops_cls)
    adj_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=reduction)(adj_label, adj_cls)
    if reduction == 'none':
        ops_loss = tf.reduce_mean(ops_loss, axis=-1)
        adj_loss = tf.reduce_mean(adj_loss, axis=-1)
    if rank_weight is not None:
        ops_loss = tf.reduce_sum(tf.multiply(ops_loss, rank_weight))
        adj_loss = tf.reduce_sum(tf.multiply(adj_loss, rank_weight))

    return ops_loss, adj_loss


class Trainer1(tf.keras.Model):
    def __init__(self, model: GraphAutoencoder):
        super(Trainer1, self).__init__()
        self.model = model
        self.ops_weight = 1
        self.adj_weight = 1
        self.kl_weight = 0.16

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
    def __init__(self, model: TransformerAutoencoderNVP, x_dim, y_dim, z_dim, finetune=False, is_rank_weight=False):
        super(Trainer2, self).__init__()
        self.model = model
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.finetune = finetune
        self.is_rank_weight = is_rank_weight

        # For reg loss weight
        self.w1 = 5.
        # For latent loss weight
        self.w2 = 1.
        # For rev loss weight
        self.w3 = 10.

        if self.is_rank_weight:
            self.reduction = 'none'
        else:
            self.reduction = 'auto'

        self.reg_loss_fn = tf.keras.losses.MeanSquaredError(reduction=self.reduction)
        self.loss_latent = MMD_multiscale
        self.loss_backward = tf.keras.losses.MeanSquaredError(reduction=self.reduction)
        self.loss_tracker = {
            'total_loss': tf.keras.metrics.Mean(name="total_loss"),
            'reg_loss': tf.keras.metrics.Mean(name="reg_loss"),
            'latent_loss': tf.keras.metrics.Mean(name="latent_loss"),
            'rev_loss': tf.keras.metrics.Mean(name="rev_loss")
        }
        if self.finetune:
            self.loss_tracker.update({
                'rec_loss': tf.keras.metrics.Mean(name="rec_loss"),
                'ops_loss': tf.keras.metrics.Mean(name="ops_loss"),
                'adj_loss': tf.keras.metrics.Mean(name="adj_loss"),
                'kl_loss': tf.keras.metrics.Mean(name="kl_loss")
            })
            self.ops_weight = 1
            self.adj_weight = 1
            self.kl_weight = 0.16

    def cal_reg_and_latent_loss(self, y, z, y_out, nan_mask, rank_weight=None):
        reg_loss = self.reg_loss_fn(tf.boolean_mask(y, nan_mask),
                                    tf.boolean_mask(y_out[:, self.z_dim:], nan_mask))
        latent_loss = self.loss_latent(tf.boolean_mask(tf.concat([z, y], axis=-1), nan_mask),
                                       tf.boolean_mask(
                                           tf.concat([y_out[:, :self.z_dim], y_out[:, -self.y_dim:]], axis=-1),
                                           nan_mask))  # * x_batch_train.shape[0]
        if self.is_rank_weight:
            # reg_loss (batch_size)
            reg_loss = tf.multiply(reg_loss, rank_weight)
            reg_loss = tf.reduce_sum(reg_loss)
        return reg_loss, latent_loss

    def cal_rev_loss(self, undirected_x_batch_train, y, z, nan_mask, noise_scale, rank_weight=None):
        y = tf.boolean_mask(y, nan_mask)
        z = tf.boolean_mask(z, nan_mask)
        y = y + noise_scale * tf.random.normal(shape=tf.shape(y), dtype=tf.float32)
        non_nan_x_batch = (tf.boolean_mask(undirected_x_batch_train[0], nan_mask),
                           tf.boolean_mask(undirected_x_batch_train[1], nan_mask))
        _, _, _, _, x_encoding = self.model(non_nan_x_batch, training=True)  # Logits for this minibatch
        x_rev = self.model.inverse(tf.concat([z, y], axis=-1))
        rev_loss = self.loss_backward(x_rev, x_encoding)  # * x_batch_train.shape[0]
        if self.is_rank_weight:
            # rev_loss (batch_size)
            rev_loss = tf.multiply(rev_loss, rank_weight)
            rev_loss = tf.reduce_sum(rev_loss)
        return rev_loss

    def train_step(self, data):
        if not self.finetune:
            self.model.encoder.trainable = False
            self.model.decoder.trainable = False

        x_batch_train, y_batch_train = data
        undirected_x_batch_train = (x_batch_train[0], to_undiredted_adj(x_batch_train[1]))
        y = y_batch_train[:, -self.y_dim:]
        z = tf.random.normal([tf.shape(y_batch_train)[0], self.z_dim])
        nan_mask = tf.where(~tf.math.is_nan(tf.reduce_sum(y, axis=-1)), x=True, y=False)
        rank_weight = get_rank_weight(tf.boolean_mask(y, nan_mask)) if self.is_rank_weight else None

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        # Forward loss and AE Reconstruct loss
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            ops_cls, adj_cls, kl_loss, y_out, x_encoding = self.model(undirected_x_batch_train, kl_reduction='none', training=True)

            # To avoid nan loss when batch size is small
            reg_loss, latent_loss = tf.cond(tf.reduce_any(nan_mask),
                                            lambda: self.cal_reg_and_latent_loss(y, z, y_out, nan_mask, rank_weight),
                                            lambda: (0., 0.))

            forward_loss = self.w1 * reg_loss + self.w2 * latent_loss
            rec_loss = 0.
            if self.finetune:
                ops_loss, adj_loss = cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls, 'auto', None)
                '''
                if rank_weight is not None:
                    kl_loss = tf.reduce_sum(tf.multiply(kl_loss, rank_weight))
                else:
                    kl_loss = tf.reduce_mean(kl_loss)
                '''
                kl_loss = tf.reduce_mean(kl_loss)
                rec_loss = self.ops_weight * ops_loss + self.adj_weight * adj_loss + self.kl_weight * kl_loss
                forward_loss += rec_loss

        grads = tape.gradient(forward_loss, self.model.trainable_weights)
        # grads = [tf.clip_by_norm(g, 1.) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # Backward loss
        with tf.GradientTape() as tape:
            self.model.encoder.trainable = False
            self.model.decoder.trainable = False
            # To avoid nan loss when batch size is small
            rev_loss = tf.cond(tf.reduce_any(nan_mask),
                               lambda: self.cal_rev_loss(undirected_x_batch_train, y, z, nan_mask, 0.0001, rank_weight),
                               lambda: 0.)
            backward_loss = self.w3 * rev_loss

        grads = tape.gradient(backward_loss, self.model.trainable_weights)
        # grads = [tf.clip_by_norm(g, 1.) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.model.encoder.trainable = True
        self.model.decoder.trainable = True

        self.loss_tracker['total_loss'].update_state(forward_loss + backward_loss)
        self.loss_tracker['reg_loss'].update_state(reg_loss)
        self.loss_tracker['latent_loss'].update_state(latent_loss)
        self.loss_tracker['rev_loss'].update_state(rev_loss)
        if self.finetune:
            self.loss_tracker['rec_loss'].update_state(rec_loss)
            self.loss_tracker['ops_loss'].update_state(ops_loss)
            self.loss_tracker['adj_loss'].update_state(adj_loss)
            self.loss_tracker['kl_loss'].update_state(kl_loss)

        return {key: value.result() for key, value in self.loss_tracker.items()}

    def test_step(self, data):
        x_batch_train, y_batch_train = data
        undirected_x_batch_train = (x_batch_train[0], to_undiredted_adj(x_batch_train[1]))
        y = y_batch_train[:, -self.y_dim:]
        z = tf.random.normal(shape=[tf.shape(y_batch_train)[0], self.z_dim])
        nan_mask = tf.where(~tf.math.is_nan(tf.reduce_sum(y, axis=-1)), x=True, y=False)
        rank_weight = get_rank_weight(tf.boolean_mask(y, nan_mask)) if self.is_rank_weight else None

        ops_cls, adj_cls, kl_loss, y_out, x_encoding = self.model(undirected_x_batch_train, kl_reduction='none', training=False)
        reg_loss, latent_loss = tf.cond(tf.reduce_any(nan_mask),
                                        lambda: self.cal_reg_and_latent_loss(y, z, y_out, nan_mask, rank_weight),
                                        lambda: (0., 0.))
        forward_loss = self.w1 * reg_loss + self.w2 * latent_loss
        rev_loss = tf.cond(tf.reduce_any(nan_mask),
                           lambda: self.cal_rev_loss(undirected_x_batch_train, y, z, nan_mask, 0., rank_weight),
                           lambda: 0.)
        backward_loss = self.w3 * rev_loss
        if self.finetune:
            ops_loss, adj_loss = cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls, 'auto', None)
            '''
            if rank_weight is not None:
                kl_loss = tf.reduce_sum(tf.multiply(kl_loss, rank_weight))
            else:
                kl_loss = tf.reduce_mean(kl_loss)
            '''
            kl_loss = tf.reduce_mean(kl_loss)
            rec_loss = self.ops_weight * ops_loss + self.adj_weight * adj_loss + self.kl_weight * kl_loss
            forward_loss += rec_loss

        self.loss_tracker['total_loss'].update_state(forward_loss + backward_loss)
        self.loss_tracker['reg_loss'].update_state(reg_loss)
        self.loss_tracker['latent_loss'].update_state(latent_loss)
        self.loss_tracker['rev_loss'].update_state(rev_loss)
        if self.finetune:
            self.loss_tracker['rec_loss'].update_state(rec_loss)
            self.loss_tracker['ops_loss'].update_state(ops_loss)
            self.loss_tracker['adj_loss'].update_state(adj_loss)
            self.loss_tracker['kl_loss'].update_state(kl_loss)

        return {key: value.result() for key, value in self.loss_tracker.items()}

    @property
    def metrics(self):
        return [value for _, value in self.loss_tracker.items()]


def train(phase: int, model, loader, train_epochs, logdir, callbacks=None, x_dim=None, y_dim=None,
          z_dim=None, finetune=False, learning_rate=1e-3):
    if phase == 1:
        trainer = Trainer1(model)
    elif phase == 2:
        trainer = Trainer2(model, x_dim, y_dim, z_dim, finetune)
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


def to_loader(datasets, batch_size: int, epochs: int):
    """
    :param datasets:
    :param batch_size:
    :param epochs:
    :return: return a dict of BatchLoader for 'train', 'valid', 'test'
    """
    loader = {}
    for key, value in datasets.items():
        if key == 'train' or key == 'valid':
            loader[key] = PackedBatchLoader(value, batch_size=batch_size, shuffle=True, epochs=epochs * 2)
        elif key == 'test':
            loader[key] = PackedBatchLoader(value, batch_size=batch_size, shuffle=False, epochs=1)

    return loader


class RandomArchGenerator:
    def __init__(self, dataset_name, is_only_validation_data):
        self.dataset_name = dataset_name
        if dataset_name == 'nb101':
            if os.path.exists('datasets/NasBench101Dataset.cache'):
                self.datasets = pickle.load(open('datasets/NasBench101Dataset.cache', 'rb'))
            else:
                self.datasets = NasBench101Dataset(start=0, end=423623)
                with open('datasets/NasBench101Dataset.cache', 'wb') as f:
                    pickle.dump(self.datasets, f)
        else:
            if os.path.exists(f'datasets/NasBench201Dataset_{dataset_name}.cache'):
                self.datasets = pickle.load(open(f'datasets/NasBench201Dataset_{dataset_name}.cache', 'rb'))
            else:
                self.datasets = NasBench201Dataset(start=0, end=15624, dataset=dataset_name, hp=str(200),
                                              seed=False)
                with open(f'datasets/NasBench201Dataset_{dataset_name}.cache', 'wb') as f:
                    pickle.dump(self.datasets, f)

        if is_only_validation_data:
            self.datasets.apply(OnlyValidAccTransform())
            self.datasets.apply(OnlyFinalAcc())
            if dataset_name != 'nb101':
                self.datasets.apply(LabelScale(scale=0.01))
        else:
            self.datasets.apply(ReshapeYTransform())

    def pad_for_nb101(self, graph):
        new_x = np.zeros((7, 5))
        new_a = np.zeros((7, 7))
        new_x[:graph.x.shape[0], :graph.x.shape[1]] = graph.x
        new_a[:graph.a.shape[0], :graph.a.shape[1]] = graph.a
        graph.x = new_x
        graph.a = new_a
        return graph

    def random_sample(self, visited, sample_amount):
        cand = []
        visited_arch = []
        while len(cand) < sample_amount:
            graphs = np.random.choice(self.datasets, sample_amount - len(cand), replace=False)
            if self.dataset_name == 'nb101':
                graphs = list(map(self.pad_for_nb101, graphs))
                for g in graphs:
                    a, x = mask_padding_vertex_for_spec(g.a, g.x)
                    spec_hash = nb101_dataset.get_spec_hash(a, np.argmax(x, axis=-1))
                    if spec_hash not in visited_arch and spec_hash not in visited:
                        cand.append({'x': g.x.astype(np.float32), 'a': g.a.astype(np.float32), 'y': g.y.reshape([1]).astype(np.float32)})
                        visited_arch.append(spec_hash)
            else:
                found_arch_list = [{'x': g.x.astype(np.float32), 'a': g.a.astype(np.float32), 'y': g.y.reshape([1]).astype(np.float32)} for g in graphs]
                found_arch_list = list(
                    filter(lambda arch: graph_to_str(arch) not in visited_arch and graph_to_str(arch) not in visited,
                           found_arch_list))
                cand.extend(found_arch_list)
                visited_arch.extend(list(map(graph_to_str, found_arch_list)))

        return cand[: sample_amount]


random_arch_generator: RandomArchGenerator = None


def sample_arch_candidates(model, dataset_name, x_dim, z_dim, visited, sample_amount=200):
    logger = logging.getLogger(__name__)
    found_arch_list_set = []
    visited_arch = []
    max_retry = 10
    std_idx = 0
    noise_std_list = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2]
    amount_scale_list = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.8, 3.0, 3.0]
    while len(found_arch_list_set) < sample_amount and std_idx < len(noise_std_list):
        retry = 0
        while len(found_arch_list_set) < sample_amount and retry < max_retry:
            _, _, _, found_arch_list = eval_query_best(model, dataset_name, x_dim, z_dim,
                                                       noise_scale=noise_std_list[std_idx],
                                                       query_amount=int(sample_amount * amount_scale_list[std_idx]))
            if dataset_name == 'nb101':
                found_arch_list = list(map(mask_for_model, found_arch_list))
                found_arch_list = list(filter(lambda arch: arch is not None and arch['x'] is not None, found_arch_list))
                found_in_this_round = []
                for arch in found_arch_list:
                    a, x = mask_padding_vertex_for_spec(arch['a'], arch['x'])
                    spec_hash = nb101_dataset.get_spec_hash(a, np.argmax(x, axis=-1))
                    if spec_hash not in visited_arch and spec_hash not in visited:
                        found_in_this_round.append(arch)
                        visited_arch.append(spec_hash)
            else:
                found_in_this_round = list(filter(lambda arch: graph_to_str(arch) not in visited_arch and graph_to_str(arch) not in visited, found_arch_list))
                found_in_this_round = arch_list_to_set(found_in_this_round)
                visited_arch.extend(list(map(graph_to_str, found_in_this_round)))

            if len(found_in_this_round) + len(found_arch_list_set) > sample_amount:
                random.shuffle(found_in_this_round)
                found_in_this_round = found_in_this_round[: sample_amount - len(found_arch_list_set)]

            found_arch_list_set.extend(found_in_this_round)
            retry += 1

        logger.info(f'std scale {noise_std_list[std_idx]}, num sample {len(found_arch_list_set)}')
        std_idx += 1

    # if retry == max_retry and len(found_arch_list_set) < sample_amount:
    #    model.set_weights_from_self_ckpt()
    #    logging.getLogger(__name__).info('Reset model weights')
    #    return None

    return found_arch_list_set


def predict_arch_acc(found_arch_list_set, model):
    """
    Predict accuracy by INN (performance predictor) and assign
     the predicted value to found_arch_list_set
    """
    # Predict accuracy by INN (performance predictor)
    x = tf.stack([tf.constant(i['x']) for i in found_arch_list_set])
    a = tf.stack([tf.constant(i['a']) for i in found_arch_list_set])
    if tf.shape(x)[0] != 0:
        a = to_undiredted_adj(a)
        _, _, _, reg, _ = model((x, a), training=False)
        for i in range(len(found_arch_list_set)):
            found_arch_list_set[i]['y'] = reg[i][-1].numpy()


def graph_to_spec_graph(graph):
    graph = copy.deepcopy(graph)
    graph.a, graph.x = mask_padding_vertex_for_spec(graph.a, graph.x)
    return graph


def retrain(trainer, datasets, dataset_name, batch_size, train_epochs, logdir, logger, repeat, top_k=5, random_sample=False):
    # Generate total 100 architectures
    if dataset_name == 'nb101':
        visited = {nb101_dataset.get_spec_hash(i.a, np.argmax(i.x, axis=-1)): i.y.tolist()
                   for i in list(map(graph_to_spec_graph, datasets['train'].graphs)) if not np.isnan(i.y).any()}
    else:
        visited = {graph_to_str(i): i.y.tolist() for i in datasets['train'].graphs if not np.isnan(i.y).any()}

    if random_sample:
        found_arch_list_set = random_arch_generator.random_sample(visited, sample_amount=100)
    else:
        found_arch_list_set = sample_arch_candidates(trainer.model, dataset_name, trainer.x_dim, trainer.z_dim, visited,
                                                     sample_amount=100)

    num_new_found = 0
    # Predict accuracy by INN (performance predictor)
    predict_arch_acc(found_arch_list_set, trainer.model)

    # Select top-k to evaluate true label and add to training dataset
    found_arch_list_set = sorted(found_arch_list_set, key=lambda g: g['y'], reverse=True)[:top_k]
    acc_list = query_tabular(dataset_name, found_arch_list_set)
    top_acc_list = [i['valid-accuracy'] for i in acc_list]
    top_test_acc_list = [i['test-accuracy'] for i in acc_list]
    for idx, _ in enumerate(found_arch_list_set):
        found_arch_list_set[idx]['y'] = np.array([acc_list[idx]['valid-accuracy']])

    if len(top_acc_list) != 0:
        logger.info('Top acc list: {}'.format(top_acc_list))
        logger.info('Top test acc list: {}'.format(top_test_acc_list))
        logger.info(f'Avg found acc {sum(top_acc_list) / len(top_acc_list)}')
        logger.info(f'Best found acc {max(top_acc_list)}')
        logger.info(f'Avg found test acc {sum(top_test_acc_list) / len(top_test_acc_list)}')
        logger.info(f'Best found test acc {max(top_test_acc_list)}')
    else:
        logger.info('Top acc list is [] in this run')

    # Add top found architecture to training dataset
    if dataset_name == 'nb101':
        valid_visited = {nb101_dataset.get_spec_hash(i.a, np.argmax(i.x, axis=-1)): i.y.tolist() for i in datasets['valid'].graphs if not np.isnan(i.y).any()}
    else:
        valid_visited = {graph_to_str(i): i.y.tolist() for i in datasets['valid'].graphs if not np.isnan(i.y).any()}

    for i in found_arch_list_set:
        if dataset_name == 'nb101':
            a, x = mask_padding_vertex_for_spec(i['a'], i['x'])
            graph_str = nb101_dataset.get_spec_hash(a, np.argmax(x, axis=-1))
        else:
            graph_str = graph_to_str(i)

        if graph_str not in visited:
            if graph_str not in valid_visited:
                logger.info(f'Data not in train and not in top_list {i["y"].tolist()}')
                num_new_found += 1
            else:
                logger.info(f'Data in valid but not in train {i["y"].tolist()}')

            datasets['train'].graphs.extend([Graph(x=i['x'], a=i['a'], y=i['y'])] * repeat)
            logger.info(f'Add to train {i["x"].tolist()} {i["a"].tolist()} {i["y"].tolist()}')

    logger.info(f'{datasets["train"]}')

    loader = to_loader(datasets, batch_size, train_epochs)
    callbacks = [CSVLogger(os.path.join(logdir, f"learning_curve_phase2_retrain.csv")),
                 # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_total_loss', factor=0.1, patience=15, verbose=1,
                 #                                     min_lr=1e-6),
                 EarlyStopping(monitor='val_total_loss', patience=10, restore_best_weights=True)]

    # tf.keras.backend.set_value(trainer.optimizer.learning_rate, 1e-3)
    trainer.fit(loader['train'].load(),
                validation_data=loader['valid'].load(),
                epochs=train_epochs,
                callbacks=callbacks,
                steps_per_epoch=loader['train'].steps_per_epoch,
                validation_steps=loader['valid'].steps_per_epoch)

    results = trainer.evaluate(loader['test'].load(), steps=loader['test'].steps_per_epoch)
    logger.info(str(dict(zip(trainer.metrics_names, results))))

    return top_acc_list, top_test_acc_list, found_arch_list_set, num_new_found


def prepare_model(nvp_config, latent_dim, num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs,
                  dropout_rate, eps_scale):
    pretrained_model = GraphAutoencoder(latent_dim=latent_dim, num_layers=num_layers,
                                        d_model=d_model, num_heads=num_heads,
                                        dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                        num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=eps_scale)
    pretrained_model(
        (tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))))

    model = GraphAutoencoderNVP(nvp_config=nvp_config, latent_dim=latent_dim, num_layers=num_layers,
                                d_model=d_model, num_heads=num_heads,
                                dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=eps_scale)
    model((tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))))

    retrain_model = GraphAutoencoderNVP(nvp_config=nvp_config, latent_dim=latent_dim, num_layers=num_layers,
                                        d_model=d_model, num_heads=num_heads,
                                        dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                        num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=eps_scale)
    retrain_model((tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))))

    return pretrained_model, model, retrain_model


def main(seed, dataset_name, train_sample_amount, valid_sample_amount, query_budget, top_k, finetune, retrain_finetune, is_rank_weight, random_sample):
    logdir, logger = get_logdir_and_logger(
        os.path.join(f'{train_sample_amount}_{valid_sample_amount}_{query_budget}_finetune{finetune}_rfinetune{retrain_finetune}_rank{is_rank_weight}',
                     dataset_name), f'trainGAE_two_phase_{seed}.log')
    random_seed = seed
    set_global_determinism(random_seed)

    is_only_validation_data = True
    train_phase = [0, 1]  # 0 not train, 1 train
    if dataset_name == 'nb101':
        pretrained_weight = 'logs/phase1_nb101_CE_64/modelGAE_weights_phase1'
    else:
        pretrained_weight = 'logs/phase1_nb201_CE_64/modelGAE_weights_phase1'

    eps_scale = 0.05  # 0.1
    d_model = 32
    dropout_rate = 0.0
    dff = 256
    num_layers = 3
    num_heads = 3

    latent_dim = 16

    if dataset_name == 'nb101':
        num_ops = len(OP_PRIMITIVES_NB101)  # 5
        num_nodes = 7
        num_adjs = num_nodes ** 2
        if os.path.exists('datasets/NasBench101Dataset.cache'):
            datasets = pickle.load(open('datasets/NasBench101Dataset.cache', 'rb'))
        else:
            datasets = NasBench101Dataset(start=0, end=423623)
            with open('datasets/NasBench101Dataset.cache', 'wb') as f:
                pickle.dump(datasets, f)
        datasets = train_valid_test_split_dataset(datasets,
                                                  ratio=[0.8, 0.1, 0.1],
                                                  shuffle=True,
                                                  shuffle_seed=random_seed)
    else:
        # 15624
        num_ops = len(OP_PRIMITIVES_NB201)  # 7
        num_nodes = 8
        num_adjs = num_nodes ** 2
        label_epochs = 200
        if os.path.exists(f'datasets/NasBench201Dataset_{dataset_name}.cache'):
            datasets = pickle.load(open(f'datasets/NasBench201Dataset_{dataset_name}.cache', 'rb'))
        else:
            datasets = NasBench201Dataset(start=0, end=15624, dataset=dataset_name, hp=str(label_epochs), seed=False)
            with open(f'datasets/NasBench201Dataset_{dataset_name}.cache', 'wb') as f:
                pickle.dump(datasets, f)
        datasets = train_valid_test_split_dataset(datasets,
                                                  ratio=[0.8, 0.1, 0.1],
                                                  shuffle=True,
                                                  shuffle_seed=random_seed)

    for key in datasets:
        if is_only_validation_data:
            datasets[key].apply(OnlyValidAccTransform())
            datasets[key].apply(OnlyFinalAcc())
            if dataset_name != 'nb101':
                datasets[key].apply(LabelScale(scale=0.01))
        else:
            datasets[key].apply(ReshapeYTransform())

    global random_arch_generator
    random_arch_generator = RandomArchGenerator(dataset_name, is_only_validation_data)

    x_dim = latent_dim * num_nodes
    y_dim = 1  # 1
    z_dim = x_dim - 1  # 127
    # z_dim = latent_dim * 4 - 1
    tot_dim = y_dim + z_dim  # 28
    # pad_dim = tot_dim - x_dim  # 14

    nvp_config = {
        'n_couple_layer': 4,
        'n_hid_layer': 4,
        'n_hid_dim': 128,
        'name': 'NVP',
        'num_couples': 2,
        'inp_dim': tot_dim
    }

    pretrained_model, model, retrain_model = prepare_model(nvp_config, latent_dim, num_layers, d_model, num_heads, dff,
                                                           num_ops, num_nodes, num_adjs, dropout_rate, eps_scale)
    model.summary(print_fn=logger.info)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    if train_phase[0]:
        logger.info('Train phase 1')
        train_epochs = 500
        patience = 100
        batch_size = 64
        '''
        pretrained_datasets = copy.deepcopy(datasets)
        pretrained_datasets['train'] = mask_graph_dataset(pretrained_datasets['train'], int(42362 * 0.9), 1, random_seed=random_seed)
        pretrained_datasets['valid'] = mask_graph_dataset(pretrained_datasets['valid'], int(42362 * 0.1), 1, random_seed=random_seed)
        pretrained_datasets['train'].filter(lambda g: not np.isnan(g.y))
        pretrained_datasets['valid'].filter(lambda g: not np.isnan(g.y))
        '''
        loader = to_loader(datasets, batch_size, train_epochs)
        callbacks = [CSVLogger(os.path.join(logdir, "learning_curve_phase1.csv")),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_rec_loss', factor=0.1, patience=patience // 2,
                                                          verbose=1, min_lr=1e-5),
                     tensorboard_callback,
                     EarlyStopping(monitor='val_rec_loss', patience=patience, restore_best_weights=True)]
        trainer = train(1, pretrained_model, loader, train_epochs, logdir, callbacks)
        results = trainer.evaluate(loader['test'].load(), steps=loader['test'].steps_per_epoch)
        logger.info(f'{dict(zip(trainer.metrics_names, results))}')
    else:
        pretrained_model.load_weights(pretrained_weight)

    # Load AE weights from pretrained model
    model.encoder.set_weights(pretrained_model.encoder.get_weights())
    model.decoder.set_weights(pretrained_model.decoder.get_weights())

    global_top_acc_list = []
    global_top_test_acc_list = []
    global_top_arch_list = []
    record_top = {'valid': [], 'test': []}

    if train_phase[1]:
        batch_size = 64
        train_epochs = 500
        retrain_epochs = 50
        patience = 50
        repeat_label = 20
        now_queried = train_sample_amount + valid_sample_amount
        logger.info('Train phase 2')
        datasets['train_1'] = mask_graph_dataset(datasets['train'], train_sample_amount, 1, random_seed=random_seed)
        datasets['valid_1'] = mask_graph_dataset(datasets['valid'], valid_sample_amount, 1, random_seed=random_seed)
        datasets['train_1'].filter(lambda g: not np.isnan(g.y))
        datasets['valid_1'].filter(lambda g: not np.isnan(g.y))
        datasets['train'] = repeat_graph_dataset_element(datasets['train_1'], repeat_label)
        datasets['valid'] = repeat_graph_dataset_element(datasets['valid_1'], repeat_label)

        # Add initial data to records
        acc_list = query_tabular(dataset_name, datasets['train_1'])
        global_top_acc_list.extend([i['valid-accuracy'] for i in acc_list])
        global_top_test_acc_list.extend([i['test-accuracy'] for i in acc_list])
        acc_list = query_tabular(dataset_name, datasets['valid_1'])
        global_top_acc_list.extend([i['valid-accuracy'] for i in acc_list])
        global_top_test_acc_list.extend([i['test-accuracy'] for i in acc_list])

        loader = to_loader(datasets, batch_size, train_epochs)
        callbacks = [CSVLogger(os.path.join(logdir, f"learning_curve_phase2.csv")),
                     # tensorboard_callback,
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_total_loss', factor=0.1, patience=patience // 2,
                                                          verbose=1, min_lr=1e-5),
                     EarlyStopping(monitor='val_total_loss', patience=patience, restore_best_weights=True)]
        trainer = train(2, model, loader, train_epochs, logdir, callbacks,
                        x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, finetune=finetune)
        results = trainer.evaluate(loader['test'].load(), steps=loader['test'].steps_per_epoch)
        logger.info(str(dict(zip(trainer.metrics_names, results))))
        #exit()
        # Recreate Trainer for retrain
        retrain_model.set_weights(model.get_weights())
        trainer = Trainer2(retrain_model, x_dim, y_dim, z_dim, finetune=retrain_finetune, is_rank_weight=is_rank_weight)
        trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), run_eagerly=False)

        # Reset the lr for retrain
        run = 0
        while now_queried < query_budget and run <= 100:
            logger.info('')
            logger.info(f'Retrain run {run}')
            top_acc_list, top_test_acc_list, top_arch_list, num_new_found = retrain(trainer, datasets, dataset_name,
                                                                                    batch_size, retrain_epochs, logdir,
                                                                                    logger, repeat_label, top_k, random_sample)
            now_queried += num_new_found
            logger.info('Now queried: ' + str(now_queried))

            if now_queried > query_budget:
                break

            global_top_acc_list += top_acc_list
            global_top_test_acc_list += top_test_acc_list
            global_top_arch_list += top_arch_list
            run += 1

            record_top['valid'].append({now_queried: sorted(global_top_acc_list, reverse=True)[:5]})
            record_top['test'].append({now_queried: sorted(global_top_test_acc_list, reverse=True)[:5]})

            logger.info(f'History top 5 acc: {sorted(global_top_acc_list, reverse=True)[:5]}')
            logger.info(f'History top 5 test acc: {sorted(global_top_test_acc_list, reverse=True)[:5]}')
            # if patience_cot >= patience:
            #    break

            target_acc = {'cifar10-valid': 0.9160, 'cifar100': 0.7349, 'ImageNet16-120': [0.4673, 0.4731], 'nb101': 0.9505}
            if dataset_name == 'ImageNet16-120':
                if max(global_top_acc_list) > target_acc.get(dataset_name, 1.0)[0] and max(global_top_test_acc_list) > target_acc.get(dataset_name, 1.0)[1]:
                    logger.info(f'Find optimal query amount {now_queried}')
                    break
            else:
                if max(global_top_acc_list) > target_acc.get(dataset_name, 1.0):
                    logger.info(f'Find optimal query amount {now_queried}')
                    break
    else:
        model.load_weights(pretrained_weight)

    # invalid, avg_acc, best_acc, found_arch_list = eval_query_best(model, x_dim, z_dim, query_amount=10)
    logger.info('Final result')
    logger.info(f'Best found acc {max(global_top_acc_list)}')
    logger.info(f'Best test acc {max(global_top_test_acc_list)}')
    return max(global_top_acc_list), max(global_top_test_acc_list), record_top


if __name__ == '__main__':
    args = parse_args()
    main(args.seed, args.dataset, args.train_sample_amount, args.valid_sample_amount, args.query_budget,
         args.top_k, args.finetune, args.retrain_finetune, args.rank_weight, args.random_sample)
