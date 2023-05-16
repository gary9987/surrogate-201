import argparse
import copy
import random
import numpy as np
from tensorflow.python.keras.callbacks import CSVLogger, EarlyStopping

from datasets.nb101_dataset import OP_PRIMITIVES_NB101, NasBench101Dataset
from datasets.transformation import ReshapeYTransform, OnlyValidAccTransform, OnlyFinalAcc, LabelScale
from invertible_neural_networks.flow import MMD_multiscale
from models.GNN import GraphAutoencoder, GraphAutoencoderEnsembleNVP, get_rank_weight
from models.TransformerAE import TransformerAutoencoderNVP
import tensorflow as tf
import os
from datasets.nb201_dataset import NasBench201Dataset, OP_PRIMITIVES_NB201
from datasets.utils import train_valid_test_split_dataset, mask_graph_dataset, arch_list_to_set, graph_to_str
from evalGAE import query_acc_by_ops, ensemble_eval_query_best
from trainGAE_two_phase import to_loader, mask_for_model, mask_for_spec
from utils.py_utils import get_logdir_and_logger
from spektral.data import Graph
from utils.tf_utils import to_undiredted_adj
from cal_upper_bound import after_theta_mse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sample_amount', type=int, default=50, help='Number of samples to train (default: 50)')
    parser.add_argument('--valid_sample_amount', type=int, default=50, help='Number of samples to train (default: 50)')
    parser.add_argument('--query_budget', type=int, default=192)
    parser.add_argument('--dataset', type=str, default='cifar10-valid', help='Could be nb101, cifar10-valid, cifar100, ImageNet16-120')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls):
    ops_label, adj_label = x_batch_train
    #adj_label = tf.reshape(adj_label, [tf.shape(adj_label)[0], -1])
    #ops_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(ops_label, ops_cls)
    ops_loss = tf.keras.losses.KLDivergence()(ops_label, ops_cls)
    adj_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(adj_label, adj_cls)
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
            reduction = 'none'
        else:
            reduction = 'auto'

        self.reg_loss_fn = tf.keras.losses.MeanSquaredError(reduction=reduction)
        self.loss_latent = MMD_multiscale
        self.loss_backward = tf.keras.losses.MeanSquaredError(reduction=reduction)
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
        # y: (batch_size, y_dim)
        # y_out: (batch_size, num_nvp, y_dim + z_dim)
        y = tf.boolean_mask(y, nan_mask)
        y = tf.expand_dims(y, axis=1)
        y = tf.repeat(y, repeats=tf.shape(y_out)[1], axis=1)
        z = tf.boolean_mask(z, nan_mask)
        z = tf.expand_dims(z, axis=1)
        z = tf.repeat(z, repeats=tf.shape(y_out)[1], axis=1)
        pred_y = tf.boolean_mask(y_out[:, :, self.z_dim:], nan_mask)
        pred_z = tf.boolean_mask(y_out[:, :, :self.z_dim], nan_mask)

        reg_loss = self.reg_loss_fn(y, pred_y)

        zy = tf.transpose(tf.concat([z, y], axis=-1), (1, 0, 2))  # (num_nvp, batch_size, y_dim + z_dim)
        pred_zy = tf.transpose(tf.concat([pred_z, pred_y], axis=-1), (1, 0, 2)) # (num_nvp, batch_size, y_dim + z_dim)
        latent_loss = tf.reduce_mean(tf.vectorized_map(lambda x: self.loss_latent(x[0], x[1]), (zy, pred_zy)))

        if self.is_rank_weight:
            # reg_loss (batch_size, num_nvp)
            reg_loss = tf.multiply(reg_loss, tf.broadcast_to(tf.expand_dims(rank_weight, -1), tf.shape(reg_loss)))
            reg_loss = tf.reduce_sum(tf.reduce_mean(reg_loss, axis=-1))
        return reg_loss, latent_loss

    def cal_rev_loss(self, undirected_x_batch_train, y, z, nan_mask, noise_scale, rank_weight=None):
        y = tf.boolean_mask(y, nan_mask)
        z = tf.boolean_mask(z, nan_mask)
        y = y + noise_scale * tf.random.normal(shape=tf.shape(y), dtype=tf.float32)
        non_nan_x_batch = (tf.boolean_mask(undirected_x_batch_train[0], nan_mask),
                           tf.boolean_mask(undirected_x_batch_train[1], nan_mask))
        _, _, _, _, x_encoding = self.model(non_nan_x_batch, training=True)  # Logits for this minibatch
        x_rev = self.model.inverse(tf.concat([z, y], axis=-1))  # (batch_size, num_nvp, x_dim)
        x_encoding = tf.expand_dims(x_encoding, axis=1)
        x_encoding = tf.repeat(x_encoding, repeats=tf.shape(x_rev)[1], axis=1)
        rev_loss = self.loss_backward(x_encoding, x_rev)  # * x_batch_train.shape[0]
        if self.is_rank_weight:
            # rev_loss (batch_size, num_nvp)
            tf.multiply(rev_loss, tf.broadcast_to(tf.expand_dims(rank_weight, -1), tf.shape(rev_loss)))
            rev_loss = tf.reduce_sum(tf.reduce_mean(rev_loss, axis=-1))
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
            ops_cls, adj_cls, kl_loss, y_out, x_encoding = self.model(undirected_x_batch_train, training=True)

            # To avoid nan loss when batch size is small
            reg_loss, latent_loss = tf.cond(tf.reduce_any(nan_mask),
                                            lambda: self.cal_reg_and_latent_loss(y, z, y_out, nan_mask, rank_weight),
                                            lambda: (0., 0.))

            forward_loss = self.w1 * reg_loss + self.w2 * latent_loss
            rec_loss = 0.
            if self.finetune:
                ops_loss, adj_loss = cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls)
                rec_loss = self.ops_weight * ops_loss + self.adj_weight * adj_loss + self.kl_weight * kl_loss
                forward_loss += rec_loss

        grads = tape.gradient(forward_loss, self.model.trainable_weights)
        #grads = [tf.clip_by_norm(g, 1.) for g in grads]
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
        #grads = [tf.clip_by_norm(g, 1.) for g in grads]
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

        ops_cls, adj_cls, kl_loss, y_out, x_encoding = self.model(undirected_x_batch_train, training=False)
        reg_loss, latent_loss = tf.cond(tf.reduce_any(nan_mask),
                                        lambda: self.cal_reg_and_latent_loss(y, z, y_out, nan_mask, rank_weight),
                                        lambda: (0., 0.))
        forward_loss = self.w1 * reg_loss + self.w2 * latent_loss
        rev_loss = tf.cond(tf.reduce_any(nan_mask),
                           lambda: self.cal_rev_loss(undirected_x_batch_train, y, z, nan_mask, 0., rank_weight),
                           lambda: 0.)
        backward_loss = self.w3 * rev_loss
        if self.finetune:
            ops_loss, adj_loss = cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls)
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
          z_dim=None, finetune=False, learning_rate=1e-3, no_valid=False):
    if phase == 1:
        trainer = Trainer1(model)
    elif phase == 2:
        trainer = Trainer2(model, x_dim, y_dim, z_dim, finetune)
    else:
        raise ValueError('phase should be 1 or 2')

    try:
        if no_valid:
            kw = {'steps_per_epoch': loader['train'].steps_per_epoch}
        else:
            kw = {'validation_steps': loader['valid'].steps_per_epoch,
                  'steps_per_epoch': loader['train'].steps_per_epoch}
    except:
        kw = {}

    if phase == 2 and no_valid:
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(learning_rate, train_epochs * loader['train'].steps_per_epoch)

    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), run_eagerly=False)
    if no_valid:
        trainer.fit(loader['train'].load(),
                    epochs=train_epochs,
                    callbacks=callbacks,
                    **kw)
    else:
        trainer.fit(loader['train'].load(),
                    validation_data=loader['valid'].load(),
                    epochs=train_epochs,
                    callbacks=callbacks,
                    **kw)
    model.save_weights(os.path.join(logdir, f'modelGAE_weights_phase{phase}'))
    return trainer


def retrain(trainer, datasets, dataset_name, batch_size, train_epochs, logdir, top_list, logger, repeat, top_k=5):
    # Generate total 10(num_nvp) * query_amount architectures
    _, _, _, found_arch_list = ensemble_eval_query_best(trainer.model, dataset_name, trainer.x_dim, trainer.z_dim,
                                                        query_amount=200 // trainer.model.num_nvp)

    theta = get_theta(trainer.model, datasets['train_1'])
    logger.info(f'Theta: {theta.numpy().tolist()}')

    if dataset_name == 'nb101':
        found_arch_list = list(map(mask_for_model, found_arch_list))
        found_arch_list = list(filter(lambda arch: arch is not None and arch['x'] is not None, found_arch_list))

    found_arch_list_set = arch_list_to_set(found_arch_list)
    logger.info(f'Length of found_arch_list_set {len(found_arch_list_set)}')
    top_acc_list = []
    top_test_acc_list = []
    # Predict accuracy by INN (performance predictor)
    x = tf.stack([tf.constant(i['x']) for i in found_arch_list_set])
    a = tf.stack([tf.constant(i['a']) for i in found_arch_list_set])
    if tf.shape(x)[0] != 0:
        _, _, _, reg, _ = trainer.model((x, to_undiredted_adj(a)), training=False)  # (batch, num_nvp, z_dim+y_dim)
        reg = reg[:, :, -1]  # (batch, num_nvp)
        theta_expanded = tf.expand_dims(theta, axis=0)  # (1, num_nvp)
        reg = reg * tf.tile(theta_expanded, (tf.shape(reg)[0], 1))
        reg = tf.reduce_sum(reg, axis=-1)  # (batch, )
        for i in range(len(found_arch_list_set)):
            # Assign the predicted accuracy to the architecture
            found_arch_list_set[i]['y'] = reg[i]

    found_arch_list_set = sorted(found_arch_list_set, key=lambda g: g['y'], reverse=True)

    def get_new_archs_and_add_to_dataset(arch_list_set):
        num_new_found = 0
        found_arch_list_set = copy.deepcopy(arch_list_set)

        # Select top-k to evaluate true label and add to training dataset
        found_arch_list_set = sorted(found_arch_list_set, key=lambda g: g['y'], reverse=True)[:top_k]
        for idx, i in enumerate(found_arch_list_set):
            if dataset_name != 'nb101':
                acc = query_acc_by_ops(i['x'], dataset_name)
                test_acc = query_acc_by_ops(i['x'], dataset_name, on='test-accuracy')
            else:
                i = mask_for_spec(i)
                acc = float(datasets['train'].get_metrics(i['a'], np.argmax(i['x'], axis=-1))[1])
            top_acc_list.append(acc)
            top_test_acc_list.append(test_acc)
            found_arch_list_set[idx]['y'] = np.array([acc])

        if len(top_acc_list) != 0:
            logger.info('Top acc list: {}'.format(top_acc_list))
            logger.info('Top test acc list: {}'.format(top_test_acc_list))
            logger.info(f'Avg found acc {sum(top_acc_list) / len(top_acc_list)}')
            logger.info(f'Best found acc {max(top_acc_list)}')
            logger.info(f'Avg found test acc {sum(top_test_acc_list) / len(top_test_acc_list)}')
            logger.info(f'Best found test acc {max(top_test_acc_list)}')
        else:
            logger.info('Top acc list is [] in this run')

        train_dict = {graph_to_str(i): i.y.tolist() for i in datasets['train'].graphs}

        # Add top found architecture to training dataset
        for i in found_arch_list_set:
            graph_str = graph_to_str(i)
            if graph_str in train_dict:
                if graph_str not in top_list and np.isnan(train_dict[graph_str]):
                    datasets['train'].graphs.extend([Graph(x=i['x'], a=i['a'], y=i['y'])] * repeat)
                    datasets['train_1'].graphs.extend([Graph(x=i['x'], a=i['a'], y=i['y'])])
                    top_list.append(graph_str)
                    logger.info(f'Data not in train and not in top_list {i["y"].tolist()}')
                    num_new_found += 1
                elif graph_str not in top_list and not np.isnan(train_dict[graph_str]):
                    logger.info(f'Data in train but not in top_list {i["y"].tolist()}')
                    top_list.append(graph_str)
                else:
                    logger.info(f'Data in train and in top_list {i["y"].tolist()}')
            else:
                if graph_str not in top_list:
                    logger.info(f'Data not in train and not in top_list {i["y"].tolist()}')
                    datasets['train'].graphs.extend([Graph(x=i['x'], a=i['a'], y=i['y'])] * repeat)
                    datasets['train_1'].graphs.extend([Graph(x=i['x'], a=i['a'], y=i['y'])])
                    top_list.append(graph_str)
                    num_new_found += 1

        return num_new_found, found_arch_list_set

    num_new_found, top_arch_list_set = get_new_archs_and_add_to_dataset(found_arch_list_set)
    if num_new_found == 0:
        logger.info('No new architecture found, filter the found_arch_list_set')
        train_graph_set = [graph_to_str(i) for i in datasets['train_1'].graphs]
        found_arch_list_set = list(filter(lambda arch: graph_to_str(arch) not in train_graph_set, found_arch_list_set))
        logger.info(f'Length of found_arch_list_set after filter {len(found_arch_list_set)}')
        num_new_foundm, top_arch_list_set = get_new_archs_and_add_to_dataset(found_arch_list_set)

    logger.info(f'{datasets["train"]}')
    logger.info(f'{datasets["train_1"]}')
    logger.info(f'Length of top_list {len(top_list)}')

    loader = to_loader(datasets, batch_size, train_epochs)
    callbacks = [CSVLogger(os.path.join(logdir, f"learning_curve_phase2_retrain.csv")),
                 #tf.keras.callbacks.ReduceLROnPlateau(monitor='val_total_loss', factor=0.1, patience=15, verbose=1,
                 #                                     min_lr=1e-6),
                 EarlyStopping(monitor='val_total_loss', patience=10, restore_best_weights=True)
                 ]

    #tf.keras.backend.set_value(trainer.optimizer.learning_rate, 1e-3)
    trainer.fit(loader['train'].load(),
                validation_data=loader['valid'].load(),
                epochs=train_epochs,
                callbacks=callbacks,
                steps_per_epoch=loader['train'].steps_per_epoch,
                validation_steps=loader['valid'].steps_per_epoch
                )

    results = trainer.evaluate(loader['test'].load(), steps=loader['test'].steps_per_epoch)
    logger.info(str(dict(zip(trainer.metrics_names, results))))

    return top_acc_list, top_test_acc_list, top_arch_list_set, num_new_found


def get_theta(model, dataset, beta=1.):
    num_nvp = len(model.nvp_list)
    n = len(dataset)
    theta = 0

    x = tf.stack([tf.constant(i.x, dtype=tf.float32) for i in dataset])
    a = tf.stack([tf.constant(i.a, dtype=tf.float32) for i in dataset])
    a = to_undiredted_adj(a)
    _, _, _, regs, _ = model((x, a), training=False)
    # regs: (n, num_nvp, z_dim+y_dim)

    loss_cache = np.zeros((n, num_nvp)).astype(np.float32)
    loss_cache2 = []
    for i in range(n):
        under_q = 0
        for k in range(num_nvp):
            loss = 0.
            for m in range(i+1):
                if m == i:
                    loss_cache[m][k] = tf.keras.losses.mean_squared_error(dataset[m].y, regs[m][k][-1])
                loss += loss_cache[m][k]

            under_q += tf.cast(tf.exp(-(beta**-1) * loss), tf.float32)

        upper_q = 0
        for m in range(i+1):
            if m == i:
                y = dataset[m].y
                y = tf.expand_dims(y, axis=0)
                y = tf.expand_dims(y, axis=0)
                y = tf.repeat(y, num_nvp, axis=1)
                loss_cache2.append(tf.keras.losses.mean_squared_error(y, regs[m][:, -1:]))
            upper_q += loss_cache2[m]

        upper_q = tf.exp(-(beta**-1) * upper_q)
        theta += (upper_q / under_q) / n

    return tf.squeeze(theta)


def prepare_model(num_nvp, nvp_config, latent_dim, num_layers, d_model, num_heads, dff, num_ops, num_nodes, num_adjs, dropout_rate, eps_scale):
    pretrained_model = GraphAutoencoder(latent_dim=latent_dim, num_layers=num_layers,
                                        d_model=d_model, num_heads=num_heads,
                                        dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                        num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=eps_scale)
    pretrained_model((tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))))

    model = GraphAutoencoderEnsembleNVP(num_nvp, nvp_config=nvp_config, latent_dim=latent_dim, num_layers=num_layers,
                                d_model=d_model, num_heads=num_heads,
                                dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=eps_scale)
    model((tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))))

    retrain_model = GraphAutoencoderEnsembleNVP(num_nvp, nvp_config=nvp_config, latent_dim=latent_dim, num_layers=num_layers,
                                        d_model=d_model, num_heads=num_heads,
                                        dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                        num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=eps_scale)
    retrain_model((tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))))

    return pretrained_model, model, retrain_model


def main(seed, dataset_name, train_sample_amount, valid_sample_amount, query_budget):
    logdir, logger = get_logdir_and_logger(dataset_name, f'trainGAE_ensemble_{seed}.log')
    random_seed = seed
    tf.random.set_seed(random_seed)
    random.seed(random_seed)

    top_k = 5

    is_only_validation_data = True
    train_phase = [0, 1]  # 0 not train, 1 train
    if dataset_name == 'nb101':
        pretrained_weight = 'logs/nb101/nb101_phase1/modelGAE_weights_phase1'
    else:
        pretrained_weight = 'logs/phase1_model_cifar100/modelGAE_weights_phase1'

    eps_scale = 0.05  # 0.1
    d_model = 32
    dropout_rate = 0.0
    dff = 256
    num_layers = 3
    num_heads = 3
    finetune = False
    retrain_finetune = False
    latent_dim = 16

    train_epochs = 500
    retrain_epochs = 50
    patience = 50

    if dataset_name == 'nb101':
        num_ops = len(OP_PRIMITIVES_NB101)  # 5
        num_nodes = 7
        num_adjs = num_nodes ** 2
        datasets = train_valid_test_split_dataset(NasBench101Dataset(start=0, end=423623),
                                                  ratio=[0.8, 0.1, 0.1],
                                                  shuffle=True,
                                                  shuffle_seed=0)
    else:
        # 15624
        num_ops = len(OP_PRIMITIVES_NB201)  # 7
        num_nodes = 8
        num_adjs = num_nodes ** 2
        label_epochs = 200
        datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, dataset=dataset_name,
                                                                     hp=str(label_epochs), seed=False),
                                                  ratio=[0.8, 0.1, 0.1],
                                                  shuffle=True,
                                                  shuffle_seed=0)

    for key in datasets:
        if is_only_validation_data:
            datasets[key].apply(OnlyValidAccTransform())
            datasets[key].apply(OnlyFinalAcc())
            if dataset_name != 'nb101':
                datasets[key].apply(LabelScale(scale=0.01))
        else:
            datasets[key].apply(ReshapeYTransform())

    x_dim = latent_dim * num_nodes
    y_dim = 1  # 1
    z_dim = x_dim - 1  # 127
    #z_dim = latent_dim * 4 - 1
    tot_dim = y_dim + z_dim  # 28
    #pad_dim = tot_dim - x_dim  # 14

    num_nvp = 10
    nvp_config = {
        'n_couple_layer': 2,
        'n_hid_layer': 4,
        'n_hid_dim': 64,
        'name': 'NVP',
        'inp_dim': tot_dim
    }

    pretrained_model, model, retrain_model = prepare_model(num_nvp, nvp_config, latent_dim, num_layers, d_model, num_heads, dff,
                                                           num_ops, num_nodes, num_adjs, dropout_rate, eps_scale)
    model.summary(print_fn=logger.info)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    if train_phase[0]:
        logger.info('Train phase 1')
        batch_size = 256
        loader = to_loader(datasets, batch_size, train_epochs)
        callbacks = [CSVLogger(os.path.join(logdir, "learning_curve_phase1.csv")),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_rec_loss', factor=0.1, patience=50, verbose=1,
                                                          min_lr=1e-5),
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
    global_top_test_acc_list =[]
    global_top_arch_list = []
    history_top = 0
    patience_cot = 0
    find_optimal_query = 0
    find_optimal = False

    if train_phase[1]:
        batch_size = 64
        repeat_label = 20
        now_queried = train_sample_amount + valid_sample_amount
        logger.info('Train phase 2')
        datasets['train_1'] = mask_graph_dataset(datasets['train'], train_sample_amount, 1, random_seed=random_seed)
        datasets['valid_1'] = mask_graph_dataset(datasets['valid'], valid_sample_amount, 1, random_seed=random_seed)
        datasets['train_1'].filter(lambda g: not np.isnan(g.y))
        datasets['valid_1'].filter(lambda g: not np.isnan(g.y))
        datasets['train'] = mask_graph_dataset(datasets['train'], train_sample_amount, repeat_label, random_seed=random_seed)
        datasets['valid'] = mask_graph_dataset(datasets['valid'], valid_sample_amount, repeat_label, random_seed=random_seed)
        datasets['train'].filter(lambda g: not np.isnan(g.y))
        datasets['valid'].filter(lambda g: not np.isnan(g.y))

        loader = to_loader(datasets, batch_size, train_epochs)
        callbacks = [CSVLogger(os.path.join(logdir, f"learning_curve_phase2.csv")),
                     #tensorboard_callback,
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_total_loss', factor=0.1, patience=patience // 2, verbose=1,
                                                          min_lr=1e-5),
                     EarlyStopping(monitor='val_total_loss', patience=patience, restore_best_weights=True)
                     ]
        trainer = train(2, model, loader, train_epochs, logdir, callbacks,
                        x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, finetune=finetune, learning_rate=1e-3, no_valid=False)
        results = trainer.evaluate(loader['test'].load(), steps=loader['test'].steps_per_epoch)
        logger.info(str(dict(zip(trainer.metrics_names, results))))

        # Recreate Trainer for retrain
        retrain_model.set_weights(model.get_weights())
        trainer = Trainer2(retrain_model, x_dim, y_dim, z_dim, finetune=retrain_finetune, is_rank_weight=False)
        trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), run_eagerly=False)

        top_list = []
        run = 0

        while now_queried < query_budget and run <= 200:
            logger.info('')
            logger.info(f'Retrain run {run}')
            top_acc_list, top_test_acc_list, top_arch_list, num_new_found = retrain(trainer, datasets, dataset_name, batch_size,
                                                                 retrain_epochs, logdir, top_list, logger,
                                                                 repeat_label, top_k)
            now_queried += num_new_found
            if now_queried > query_budget:
                break
            global_top_acc_list += top_acc_list
            global_top_test_acc_list += top_test_acc_list
            global_top_arch_list += top_arch_list
            run += 1

            if len(top_acc_list) != 0 and max(top_acc_list) > history_top:
                history_top = max(top_acc_list)
                patience_cot = 0
            else:
                patience_cot += 1

            logger.info(f'History top acc: {max(global_top_acc_list)}')
            logger.info(f'History top test acc: {max(global_top_test_acc_list)}')
            #if patience_cot >= patience:
            #    break

            target_acc = {'cifar10-valid': 0.9160, 'cifar100': 0.7349}
            if not find_optimal and max(global_top_acc_list) > target_acc.get(dataset_name, 1.0):
                find_optimal_query = now_queried
                find_optimal = True
                break

    else:
        model.load_weights(pretrained_weight)

    logger.info('Final result')
    logger.info(f'Find optimal query amount {find_optimal_query}')
    logger.info(f'Avg found acc {sum(global_top_acc_list) / len(global_top_acc_list)}')
    logger.info(f'Best found acc {max(global_top_acc_list)}')
    logger.info(f'Avg test acc {sum(global_top_test_acc_list) / len(global_top_test_acc_list)}')
    logger.info(f'Best test acc {max(global_top_test_acc_list)}')

    return sum(global_top_acc_list) / len(global_top_acc_list), max(global_top_acc_list), sum(top_test_acc_list) / len(top_test_acc_list), max(top_test_acc_list)


if __name__ == '__main__':
    args = parse_args()
    main(args.seed, args.dataset, args.train_sample_amount, args.valid_sample_amount, args.query_budget)
