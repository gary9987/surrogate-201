import argparse
import random
import numpy as np
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
from evalGAE import eval_query_best, query_acc_by_ops
from utils.py_utils import get_logdir_and_logger
from spektral.data import Graph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sample_amount', type=int, default=350, help='Number of samples to train (default: 350)')
    parser.add_argument('--valid_sample_amount', type=int, default=50, help='Number of samples to train (default: 50)')
    parser.add_argument('--query_budget', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def cal_ops_adj_loss_for_graph(x_batch_train, ops_cls, adj_cls):
    ops_label, adj_label = x_batch_train
    #adj_label = tf.reshape(adj_label, [tf.shape(adj_label)[0], -1])
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
    def __init__(self, model: TransformerAutoencoderNVP, x_dim, y_dim, z_dim, finetune=False):
        super(Trainer2, self).__init__()
        self.model = model
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.finetune = finetune

        # For reg loss weight
        self.w1 = 5.
        # For latent loss weight
        self.w2 = 1.
        # For rev loss weight
        self.w3 = 10.

        #self.reg_loss_fn = tf.keras.losses.MeanSquaredError()
        self.reg_loss_fn = weighted_mse
        self.loss_latent = MMD_multiscale
        self.loss_backward = MSE
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
            ops_cls, adj_cls, kl_loss, y_out, x_encoding = self.model(undirected_x_batch_train, training=False)

            # To avoid nan loss when batch size is small
            reg_loss, latent_loss = tf.cond(tf.shape(nan_mask)[0] != 0,
                                            lambda: self.cal_reg_and_latent_loss(y, z, y_out, nan_mask),
                                            lambda: (0., 0.))
            ''' Original code for Eagerly execution
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
                rec_loss = self.ops_weight * ops_loss + self.adj_weight * adj_loss + self.kl_weight * kl_loss
                forward_loss += rec_loss

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
            self.loss_tracker['ops_loss'].update_state(ops_loss)
            self.loss_tracker['adj_loss'].update_state(adj_loss)
            self.loss_tracker['kl_loss'].update_state(kl_loss)

        return {key: value.result() for key, value in self.loss_tracker.items()}

    def test_step(self, data):
        x_batch_train, y_batch_train = data
        undirected_x_batch_train = (x_batch_train[0], to_undiredted_adj(x_batch_train[1]))
        y = y_batch_train[:, -self.y_dim:]
        z = tf.random.normal(shape=[tf.shape(y_batch_train)[0], self.z_dim])
        nan_mask = tf.squeeze(tf.where(~tf.math.is_nan(tf.reduce_sum(y_batch_train, axis=-1)), x=0, y=1))

        ops_cls, adj_cls, kl_loss, y_out, x_encoding = self.model(undirected_x_batch_train, training=False)
        reg_loss, latent_loss = tf.cond(tf.shape(nan_mask)[0] != 0,
                                        lambda: self.cal_reg_and_latent_loss(y, z, y_out, nan_mask),
                                        lambda: (0., 0.))
        forward_loss = self.w1 * reg_loss + self.w2 * latent_loss
        rev_loss = tf.cond(tf.shape(nan_mask)[0] != 0,
                           lambda: self.cal_rev_loss(undirected_x_batch_train, y, z, nan_mask, 0.),
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
          z_dim=None, finetune=False, learning_rate=1e-3):
    #optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
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


def retrain(trainer, datasets, batch_size, train_epochs, logdir, top_list, logger, repeat):
    # Generate total 200 architectures
    _, _, _, found_arch_list = eval_query_best(trainer.model, trainer.x_dim,
                                                                  trainer.z_dim, query_amount=100)
    _, _, _, found_arch_list2 = eval_query_best(trainer.model, trainer.x_dim,
                                                                  trainer.z_dim, query_amount=100, noise_scale=0.05)
    num_new_found = 0
    found_arch_list_set = []
    visited = []
    found_arch_list.extend(found_arch_list2)
    for i in found_arch_list:
        if str(i['x'].tolist()) not in visited:
            found_arch_list_set.append(i)
            visited.append(str(i['x'].tolist()))

    # Predict accuracy by INN (performance predictor)
    for idx, i in enumerate(found_arch_list_set):
        _, _, _, reg, _ = trainer.model((tf.constant([i['x']]), tf.constant([i['a']])), training=False)
        found_arch_list_set[idx]['y'] = reg[0][-1].numpy()

    # Select top-5 to evaluate true label and add to training dataset
    top_acc_list = []
    found_arch_list_set = sorted(found_arch_list_set, key=lambda x: x['y'], reverse=True)[:20]
    for idx, i in enumerate(found_arch_list_set):
        acc = query_acc_by_ops(i['x'])
        top_acc_list.append(acc)
        found_arch_list_set[idx]['y'] = np.array([acc])

    logger.info('Top acc list: {}'.format(top_acc_list))
    logger.info(f'Avg found acc {sum(top_acc_list) / len(top_acc_list)}')
    logger.info(f'Best found acc {max(top_acc_list)}')

    train_dict = {str(i.x.tolist()): i.y.tolist() for i in datasets['train']}

    # Add top found architecture to training dataset
    for i in found_arch_list_set:
        if str(i['x'].tolist()) in train_dict:
            if i['x'].tolist() not in top_list and train_dict[str(i['x'].tolist())] == [np.nan]:
                datasets['train'].graphs.extend([Graph(x=i['x'], a=i['a'], y=i['y'])] * repeat)
                top_list.append(i['x'].tolist())
                logger.info(f'Data not in training set {i["y"].tolist()}')
                num_new_found += 1
            elif i['x'].tolist() not in top_list and train_dict[str(i['x'].tolist())] != [np.nan]:
                logger.info(f'Data already in training set {i["y"].tolist()}')
                #datasets['train'].graphs.extend([Graph(x=i['x'], a=i['a'], y=i['y'])] * 10)
                top_list.append(i['x'].tolist())
        else:
            if i['x'].tolist() not in top_list:
                logger.info(f'Data not in train and not in top_list {i["y"].tolist()}')
                datasets['train'].graphs.extend([Graph(x=i['x'], a=i['a'], y=i['y'])] * repeat)
                top_list.append(i['x'].tolist())
                num_new_found += 1


    logger.info(f'{datasets["train"]}')
    logger.info(f'Length of top_list {len(top_list)}')

    loader = {'train': BatchLoader(datasets['train'], batch_size=batch_size, shuffle=True, epochs=train_epochs * 2),
              'valid': BatchLoader(datasets['valid'], batch_size=batch_size, shuffle=True, epochs=train_epochs * 2),
              'test': BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)}
    callbacks = [CSVLogger(os.path.join(logdir, f"learning_curve_phase2_retrain.csv")),
                 #tf.keras.callbacks.ReduceLROnPlateau(monitor='val_total_loss', factor=0.1, patience=15, verbose=1,
                 #                                     min_lr=1e-6),
                 EarlyStopping(monitor='val_total_loss', patience=15, restore_best_weights=True)]

    #tf.keras.backend.set_value(trainer.optimizer.learning_rate, 1e-3)
    trainer.fit(loader['train'].load(),
                validation_data=loader['valid'].load(),
                epochs=train_epochs,
                callbacks=callbacks,
                steps_per_epoch=loader['train'].steps_per_epoch,
                validation_steps=loader['valid'].steps_per_epoch)

    results = trainer.evaluate(loader['test'].load(), steps=loader['test'].steps_per_epoch)
    logger.info(str(dict(zip(trainer.metrics_names, results))))
    '''
    invalid, avg_acc, best_acc, found_arch_list = eval_query_best(trainer.model, trainer.x_dim,
                                                                  trainer.z_dim, query_amount=10)
    logger.info('After retrain')
    logger.info(f'Number of invalid decode {invalid}')
    logger.info(f'Avg found acc {avg_acc}')
    logger.info(f'Best found acc {best_acc}')
    '''
    return top_acc_list, found_arch_list_set, num_new_found


def main(seed, train_sample_amount, valid_sample_amount, query_budget=100):
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
    pretrained_weight = 'logs/20230420-165707/modelGAE_weights_phase1'

    retrain_epochs = 20

    eps_scale = 0.05  # 0.1
    d_model = 32
    dropout_rate = 0.0
    dff = 256
    num_layers = 3
    num_heads = 3
    finetune = True
    latent_dim = 16

    train_epochs = 1000
    patience = 200

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

    x_dim = latent_dim * num_nodes
    y_dim = 1  # 1
    z_dim = x_dim - 1  # 27
    #z_dim = latent_dim * 4 - 1
    tot_dim = y_dim + z_dim  # 28
    #pad_dim = tot_dim - x_dim  # 14

    nvp_config = {
        'n_couple_layer': 4,
        'n_hid_layer': 4,
        'n_hid_dim': 256,
        'name': 'NVP',
        'inp_dim': tot_dim
    }
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
    model.summary(print_fn=logger.info)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    if train_phase[0]:
        logger.info('Train phase 1')
        batch_size = 256
        loader = {'train': BatchLoader(datasets['train'], batch_size=batch_size, shuffle=True, epochs=train_epochs * 2),
                  'valid': BatchLoader(datasets['valid'], batch_size=batch_size, shuffle=False, epochs=train_epochs * 2),
                  'test': BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)}
        callbacks = [CSVLogger(os.path.join(logdir, "learning_curve_phase1.csv")),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_rec_loss', factor=0.1, patience=100, verbose=1,
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
    #model.nvp.set_weights(pretrained_model.nvp.get_weights())
    model.decoder.set_weights(pretrained_model.decoder.get_weights())

    global_top_acc_list = []
    global_top_arch_list = []
    if train_phase[1]:
        batch_size = 256
        repeat_label = 80
        now_queried = train_sample_amount + valid_sample_amount
        logger.info('Train phase 2')
        '''
        tmp_train = datasets['train'][:train_sample_amount]
        tmp_valid = datasets['valid'][:valid_sample_amount]
        datasets['train'] = tmp_train
        datasets['valid'] = tmp_valid
        for i in range(20):
            datasets['train'] += tmp_train
            datasets['valid'] += tmp_valid
        '''
        datasets['train'] = mask_graph_dataset(datasets['train'], train_sample_amount, repeat_label, random_seed=random_seed)
        datasets['valid'] = mask_graph_dataset(datasets['valid'], valid_sample_amount, repeat_label, random_seed=random_seed)
        # datasets['train'] = datasets['train'][:args.train_sample_amount]
        # datasets['valid'] = datasets['valid'][:args.valid_sample_amount]
        loader = {'train': BatchLoader(datasets['train'], batch_size=batch_size, shuffle=True, epochs=train_epochs * 2),
                  'valid': BatchLoader(datasets['valid'], batch_size=batch_size, shuffle=True, epochs=train_epochs * 2),
                  'test': BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)}

        callbacks = [CSVLogger(os.path.join(logdir, f"learning_curve_phase2.csv")),
                     tensorboard_callback,
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_total_loss', factor=0.1, patience=100, verbose=1,
                                                          min_lr=1e-5),
                     EarlyStopping(monitor='val_total_loss', patience=patience, restore_best_weights=True)]
        trainer = train(2, model, loader, 20, logdir, callbacks,
                        x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, finetune=finetune)
        results = trainer.evaluate(loader['test'].load(), steps=loader['test'].steps_per_epoch)
        logger.info(str(dict(zip(trainer.metrics_names, results))))
        #trainer = Trainer2(model, x_dim, y_dim, z_dim, finetune=finetune)
        #trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        del trainer
        tf.keras.backend.clear_session()

        trainer = Trainer2(model, x_dim, y_dim, z_dim, finetune=False)
        trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        # Reset the lr for retrain
        tf.keras.backend.set_value(trainer.optimizer.learning_rate, 1e-3)
        top_list = []
        run = 0
        while now_queried < query_budget:
            logger.info('')
            logger.info(f'Retrain run {run}')
            top_acc_list, top_arch_list, num_new_found = retrain(trainer, datasets, batch_size, retrain_epochs, logdir, top_list, logger, repeat_label)
            now_queried += num_new_found
            if now_queried >= query_budget:
                break
            global_top_acc_list += top_acc_list
            global_top_arch_list += top_arch_list
            #if run % 5 == 0:
            #    trainer.model.save_weights(os.path.join(logdir, f'modelGAE_retrain_weights_{run}'))
            run += 1
    else:
        model.load_weights(pretrained_weight)

    #invalid, avg_acc, best_acc, found_arch_list = eval_query_best(model, x_dim, z_dim, query_amount=10)
    logger.info('Final result')
    logger.info(f'Avg found acc {sum(global_top_acc_list) / len(global_top_acc_list)}')
    logger.info(f'Best found acc {max(global_top_acc_list)}')
    top_test_acc_list = []
    for i in global_top_arch_list:
        acc = query_acc_by_ops(i['x'], is_random=False, on='test-accuracy')
        top_test_acc_list.append(acc)
    logger.info(f'Avg test acc {sum(top_test_acc_list) / len(top_test_acc_list)}')
    logger.info(f'Best test acc {max(top_test_acc_list)}')

    return sum(global_top_acc_list) / len(global_top_acc_list), max(global_top_acc_list), sum(top_test_acc_list) / len(top_test_acc_list), max(top_test_acc_list)


if __name__ == '__main__':
    args = parse_args()
    main(args.seed, args.train_sample_amount, args.valid_sample_amount, args.query_budget),