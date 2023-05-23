import os
import pickle
from math import sqrt, exp, log
from models.GNN import GraphAutoencoderEnsembleNVP, GraphAutoencoderNVP, GraphAutoencoder
import numpy as np
import tensorflow as tf
from utils.tf_utils import to_undiredted_adj
import logging


def after_theta_mse(dataset, model, theta):
    mse = 0
    num_nvp = len(model.nvp_list)
    n = len(dataset)
    for idx, data in enumerate(dataset):
        y = data.y[-1]
        xa = (tf.constant([data.x]), to_undiredted_adj(tf.constant([data.a])))
        ops_cls, adj_cls, kl_loss, y_out, x_encoding = model(xa, training=False)
        pred_y = 0
        for i in range(num_nvp):
            pred_y += float(y_out[0][i][-1]) * theta[i]
        mse += (pred_y - y) ** 2 / n
    return mse


def individual_mse(dataset, model):
    num_nvp = len(model.nvp_list)
    mse_list = [0] * num_nvp
    n = len(dataset)
    for idx, data in enumerate(dataset):
        y = data.y[-1]
        xa = (tf.constant([data.x]), to_undiredted_adj(tf.constant([data.a])))
        ops_cls, adj_cls, kl_loss, y_out, x_encoding = model(xa, training=False)
        for i in range(num_nvp):
            mse_list[i] += (float(y_out[0][i][-1]) - y) ** 2 / n

    return mse_list


def get_min_loss_nvp_idx(dataset, model, pred_y_list):
    num_nvp = len(model.nvp_list)
    sq_err = np.zeros((num_nvp, len(dataset)))
    for idx, data in enumerate(dataset):
        y = data.y[-1]
        #xa = (tf.constant([data.x]), to_undiredted_adj(tf.constant([data.a])))
        #ops_cls, adj_cls, kl_loss, y_out, x_encoding = model(xa)
        for i in range(num_nvp):
            #print(y, float(y_out[0][i][-1]))
            sq_err[i][idx] = (y - float(pred_y_list[idx][i])) ** 2

    #logging.info(f'{np.mean(sq_err, axis=-1)}')
    #logging.info(f'np.argmin(np.mean(sq_err, axis=-1)) {np.argmin(np.mean(sq_err, axis=-1))}')
    min_loss_nvp_idx = np.argmin(np.mean(sq_err, axis=-1))
    return min_loss_nvp_idx


def get_pred_result(dataset, model):
    pred_y_list = []
    for data in dataset:
        xa = (tf.constant([data.x]), to_undiredted_adj(tf.constant([data.a])))
        _, _, _, y_out, _ = model(xa, training=False)
        y_out = tf.squeeze(y_out)
        pred_y_list.append(y_out[:, -1])  # (num_nvp, 1)
    return pred_y_list


def get_bound(dataset, model):
    num_nvp = len(model.nvp_list)
    pred_y_list = get_pred_result(dataset, model)
    min_loss_nvp_idx = get_min_loss_nvp_idx(dataset, model, pred_y_list)
    # First term
    first_term = 0
    n = len(dataset)
    logging.info(f'data len = {n}')

    for idx, data in enumerate(dataset):
        y = data.y[-1]
        #xa = (tf.constant([data.x]), to_undiredted_adj(tf.constant([data.a])))
        #ops_cls, adj_cls, kl_loss, y_out, x_encoding = model(xa)
        first_term += (1 / n) * ((y - float(pred_y_list[idx][min_loss_nvp_idx])) ** 2)
    first_term = sqrt(first_term)

    # Rem term
    rem_term = 0
    beta = 1.
    for i in range(num_nvp):
        mse  = 0
        for idx, data in enumerate(dataset):
            #xa = (tf.constant([data.x]), to_undiredted_adj(tf.constant([data.a])))
            #ops_cls, adj_cls, kl_loss, y_out, x_encoding = model(xa)
            mse += (1 / n) * ((float(pred_y_list[idx][min_loss_nvp_idx]) - float(pred_y_list[idx][i])) ** 2)

        rem_term += (1 / num_nvp) * exp(-1 * (n / beta) * mse)

    rem_term = sqrt(-1 * (beta / n) * log(rem_term))
    return first_term + rem_term


def get_theta_list(model, dataset, beta=1. + 1e-15):
    num_nvp = len(model.nvp_list)
    n = len(dataset)
    theta_list = [tf.constant([1. / num_nvp] * num_nvp)]

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
            for m in range(i + 1):
                if m == i:
                    loss_cache[m][k] = tf.keras.losses.mean_squared_error(dataset[m].y, regs[m][k][-1])
                loss += loss_cache[m][k]

            under_q += tf.cast(tf.exp(-(beta ** -1) * loss), tf.float32)

        upper_q = 0
        for m in range(i + 1):
            if m == i:
                y = dataset[m].y
                y = tf.expand_dims(y, axis=0)
                y = tf.expand_dims(y, axis=0)
                y = tf.repeat(y, num_nvp, axis=1)
                loss_cache2.append(tf.keras.losses.mean_squared_error(y, regs[m][:, -1:]))
            upper_q += loss_cache2[m]

        upper_q = tf.exp(-(beta ** -1) * upper_q)
        theta_list.append(tf.squeeze(upper_q / under_q))

    return theta_list


def get_theta(model, dataset, beta=1. + 1e-15):
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
            for m in range(i + 1):
                if m == i:
                    loss_cache[m][k] = tf.keras.losses.mean_squared_error(dataset[m].y, regs[m][k][-1])
                loss += loss_cache[m][k]

            under_q += tf.cast(tf.exp(-(beta ** -1) * loss), tf.float32)

        upper_q = 0
        for m in range(i + 1):
            if m == i:
                y = dataset[m].y
                y = tf.expand_dims(y, axis=0)
                y = tf.expand_dims(y, axis=0)
                y = tf.repeat(y, num_nvp, axis=1)
                loss_cache2.append(tf.keras.losses.mean_squared_error(y, regs[m][:, -1:]))
            upper_q += loss_cache2[m]

        upper_q = tf.exp(-(beta ** -1) * upper_q)
        theta += (upper_q / under_q) / n

    return tf.squeeze(theta)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    logdir = 'logs/aggregate50_10'
    logging.basicConfig(filename=f'{logdir}/cal_upper_bound.log', level=logging.INFO)

    with open(f'{logdir}/model.pkl', 'rb') as f:
        model: GraphAutoencoderEnsembleNVP = pickle.load(f)
    with open(f'{logdir}/datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
    #with open(f'{logdir}/theta.pkl', 'rb') as f:
    #    theta = pickle.load(f)

    # Calculate paper: learning by mirror average formula (5.5) left term
    expected_mse = 0
    theta_list = get_theta_list(model, datasets['train_1'])
    for i in range(1, len(datasets['train_1']) + 1):
        expected_mse += after_theta_mse(datasets['train_1'][:i], model, theta_list[i-1]) / len(datasets['train_1'])

    logging.info(f'expected mse {expected_mse}')
    # 4.533987521426752e-05
    # 4.537969289231114e-05

    mse_list = individual_mse(datasets['train_1'], model)
    logging.info(f'individual mse {mse_list}')
    logging.info(f'individual mse min {np.min(mse_list)}')

    theta = get_theta(model, datasets['train_1'])
    print(theta)
    print(after_theta_mse(datasets['valid_1'], model, theta))
    # 0.0035248573
    # 0.0035256199
    '''
    bound = get_bound(datasets['train_1'], model)

    logging.info(f'bound {bound}')
    logging.info(f'bound^2 {bound ** 2}')
    '''
    #mse = after_theta_mse(datasets['train_1'], model, theta)
    #logging.info(f'after theta mse {mse}')