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
        #print(y, pred_y)
        mse += (pred_y - y) ** 2 / n
    return mse


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


if __name__ == '__main__':
    logdir = 'logs/cifar10-valid/aggregate30nvp'
    logging.basicConfig(filename=f'{logdir}/cal_upper_bound.log', level=logging.INFO)

    with open(f'{logdir}/model.pkl', 'rb') as f:
        model: GraphAutoencoderEnsembleNVP = pickle.load(f)
    with open(f'{logdir}/datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
    with open(f'{logdir}/theta.pkl', 'rb') as f:
        theta = pickle.load(f)

    weight_list = []
    for nvp in model.nvp_list:
        weight = nvp.get_weights()
        weight_list.append(np.array(weight))

    mean_weight = np.mean(weight_list, axis=0).tolist()


    eps_scale = 0.05  # 0.1
    d_model = 32
    dropout_rate = 0.0
    dff = 256
    num_layers = 3
    num_heads = 3
    finetune = True
    retrain_finetune = True
    latent_dim = 16
    num_nodes = 8
    num_ops = 7
    num_adjs = num_nodes ** 2
    x_dim = latent_dim * num_nodes
    y_dim = 1  # 1
    z_dim = x_dim - 1  # 27
    # z_dim = latent_dim * 4 - 1
    tot_dim = y_dim + z_dim  # 28
    # pad_dim = tot_dim - x_dim  # 14

    num_nvp = 30

    #min_loss_nvp_idx = get_min_loss_nvp_idx(datasets, model)
    #logging.info(f'min_loss_nvp_idx {min_loss_nvp_idx}')

    weight_mse = []
    for i in range(num_nvp):
        c = (weight_list[i] - mean_weight) ** 2
        c = [np.mean(i) for i in c]
        weight_mse.append(np.mean(c))
    logging.info(f'{weight_mse}')
    logging.info(f'np.argmin(weight_mse) {np.argmin(weight_mse)}')

    weight_list = []
    for l in range(len(model.nvp_list[0].get_weights())):
        tmp = None
        shape = model.nvp_list[0].get_weights()[l].shape
        for nvp in model.nvp_list:
            if tmp is None:
                tmp = np.reshape(nvp.get_weights()[l], (-1, 1))
            else:
                tmp = np.concatenate((tmp, np.reshape(nvp.get_weights()[l], (-1, 1))), axis=-1)  # (num_params, num_nvp)

        mean = np.mean(tmp, axis=-1)
        std = np.std(tmp, axis=-1)
        sample_weight = np.random.normal(loc=mean, scale=std, size=(np.shape(mean)[0]))
        #sample_weight = np.random.normal(loc=mean, scale=std, size=(100, np.shape(mean)[0]))
        #sample_weight = np.mean(sample_weight, axis=0)
        weight_list.append(np.reshape(sample_weight, shape))

    model.nvp_list[0].set_weights(weight_list)
    mse = 0
    num_nvp = len(model.nvp_list)
    n = len(datasets['train_1'])
    for idx, data in enumerate(datasets['train_1']):
        y = data.y[-1]
        xa = (tf.constant([data.x]), to_undiredted_adj(tf.constant([data.a])))
        #xa = (tf.constant([data.x]), tf.constant([data.a]))
        ops_cls, adj_cls, kl_loss, y_out, x_encoding = model(xa, training=False)
        pred_y = 0
        pred_y += float(y_out[0][0][-1])
        print(y, pred_y)
        mse += (pred_y - y) ** 2 / n
    print(mse)
    '''
    bound = get_bound(datasets['train_1'], model)

    logging.info(f'bound {bound}')
    logging.info(f'bound^2 {bound ** 2}')
    '''
    mse = after_theta_mse(datasets['train_1'], model, theta)
    logging.info(f'after theta mse {mse}')