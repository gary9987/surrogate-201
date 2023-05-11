import pickle
import random

import numpy as np
from tqdm import tqdm
from models.GNN import GraphAutoencoder, GraphAutoencoderNVP, weighted_mse, get_rank_weight
import tensorflow as tf
from utils.tf_utils import to_undiredted_adj


def main():
    random_seed = 0
    tf.random.set_seed(random_seed)
    random.seed(random_seed)


    model_file_p = 'logs/cifar10-valid/test20/model.pkl'
    model_file_q = 'logs/cifar10-valid/test50/model.pkl'
    datasets_file = 'logs/cifar10-valid/test50/datasets.pkl'

    with open(model_file_p, 'rb') as f:
        p_model: GraphAutoencoderNVP = pickle.load(f)

    with open(model_file_q, 'rb') as f:
        q_model: GraphAutoencoderNVP = pickle.load(f)

    with open(datasets_file, 'rb') as f:
        datasets = pickle.load(f)



    print('dataset', datasets['train_1'])

    tf.config.run_functions_eagerly(False)
    def get_pred_list(model, dataset):
        x = tf.stack([tf.constant(data.x) for data in dataset])
        a = tf.stack([tf.constant(data.a) for data in dataset])
        xa = (x, to_undiredted_adj(a))
        _, _, _, y_out, _ = model(xa, training=False)
        pred_list = [float(y[-1]) for y in y_out]
        return pred_list

    def check_all_smaller_than_eps(pred_list, noise_pred_list, eps):
        for i in range(len(pred_list)):
            #print(pred_list[i], noise_pred_list[i])
            if abs(pred_list[i] - noise_pred_list[i]) > eps:
                return False
        return True

    p_std = 0.00020849203426585205
    q_std = 0.00015682785664062503
    std = min(q_std, p_std) ** 2
    std = 1

    p_weight_list = p_model.nvp.get_weights()
    p_mean = []
    q_mean = []
    for l in range(len(p_weight_list) - 2):
        p_mean.extend(np.reshape(p_weight_list[l], -1).tolist())

    q_weight_list = q_model.nvp.get_weights()
    for l in range(len(q_weight_list) - 2):
        q_mean.extend(np.reshape(q_weight_list[l], -1).tolist())

    std_inv = np.array([1. / std] * len(p_mean)).astype(np.float32).reshape(1, -1)
    p_mean = np.array(p_mean).astype(np.float32).reshape(-1, 1)
    q_mean = np.array(q_mean).astype(np.float32).reshape(-1, 1)
    p_minus_q = p_mean - q_mean
    kl = 0.5 * (np.dot(p_minus_q.T * std_inv, p_minus_q))
    print('kl', kl)

if __name__ == '__main__':
    main()
