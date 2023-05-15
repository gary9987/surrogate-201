import pickle
import random
import numpy as np
from models.GNN import GraphAutoencoderEnsembleNVP
import tensorflow as tf
from scipy.stats import gaussian_kde, entropy
from scipy.special import kl_div


def main():
    random_seed = 0
    tf.random.set_seed(random_seed)
    random.seed(random_seed)


    model_file_p = 'logs/cifar10-valid/aggregate20_tmp_full/model.pkl'
    model_file_q = 'logs/cifar10-valid/aggregate50_tmp/model.pkl'
    datasets_file = 'logs/cifar10-valid/aggregate50_tmp/datasets.pkl'

    with open(model_file_p, 'rb') as f:
        p_model: GraphAutoencoderEnsembleNVP = pickle.load(f)

    with open(model_file_q, 'rb') as f:
        q_model: GraphAutoencoderEnsembleNVP = pickle.load(f)

    with open(datasets_file, 'rb') as f:
        datasets = pickle.load(f)


    print('dataset', datasets['train_1'])

    tf.config.run_functions_eagerly(False)
    p_pdf = []
    q_pdf = []
    p_weights = [nvp.get_weights()[:-2] for nvp in p_model.nvp_list]  # 10, 48
    q_weights = [nvp.get_weights()[:-2] for nvp in q_model.nvp_list]  # 10, 48

    for l in range(len(p_weights[0])):
        p_matrix = np.stack([p_weights[i][l].flatten() for i in range(len(p_weights))])
        q_matrix = np.stack([q_weights[i][l].flatten() for i in range(len(q_weights))])

        x = [np.linspace(min(np.min(p_matrix[i]), np.min(q_matrix[i])), max(np.max(p_matrix[i]), np.max(q_matrix[i])), num=1000) for i in range(p_matrix.shape[0])]
        p_kde = gaussian_kde(p_matrix)
        p_pdf.append(p_kde.evaluate(x))

        q_kde = gaussian_kde(q_matrix)
        q_pdf.append(q_kde.evaluate(x))

    kl = 0.
    for i, j in zip(p_pdf, q_pdf):
        kl += tf.losses.KLDivergence()(i, j)
        print(tf.losses.KLDivergence()(i, j))

    print(kl)


if __name__ == '__main__':
    main()
