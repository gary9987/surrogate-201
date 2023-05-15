import pickle
import random
import tensorflow_probability as tfp
import numpy as np
import pylab as p
from tqdm import tqdm
from models.GNN import GraphAutoencoder, GraphAutoencoderNVP_BNN, weighted_mse, get_rank_weight
import tensorflow as tf
from utils.tf_utils import to_undiredted_adj


def main():
    random_seed = 0
    tf.random.set_seed(random_seed)
    random.seed(random_seed)


    model_file_p = 'logs/cifar10-valid/BNN_1_nvp_20/model.pkl'
    model_file_q = 'logs/cifar10-valid/BNN_1_nvp_50/model.pkl'
    datasets_file = 'logs/cifar10-valid/BNN_1_nvp_50/datasets.pkl'

    with open(model_file_p, 'rb') as f:
        p_model: GraphAutoencoderNVP_BNN = pickle.load(f)

    m = p_model.nvp.layers[0].nn1.layer_list[0]
    s = p_model.nvp.layers[0].nn1.layer_list[0].kernel_posterior.stddev()
    with open(model_file_q, 'rb') as f:
        q_model: GraphAutoencoderNVP_BNN = pickle.load(f)

    with open(datasets_file, 'rb') as f:
        datasets = pickle.load(f)

    print('dataset', datasets['train'])


    p_mean = []
    p_std = []
    for weight in p_model.nvp.weights[:-2]:
        if 'kernel' in weight.name and 'loc' in weight.name:
            p_mean.extend(weight.numpy().flatten().tolist())
        elif 'kernel' in weight.name and 'scale' in weight.name:
            p_std.extend(weight.numpy().flatten().tolist())
        else:
            pass
            #raise ValueError('Unknown weight name: {}'.format(weight.name))

    q_mean = []
    q_std = []
    for weight in q_model.nvp.weights[:-2]:
        if 'kernel' in weight.name and 'loc' in weight.name:
            q_mean.extend(weight.numpy().flatten().tolist())
        elif'kernel' in weight.name and 'scale' in weight.name:
            q_std.extend(weight.numpy().flatten().tolist())
        else:
            #raise ValueError('Unknown weight name: {}'.format(weight.name))
            pass

    def KL(mean_p, std_p, mean_q, std_q, num_samples=1000):
        x = np.random.normal(loc=mean_p, scale=std_p, size=num_samples)
        pdf_p = (1 / (std_p * np.sqrt(2 * np.pi))) * np.exp(-(x - mean_p) ** 2 / (2 * std_p ** 2))
        pdf_q = (1 / (std_q * np.sqrt(2 * np.pi))) * np.exp(-(x - mean_q) ** 2 / (2 * std_q ** 2))

        kl = np.sum(pdf_p * np.log(pdf_p / pdf_q))
        return kl

    kl_divergence = []
    for pm, ps, qm, qs in zip(p_mean, p_std, q_mean, q_std):
        #pdf_model1 = tfp.distributions.Normal(loc=pm, scale=tf.math.softplus(ps))
        #pdf_model2 = tfp.distributions.Normal(loc=qm, scale=tf.math.softplus(qs))
        #kl = tfp.distributions.kl_divergence(pdf_model1, pdf_model2)
        kl = KL(pm, tf.math.softplus(ps).numpy(), qm, tf.math.softplus(qs).numpy())
        kl_divergence.append(kl)
        print('kl', kl)
    
    print(np.mean(kl_divergence))
    print(np.sum(kl_divergence))


    #kl_divergence_avg = np.mean(kl_divergence)
    p_std = tf.math.softplus(p_std).numpy()
    q_std = tf.math.softplus(q_std).numpy()
    p_var = np.array(p_std).reshape(1, -1) ** 2
    q_var = np.array(q_std).reshape(1, -1) ** 2
    p_mean = np.array(p_mean).astype(np.float32).reshape(-1, 1)
    q_mean = np.array(q_mean).astype(np.float32).reshape(-1, 1)
    q_minus_p = q_mean - p_mean
    q_var_inv = 1. / q_var
    term2 = np.sum(q_var_inv * p_var)

    #term3 = np.log(np.prod(p_var / q_var))
    term1 = float(np.dot(q_minus_p.T * q_var_inv, q_minus_p))
    kl = 0.5 * (term2 + term1 - p_std.shape[0])


    print('kl', kl)

if __name__ == '__main__':
    main()
