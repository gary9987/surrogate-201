import pickle
import random
import numpy as np
from keras.losses import mse
from tqdm import tqdm
from models.GNN import GraphAutoencoder, GraphAutoencoderNVP, weighted_mse, get_rank_weight
import tensorflow as tf
from utils.tf_utils import to_undiredted_adj


def get_pred_list(model, dataset):
    x = tf.stack([tf.constant(data.x) for data in dataset])
    a = tf.stack([tf.constant(data.a) for data in dataset])
    xa = (x, to_undiredted_adj(a))
    _, _, _, y_out, _ = model(xa, training=False)
    pred_list = [float(y[-1]) for y in y_out]
    return pred_list


def log_posterior(model, dataset, noise=0.1, prior_std=1.0):
    # Compute predictions
    y_pred = np.array(get_pred_list(model, dataset))
    y = np.array([graph.y[-1] for graph in dataset])
    # Compute log-likelihood
    log_likelihood = -0.5 * np.sum((y - y_pred) ** 2) / (noise ** 2)

    # Compute log-prior (using Gaussian prior)
    nvp_weight = model.nvp.get_weights()
    weights = np.concatenate([nvp_weight[l].flatten() for l in range(len(model.nvp.weights) - 2)])
    log_prior = -0.5 * np.sum(weights ** 2) / (prior_std ** 2)

    return log_likelihood + log_prior


def metropolis_hastings(model, dataset, noise, n_samples=1000, proposal_std=0.01):
    # Get the initial weights of the model
    tf_var = model.nvp.get_weights()[-2:]
    current_weights = model.nvp.get_weights()

    # Initialize samples list
    samples = []

    # Perform Metropolis-Hastings sampling
    for _ in tqdm(range(n_samples)):
        # Propose new weights by adding Gaussian noise
        proposed_weights = [w + proposal_std * np.random.randn(*w.shape) for w in current_weights][:-2] + tf_var

        # Compute the log-posterior for the proposed and current weights
        log_posterior_current = log_posterior(model, dataset, noise=noise)
        # Set the proposed weights to the model
        model.nvp.set_weights(proposed_weights)
        log_posterior_proposed = log_posterior(model, dataset, noise=noise)

        # Compute the acceptance probability
        accept_prob = min(1, np.exp(log_posterior_proposed - log_posterior_current))

        # Accept or reject the proposal
        if np.random.rand() < accept_prob:
            current_weights = proposed_weights
        else:
            # Set the current weights back to the model
            model.nvp.set_weights(current_weights)

        # Append the accepted weights to the samples list
        samples.append(current_weights)

    return samples


def get_mcmc_std_mean(model, dataset, n_samples, noise):
    # Sample the weight distribution using Metropolis-Hastings
    samples = metropolis_hastings(model, dataset, n_samples=n_samples, noise=noise)
    # Initialize empty lists to store the posterior standard deviations
    posterior_std_weights = []
    posterior_mean_weights = []
    # Iterate through the layers and compute the posterior standard deviation
    for layer_index, _ in enumerate(model.nvp.weights[:-2]):
        # Compute the posterior standard deviation for the weights and biases
        a = [sample[layer_index] for sample in samples]
        weight_std = np.std([sample[layer_index] for sample in samples], axis=0)
        weight_mean = np.mean([sample[layer_index] for sample in samples], axis=0)
        # Append the standard deviations to the lists
        posterior_std_weights.append(weight_std)
        posterior_mean_weights.append(weight_mean)

    return posterior_std_weights, posterior_mean_weights


def main():
    random_seed = 0
    tf.random.set_seed(random_seed)
    random.seed(random_seed)


    model_file_p = 'logs/cifar10-valid/testp/model.pkl'
    model_file_q = 'logs/cifar10-valid/test50/model.pkl'
    datasets_file = 'logs/cifar10-valid/test50/datasets.pkl'

    with open(model_file_p, "rb") as f:
        p_model: GraphAutoencoderNVP = pickle.load(f)

    with open(model_file_q, "rb") as f:
        q_model: GraphAutoencoderNVP = pickle.load(f)

    with open(datasets_file, "rb") as f:
        datasets = pickle.load(f)

    print('dataset', datasets['train_1'])

    tf.config.run_functions_eagerly(False)
    mcmc_sample = False
    if mcmc_sample:
        # Initialize empty lists to store the posterior standard deviations
        posterior_std_weights, posterior_mean_weights = get_mcmc_std_mean(p_model, datasets['train_1'][:20], n_samples=1000, noise=0.1)
        print("Posterior standard deviation (weights):", posterior_std_weights)
        print("Posterior mean (weights):", posterior_mean_weights)

        with open('logs/cifar10-valid/test50/p_mean.pkl', 'wb') as f:
            pickle.dump(posterior_mean_weights, f)
        with open('logs/cifar10-valid/test50/p_std.pkl', 'wb') as f:
            pickle.dump(posterior_std_weights, f)

        # Initialize empty lists to store the posterior standard deviations
        posterior_std_weights, posterior_mean_weights = get_mcmc_std_mean(q_model, datasets['train_1'], n_samples=1000, noise=0.1)
        print("Posterior standard deviation (weights):", posterior_std_weights)
        print("Posterior mean (weights):", posterior_mean_weights)

        with open('logs/cifar10-valid/test50/q_mean.pkl', 'wb') as f:
            pickle.dump(posterior_mean_weights, f)
        with open('logs/cifar10-valid/test50/q_std.pkl', 'wb') as f:
            pickle.dump(posterior_std_weights, f)


    with open('logs/cifar10-valid/test50/p_mean.pkl', 'rb') as f:
        p_mean = pickle.load(f)
    with open('logs/cifar10-valid/test50/p_std.pkl', 'rb') as f:
        p_std = pickle.load(f)
    with open('logs/cifar10-valid/test50/q_mean.pkl', 'rb') as f:
        q_mean = pickle.load(f)
    with open('logs/cifar10-valid/test50/q_std.pkl', 'rb') as f:
        q_std = pickle.load(f)
    '''
    q_mean.extend(q_model.nvp.get_weights()[-2:])
    q_model.nvp.set_weights(q_mean)
    pred_y = get_pred_list(q_model, datasets['train_1'])
    ori_y = [float(graph.y[-1]) for graph in datasets['train_1']]
    print(mse(ori_y, pred_y))
    '''
    sample_weights = []
    for l, _ in enumerate(p_mean):
        shape = p_mean[l].shape
        matrix = np.random.normal(p_mean[l].flatten(), p_std[l].flatten(), p_mean[l].flatten().shape)
        matrix = matrix.reshape(shape)
        sample_weights.append(matrix)

    sample_weights.extend(p_model.nvp.get_weights()[-2:])
    p_model.nvp.set_weights(sample_weights)
    pred_y = get_pred_list(p_model, datasets['train_1'])
    ori_y = [float(graph.y[-1]) for graph in datasets['train_1']]
    print(mse(ori_y, pred_y))

    p_mean_tmp = []
    for i in p_mean:
        p_mean_tmp.extend(i.flatten().tolist())
    p_mean = np.array(p_mean_tmp).reshape(-1, 1)
    p_std_tmp = []
    for i in p_std:
        p_std_tmp.extend(i.flatten().tolist())
    p_std = np.array(p_std_tmp).reshape(1, -1)
    q_mean_tmp = []
    for i in q_mean:
        q_mean_tmp.extend(i.flatten().tolist())
    q_mean = np.array(q_mean_tmp).reshape(-1, 1)
    q_std_tmp = []
    for i in q_std:
        q_std_tmp.extend(i.flatten().tolist())
    q_std = np.array(q_std_tmp).reshape(1, -1)

    q_var = np.square(q_std)
    p_var = np.square(p_std)

    p_mean = p_mean.flatten().tolist()
    p_std = p_std.flatten().tolist()
    q_mean = q_mean.flatten().tolist()
    q_std = q_std.flatten().tolist()
    '''
    kl_list = []
    for mean1, std1, mean2, std2 in zip(p_mean, p_std, q_mean, q_std):
        kl_divergence = np.log(std2 / std1) + (std1 ** 2 + (mean1 - mean2) ** 2) / (2 * std2 ** 2) - 0.5
        kl_list.append(kl_divergence)

    print('mean kl', np.mean(kl_list))
    '''
    mean1 = np.mean(p_mean)
    std1 = np.std(p_mean)
    mean2 = np.mean(q_mean)
    std2 = np.std(q_mean)
    print(mean1, std1, mean2, std2)
    kl_divergence = np.log(std2 / std1) + (std1 ** 2 + (mean1 - mean2) ** 2) / (2 * std2 ** 2) - 0.5
    print('kl', np.mean(kl_divergence))

    '''
    q_minus_p = q_mean - p_mean
    q_var_div_p_var = q_var / p_var
    q_var_div_p_var = np.prod(q_var_div_p_var)
    term1 = (1. / q_var) * p_var
    term1 = np.sum(term1)
    term2 = float(np.dot(q_minus_p.T * (1. / q_var),  q_minus_p))
    term3 = p_mean.shape[0]
    #term4 = np.log(q_var_div_p_var)
    term4 = 0
    kl_p_q = 0.5 * (term1 + term2 - term3 + term4)
    print('kl_p_q', kl_p_q)

    ori_weight = q_model.nvp.get_weights()
    sample_weight = []

    for l, _ in enumerate(ori_weight[:-2]):
        ori = ori_weight[l].flatten()
        q_flat = q_mean.flatten()[l]

        sample = np.zeros(ori.shape)
        for _ in range(1000):
            r_sample = np.random.normal(q_mean.flatten()[l], q_std.flatten()[l], size=ori.shape)
            sample += np.sort(r_sample)
        sample /= 1000

        ori_arg_sort = np.argsort(ori).tolist()
        sample_arg_sort = np.argsort(sample).tolist()
        new_sample = np.zeros(ori.shape)
        for i in range(sample.shape[0]):
            new_sample[ori_arg_sort[i]] = sample[i]
        new_sample = np.reshape(np.array(new_sample), ori_weight[l].shape)
        sample_weight.append(new_sample)

    sample_weight.extend(ori_weight[-2:])
    q_model.nvp.set_weights(sample_weight)
    pred_y = get_pred_list(q_model, datasets['train_1'])
    ori_y = [float(graph.y[-1]) for graph in datasets['train_1']]
    print(mse(ori_y, pred_y))
    '''


if __name__ == '__main__':
    main()
