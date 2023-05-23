import os
import pickle
import random
import tensorflow_probability as tfp
from models.GNN import GraphAutoencoder, GraphAutoencoderNVP_BNN, weighted_mse, get_rank_weight
import tensorflow as tf
from utils.tf_utils import to_undiredted_adj


def get_KL_between_two_models(model_p, model_q):
    std = 0.72

    p_weights = model_p.nvp.weights[:-2]
    q_weights = model_q.nvp.weights[:-2]

    def compute_kl(w):
        p_weight, q_weight = w[0], w[1]  # (2 , 64, 64)

        p = tfp.distributions.Normal(
            loc=p_weight,
            scale=tf.fill(tf.shape(p_weight), std)
        )
        q = tfp.distributions.Normal(
            loc=q_weight,
            scale=tf.fill(tf.shape(q_weight), std)
        )
        kl = tfp.distributions.kl_divergence(p, q)
        return tf.reduce_sum(kl)

    '''
    for i in range(len(model_p.nvp.weights[:-2])):
        if 'kernel' in model_p.nvp.weights[i].name and 'loc' in model_p.nvp.weights[i].name:
            p = tfp.distributions.Normal(loc=model_p.nvp.weights[i], scale=tf.fill(tf.shape(model_q.nvp.weights[i]), 0.72))
            q = tfp.distributions.Normal(loc=model_q.nvp.weights[i], scale=tf.fill(tf.shape(model_q.nvp.weights[i]), 0.72))
            kl = tfp.distributions.kl_divergence(p, q)
            kl_loss += tf.reduce_sum(kl)
    '''
    w = []
    for i in range(len(p_weights)):
        if 'loc' in model_p.nvp.weights[i].name:
            if len(tf.shape(p_weights[i])) == 2:
                w.append(tf.stack([p_weights[i], q_weights[i]]))  # (2 , 64, 64)

    w = tf.stack(w) # N, 2, 64, 64
    kl_values = tf.vectorized_map(
        compute_kl,
        w
    )
    kl_loss = tf.reduce_sum(kl_values)

    w = []
    for i in range(len(p_weights)):
        if 'loc' in model_p.nvp.weights[i].name:
            if len(tf.shape(p_weights[i])) == 1:
                w.append(tf.stack([p_weights[i], q_weights[i]]))

    w = tf.stack(w)  # N, 2, 64, 64
    kl_values = tf.vectorized_map(
        compute_kl,
        w
    )
    kl_loss += tf.reduce_sum(kl_values)
    '''
    for weight in model_p.nvp.weights[:-2]:
        if 'kernel' in weight.name and 'loc' in weight.name:
            p_mean.extend(weight.numpy().flatten().tolist())

    q_mean = []
    q_std = []
    for weight in model_q.nvp.weights[:-2]:
        if 'kernel' in weight.name and 'loc' in weight.name:
            q_mean.extend(weight.numpy().flatten().tolist())
        elif 'kernel' in weight.name and 'scale' in weight.name:
            q_std.extend(weight.numpy().flatten().tolist())
        else:
            # raise ValueError('Unknown weight name: {}'.format(weight.name))
            pass

    pdf_model1 = tfp.distributions.Normal(loc=p_mean, scale=[0.72] * len(p_mean))
    pdf_model2 = tfp.distributions.Normal(loc=q_mean, scale=[0.72] * len(p_mean))
    kl = tfp.distributions.kl_divergence(pdf_model2, pdf_model1)
    '''
    return kl_loss


def get_pred_result(dataset, model):
    x = tf.stack([tf.constant(data.x) for data in dataset])
    a = tf.stack([tf.constant(data.a) for data in dataset])
    xa = (x, to_undiredted_adj(a))
    _, _, _, y_out, _ = model(xa, training=True)  # bs, 128
    return tf.reshape(y_out[:, -1], -1)


def get_mse_loss(dataset, model):
    pred_y_list = get_pred_result(dataset, model)
    true_y_list = tf.constant([data.y for data in dataset])
    mse = tf.keras.losses.mse(true_y_list, pred_y_list)
    return tf.reduce_mean(mse)


def kl_bound(model_p, model_q, dataset):
    kl_value = get_KL_between_two_models(model_q, model_p)
    sample_size = float(len(dataset))
    B = (kl_value + tf.math.log(2 * tf.math.sqrt(sample_size) / 0.05)) / sample_size
    L_S = get_mse_loss(dataset, model_q)
    term1 = L_S + B + tf.math.sqrt(B * (B * 2 * L_S))
    term2 = L_S + tf.math.sqrt(B / 2)
    return L_S, tf.reduce_min([term1, term2])


def main():
    random_seed = 0
    tf.random.set_seed(random_seed)
    random.seed(random_seed)


    model_file_p = 'logs/cifar10-valid/get_bound/model_ghost.pkl'
    model_file_q = 'logs/cifar10-valid/get_bound/model_full.pkl'
    datasets_file = 'logs/cifar10-valid/get_bound/datasets.pkl'

    with open(model_file_p, 'rb') as f:
        p_model: GraphAutoencoderNVP_BNN = pickle.load(f)

    with open(model_file_q, 'rb') as f:
        q_model: GraphAutoencoderNVP_BNN = pickle.load(f)

    with open(datasets_file, 'rb') as f:
        datasets = pickle.load(f)

    print(datasets)

    '''
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
    '''

    print(kl_bound(p_model, q_model, datasets['train_post']))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    main()
