import numpy as np
import tensorflow as tf
from datasets.nb201_dataset import NasBench201Dataset
from datasets.query_nb201 import OPS_by_IDX_201
from datasets.transformation import OnlyValidAccTransform
from datasets.utils import train_valid_test_split_dataset, ops_list_to_nb201_arch_str, to_latent_feature_data
from models.Diffusion import TransformerAutoencoderDiffusion
from nats_bench import create
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

random_seed = 0
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

num_ops = 7
num_nodes = 8
num_adjs = 64

def inverse_from_acc(model: TransformerAutoencoderDiffusion, num_sample_z: int, to_inv_acc):

    batch_size = int(tf.shape(to_inv_acc)[0])

    noise = tf.random.normal(shape=[batch_size, 10, 12, 4])
    for i in range(model.diffusion_steps)[::-1]:
        acc = tf.constant(to_inv_acc, shape=[batch_size, 1])
        t = tf.repeat(tf.constant(i, shape=[1, 1]), repeats=batch_size, axis=0)
        noise = model.denoise(noise, acc, t)

    latent = tf.reshape(noise, (batch_size, -1, model.d_model))
    _, adj, ops_cls, adj_cls = model.decode(tf.reshape(noise, (batch_size, -1, model.d_model)))  # (num_sample_z, input_size(120))

    ops_cls = tf.reshape(ops_cls, (batch_size, num_sample_z, -1, model.num_ops)) # (batch_size, num_sample_z, 8, 7)
    ops_vote = tf.reduce_sum(ops_cls, axis=1).numpy()  # (batch_size, 1, 8 * 7)

    adj = tf.reshape(adj, (batch_size, num_sample_z, model.num_adjs))  # (batch_size, num_sample_z, 8 * 8)
    adj = tf.where(tf.reduce_mean(adj, axis=1) >= 0.5, x=1., y=0.).numpy()  # (batch_size, 8 * 8)
    adj = np.reshape(adj, (batch_size, int(adj.shape[-1]**(1/2)), int(adj.shape[-1]**(1/2))))

    ops_idx_list = []
    adj_list = []
    for i, j in zip(ops_vote, adj):
        ops_idx_list.append(np.argmax(i, axis=-1).tolist())
        adj_list.append(j)

    return ops_idx_list, adj_list


if __name__ == '__main__':
    d_model = 4
    dropout_rate = 0.0
    dff = 512
    num_layers = 3
    num_heads = 3
    diffusion_steps = 500
    input_size = 120

    model = TransformerAutoencoderDiffusion(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                            input_size=input_size, num_ops=num_ops, num_nodes=num_nodes, num_adjs=num_adjs,
                                            diffusion_steps=diffusion_steps, dropout_rate=dropout_rate)

    model.load_weights('logs/trainTAE_diffusion/20230408-010759/modelTAE_diffusion_weights')
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(200), seed=777),
                                              ratio=[0.9, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    for key in datasets:
        datasets[key].apply(OnlyValidAccTransform())

    x_valid, y_valid = to_latent_feature_data(datasets['train'], -1)
    loader = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size=256)

    invalid = 0
    x = []
    y = []
    for _, label_acc in loader:
        #try:
        ops_idx_lis, adj_list = inverse_from_acc(model, 1, to_inv_acc=label_acc[:, -1:])
        for ops_idx, adj, query_acc in zip(ops_idx_lis, adj_list, label_acc[:, -1]):
            try:
                ops_str_list = [OPS_by_IDX_201[i] for i in ops_idx]
                #print(adj)
                arch_str = ops_list_to_nb201_arch_str(ops_str_list)
                #print(arch_str)

                arch_idx = nb201api.query_index_by_arch(arch_str)

                accumulate_acc = 0
                acc_list = [float(query_acc)]
                for seed in [777, 888]:
                    data = nb201api.get_more_info(arch_idx, 'cifar10-valid', iepoch=199, hp='200', is_random=seed)
                    accumulate_acc += data['valid-accuracy'] / 100.
                    acc_list.append(data['valid-accuracy'] / 100.)

                    # data = nb201api.get_more_info(arch_idx, 'cifar10', iepoch=199, hp='200', is_random=seed)
                    # print(data['test-accuracy'])

                x.append(float(query_acc))
                y.append(accumulate_acc / 2)
                print(acc_list)  # [query_acc, 777_acc, 888_acc]
            except:
                print('invalid')
                invalid += 1
    '''
    while to_inv_acc >= 0.60:
        for i in range(20):
            ops_idx, adj = inverse_from_acc(model, num_sample_z=10000, z_dim=120 * d_model-1, to_inv_acc=to_inv_acc)
            ops = [OPS_by_IDX_201[i] for i in ops_idx]
            print(ops)
            print(adj)

            arch_str = ops_list_to_nb201_arch_str(ops)
            print(arch_str)

            nb201api = create(None, 'tss', fast_mode=True, verbose=True)
            idx = nb201api.query_index_by_arch(arch_str)

            acc = 0
            for seed in [777, 888]:
                data = nb201api.get_more_info(idx, 'cifar10-valid', iepoch=199, hp='200', is_random=seed)
                print(data['valid-accuracy'])
                acc += data['valid-accuracy'] / 100.

                data = nb201api.get_more_info(idx, 'cifar10', iepoch=199, hp='200', is_random=seed)
                print(data['test-accuracy'])

            acc /= 2
            x.append(to_inv_acc)
            y.append(acc)

        to_inv_acc -= 0.005
    '''
    print(invalid)
    plt.scatter(x, y, s=[1]*len(x))
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.show()
