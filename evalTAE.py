import numpy as np
import tensorflow as tf

from datasets.nb201_dataset import NasBench201Dataset
from datasets.query_nb201 import OPS_by_IDX_201
from datasets.transformation import OnlyValidAccTransform, ReshapeYTransform
from datasets.utils import train_valid_test_split_dataset, ops_list_to_nb201_arch_str, to_NVP_data
from models.TransformerAE import TransformerAutoencoderNVP
from nats_bench import create
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

random_seed = 0
np.random.seed(random_seed)
tf.random.set_seed(random_seed)


def inverse_from_acc(model: tf.keras.Model, num_sample_z: int, z_dim: int, to_inv_acc):
    batch_size = int(tf.shape(to_inv_acc)[0])
    y = tf.repeat(to_inv_acc, num_sample_z, axis=0) # (batch_size * num_sample_z, 1)

    z = np.random.multivariate_normal([1.] * z_dim, np.eye(z_dim), size=batch_size * num_sample_z)  # (num_sample_z, z_dim)
    y = np.concatenate([z, y], axis=-1).astype(np.float32)  # (num_sample_z, z_dim + 1)

    rev_latent = model.inverse(y)  # (num_sample_z, x_dim(input_size*d_model ))
    _, adj, ops_cls, adj_cls = model.decode(tf.reshape(rev_latent, (tf.shape(rev_latent)[0], -1, model.d_model)))  # (num_sample_z, input_size(120))
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
    num_ops = 7
    num_nodes = 8
    num_adjs = 64
    d_model = 4
    dropout_rate = 0.0
    dff = 512
    num_layers = 3
    num_heads = 3
    input_size = 120
    nvp_config = {
        'n_couple_layer': 4,
        'n_hid_layer': 4,
        'n_hid_dim': 256,
        'name': 'NVP'
    }
    model = TransformerAutoencoderNVP(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                      input_size=input_size, nvp_config=nvp_config,
                                      num_ops=num_ops, num_nodes=num_nodes, num_adjs=num_adjs)

    model.load_weights('logs/20230408-003521/modelTAE_weights_phase2')
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(200), seed=777),
                                              ratio=[0.9, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    for key in datasets:
        datasets[key].apply(OnlyValidAccTransform())

    x_valid, y_valid = to_NVP_data(datasets['valid'], 479, -1)
    loader = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size=256)
    '''
    diff = 0
    for x, y in zip(x_valid, y_valid):
        ops_idx = []
        for i in range(8):
            ops_idx.append(np.argmax(x[i * 7: (i + 1) * 7], axis=-1))
        print(ops_idx)

        z = tf.random.normal(shape=[1, 479])
        y = tf.concat([z, tf.constant([[y[-1]]])], axis=-1)
        a = model.inverse(y)
        de = model.decode(tf.reshape(a, (1, -1, model.d_model))).numpy().reshape(-1)
        ops_idx = []
        for i in range(8):
            ops_idx.append(np.argmax(de[i * 7: (i + 1) * 7], axis=-1))
        print(ops_idx)

    print(diff)
    '''

    invalid = 0
    nb201api = create(None, 'tss', fast_mode=True, verbose=False)
    x = []
    y = []
    for _, label_acc in loader:
        ops_idx_lis, adj_list = inverse_from_acc(model, num_sample_z=50, z_dim=120 * d_model - 1, to_inv_acc=label_acc[:, -1:])

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
    to_inv_acc = 0.95
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
    print('Number of invalid decode', invalid)
    plt.scatter(x, y, s=[1]*len(x))
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.show()


