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


def inverse_from_acc(model: tf.keras.Model, num_sample_z: int, z_dim: int, to_inv_acc: float):
    num_ops = 7
    num_nodes = 8
    y = np.array([to_inv_acc] * num_sample_z).reshape((num_sample_z, -1))  # (num_sample_z, 1)
    z = np.random.multivariate_normal([1.] * z_dim, np.eye(z_dim), num_sample_z)  # (num_sample_z, z_dim)
    y = np.concatenate([z, y], axis=-1).astype(np.float32)  # (num_sample_z, x_dim)

    rev_latent = model.inverse(y)  # (num_sample_z, x_dim(input_size*d_model ))
    rev_x = model.decode(tf.reshape(rev_latent, (num_sample_z, -1, model.d_model)))  # (num_sample_z, input_size(120))

    ops_vote = tf.reduce_sum(rev_x[:, :num_ops * num_nodes], axis=0).numpy()  # 7 ops 8 nodes
    adj = tf.where(tf.reduce_mean(rev_x[:, num_ops * num_nodes:], axis=0) >= 0.5, x=1., y=0.).numpy()  # (1, 8 * 8)
    adj = np.reshape(adj, (int(adj.shape[-1]**(1/2)), int(adj.shape[-1]**(1/2))))
    ops_idx = []
    for i in range(num_nodes):
        ops_idx.append(np.argmax(ops_vote[i * num_ops: (i + 1) * num_ops], axis=-1))

    return ops_idx, adj


if __name__ == '__main__':
    d_model = 4
    dropout_rate = 0.0
    dff = 512
    num_layers = 3
    num_heads = 3
    input_size = 120
    nvp_config = {
        'n_couple_layer': 3,
        'n_hid_layer': 3,
        'n_hid_dim': 512,
        'name': 'NVP'
    }
    model = TransformerAutoencoderNVP(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                      input_size=input_size, nvp_config=nvp_config)

    model.load_weights('logs/20230403-132134/modelTAE_weights')
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(200), seed=777),
                                              ratio=[0.9, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    for key in datasets:
        datasets[key].apply(OnlyValidAccTransform())

    x_valid, y_valid = to_NVP_data(datasets['train'], 479, -1)

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

    to_inv_acc = 0.95
    invalid = 0
    x = []
    y = []
    for ly in y_valid:
        try:
            ops_idx, adj = inverse_from_acc(model, num_sample_z=50, z_dim=120 * d_model - 1, to_inv_acc=ly[-1])
            '''
            a = model.inverse(ly)
            de = model.decode(tf.reshape(a, (1, -1, model.d_model))).numpy().reshape(-1)
            ops_idx = []
            for i in range(8):
                ops_idx.append(np.argmax(de[i * 7: (i + 1) * 7], axis=-1))
            '''
            ops = [OPS_by_IDX_201[i] for i in ops_idx]
            #print(ops)
            #print(adj)

            arch_str = ops_list_to_nb201_arch_str(ops)
            print(arch_str)

            nb201api = create(None, 'tss', fast_mode=True, verbose=False)
            idx = nb201api.query_index_by_arch(arch_str)

            acc = 0
            acc_l = [ly[-1]]
            for seed in [777, 888]:
                data = nb201api.get_more_info(idx, 'cifar10-valid', iepoch=199, hp='200', is_random=seed)
                #print(data['valid-accuracy'])
                acc += data['valid-accuracy'] / 100.
                acc_l.append(data['valid-accuracy'])

                data = nb201api.get_more_info(idx, 'cifar10', iepoch=199, hp='200', is_random=seed)
                #print(data['test-accuracy'])

            acc /= 2
            x.append(ly[-1])
            y.append(acc)
            print(acc_l)
        except:
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


