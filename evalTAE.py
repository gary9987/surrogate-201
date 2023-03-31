import numpy as np
import tensorflow as tf

from datasets.nb201_dataset import NasBench201Dataset
from datasets.query_nb201 import OPS_by_IDX_201
from datasets.transformation import OnlyValidAccTransform, ReshapeYTransform
from datasets.utils import train_valid_test_split_dataset
from models.TransformerAE import TransformerAutoencoderNVP
from nats_bench import create
import matplotlib.pyplot as plt


random_seed = 0
np.random.seed(random_seed)
tf.random.set_seed(random_seed)


def to_NVP_data(graph_dataset, z_dim, reg_size):
    features = []
    y_list = []
    if reg_size == -1:
        nan_size = 0
    else:
        nan_size = len(graph_dataset) - reg_size

    to_nan_idx = np.random.choice(range(len(graph_dataset)), nan_size, replace=False)

    for data in graph_dataset:
        x = np.reshape(data.x, -1)
        a = np.reshape(data.a, -1)
        features.append(np.concatenate([x, a]))

        #z = np.reshape(np.random.multivariate_normal([0.]*z_dim, np.eye(z_dim), 1), -1)
        y = np.array([data.y[-1]])
        y_list.append(y)

    y_list = np.array(y_list)
    z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim), y_list.shape[0])
    y_list = np.concatenate([z, y_list], axis=-1)
    y_list[to_nan_idx, :] = np.nan

    return np.array(features).astype(np.float32), np.array(y_list).astype(np.float32)


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

def ops_list_to_nb201_arch_str(ops):
    # partial code from: https://github.com/jovitalukasik/SVGe/blob/main/datasets/NASBench201.py#L239
    steps_coding = ['0', '0', '1', '0', '1', '2']

    node_1 = '|' + ops[1] + '~' + steps_coding[0] + '|'
    node_2 = '|' + ops[2] + '~' + steps_coding[1] + '|' + ops[3] + '~' + steps_coding[2] + '|'
    node_3 = '|' + ops[4] + '~' + steps_coding[3] + '|' + ops[5] + '~' + steps_coding[4] + '|' + ops[
        6] + '~' + steps_coding[5] + '|'
    nodes_nb201 = node_1 + '+' + node_2 + '+' + node_3

    return nodes_nb201


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
        'n_hid_dim': 128,
        'name': 'NVP'
    }
    model = TransformerAutoencoderNVP(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                      input_size=input_size, nvp_config=nvp_config)

    model.load_weights('modelTAE_weights')
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(200), seed=777),
                                              ratio=[0.9, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    for key in datasets:
        datasets[key].apply(OnlyValidAccTransform())

    x_valid, y_valid = to_NVP_data(datasets['valid'], 479, -1)


    to_inv_acc = 0.95
    x = []
    y = []
    for ly in y_valid:
        ops_idx, adj = inverse_from_acc(model, num_sample_z=10000, z_dim=120 * d_model - 1, to_inv_acc=ly[-1])
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
            acc += data['valid-accuracy']
            acc_l.append(data['valid-accuracy'])

            data = nb201api.get_more_info(idx, 'cifar10', iepoch=199, hp='200', is_random=seed)
            #print(data['test-accuracy'])

        acc /= 2
        x.append(ly[-1])
        y.append(acc)
        print(acc_l)
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
    plt.scatter(x, y, s=[1]*len(x))
    plt.xlim(0., 100.)
    plt.ylim(0., 100.)
    plt.show()

