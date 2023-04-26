from typing import Union
import numpy as np
import tensorflow as tf
from datasets.nb201_dataset import NasBench201Dataset
from datasets.query_nb201 import OPS_by_IDX_201
from datasets.transformation import OnlyValidAccTransform, ReshapeYTransform, OnlyFinalAcc, LabelScale
from datasets.utils import train_valid_test_split_dataset, ops_list_to_nb201_arch_str
from models.GNN import GraphAutoencoderNVP
from nats_bench import create
import matplotlib.pyplot as plt
from spektral.data import BatchLoader
import matplotlib as mpl
from utils.tf_utils import to_undiredted_adj

mpl.rcParams['figure.dpi'] = 300

random_seed = 0
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

nb201api = create(None, 'tss', fast_mode=True, verbose=False)


def inverse_from_acc(model: tf.keras.Model, num_sample_z: int, x_dim: int, z_dim: int, to_inv_acc, version=2):
    batch_size = int(tf.shape(to_inv_acc)[0])
    y = tf.repeat(to_inv_acc, num_sample_z, axis=0)  # (batch_size * num_sample_z, 1)
    z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim),
                                      size=batch_size * num_sample_z)  # (num_sample_z, z_dim)
    y = np.concatenate([z, y], axis=-1).astype(np.float32)  # (num_sample_z, z_dim + 1)

    rev_latent = model.inverse(y)  # (num_sample_z, latent_dim)
    if version == 1:
        rev_latent = rev_latent[:, :x_dim]
    elif version == 2:
        rev_latent = tf.reshape(rev_latent, (batch_size, 8, -1))  # (batch_size, num_sample_z, latent_dim)
    else:
        raise ValueError('version')

    _, adj, ops_cls, adj_cls = model.decode(rev_latent)
    ops_cls = tf.reshape(ops_cls, (batch_size, num_sample_z, -1, model.num_ops))  # (batch_size, num_sample_z, 8, 7)
    ops_vote = tf.reduce_sum(ops_cls, axis=1).numpy()  # (batch_size, 1, 8 * 7)

    adj = tf.reshape(adj, (batch_size, num_sample_z, 8, 8))  # (batch_size, num_sample_z, 8 * 8)
    adj = tf.where(tf.reduce_mean(adj, axis=1) >= 0.5, x=1., y=0.).numpy()  # (batch_size, 8 * 8)
    #adj = np.reshape(adj, (batch_size, int(adj.shape[-1] ** (1 / 2)), int(adj.shape[-1] ** (1 / 2))))

    ops_idx_list = []
    adj_list = []
    for i, j in zip(ops_vote, adj):
        ops_idx_list.append(np.argmax(i, axis=-1).tolist())
        adj_list.append(j)

    return ops_idx_list, adj_list


def query_acc_by_ops(ops: Union[list, np.ndarray], dataset_name, is_random=False, on='valid-accuracy') -> float:
    """
    :param ops: ops_idx or ops_cls
    :param is_random: False will return the avg. of result
    :param on: valid-accuracy or test-accuracy
    :return: acc
    """
    if isinstance(ops, np.ndarray):
        ops_idx = np.argmax(ops, axis=-1)
    else:
        ops_idx = ops

    ops = [OPS_by_IDX_201[i] for i in ops_idx]
    arch_str = ops_list_to_nb201_arch_str(ops)
    idx = nb201api.query_index_by_arch(arch_str)
    meta_info = nb201api.query_meta_info_by_index(idx, hp='200')

    if on == 'valid-accuracy':
        data = meta_info.get_metrics(dataset_name, 'x-valid', iepoch=None, is_random=is_random)
        acc = data['accuracy'] / 100
    elif on == 'test-accuracy':
        if dataset_name == 'cifar10-valid':
            data = meta_info.get_metrics('cifar10', 'ori-test', iepoch=None, is_random=is_random)
        else:
            data = meta_info.get_metrics(dataset_name, 'x-test', iepoch=None, is_random=is_random)
        acc = data['accuracy'] / 100
    else:
        raise ValueError('on should be valid-accuracy or test-accuracy')
    return acc


def eval_query_best(model: tf.keras.Model, dataset_name, x_dim: int, z_dim: int, query_amount=10, noise_scale=0.0, version=2):
    # Eval query 1.0
    x = []
    y = []
    found_arch_list = []
    invalid = 0
    to_inv_acc = 1.0
    to_inv = tf.repeat(tf.reshape(tf.constant(to_inv_acc), [-1, 1]), query_amount, axis=0)
    to_inv += noise_scale * tf.random.normal(tf.shape(to_inv))
    ops_idx_lis, adj_list = inverse_from_acc(model, num_sample_z=1, x_dim=x_dim, z_dim=z_dim, to_inv_acc=to_inv, version=version)
    for ops_idx, adj, query_acc in zip(ops_idx_lis, adj_list, to_inv[:, -1]):
        try:
            acc = query_acc_by_ops(ops_idx, dataset_name, is_random=False)
            x.append(query_acc)
            y.append(acc)
            found_arch_list.append({'x': np.eye(len(OPS_by_IDX_201))[ops_idx], 'a': adj, 'y': np.array([acc])})
        except:
            print('invalid')
            invalid += 1

    fig, ax = plt.subplots()
    ax.axline((0, 0), slope=1, linewidth=0.2, color='black')
    plt.scatter(x, y, s=[1] * len(x))
    plt.xlim(0.85, 1.2)
    plt.ylim(0.85, 1.2)
    plt.savefig('top.png')
    if len(y) == 0:
        return invalid, 0, 0, found_arch_list

    return invalid, sum(y) / len(y), max(y), found_arch_list


if __name__ == '__main__':
    dataset = 'cifar10-valid'
    num_ops = 7
    num_nodes = 8
    num_adjs = 64

    d_model = 32
    dropout_rate = 0.0
    dff = 256
    num_layers = 3
    num_heads = 3

    latent_dim = 16
    x_dim = latent_dim * num_nodes
    y_dim = 1  # 1
    z_dim = x_dim - 1  # 27
    # z_dim = latent_dim * 4 - 1
    tot_dim = y_dim + z_dim  # 28
    pad_dim = tot_dim - x_dim  # 14

    plot_on_slit = 'train'  # train, valid, test

    nvp_config = {
        'n_couple_layer': 4,
        'n_hid_layer': 4,
        'n_hid_dim': 256,
        'name': 'NVP',
        'inp_dim': x_dim
    }

    model = GraphAutoencoderNVP(nvp_config=nvp_config, latent_dim=latent_dim, num_layers=num_layers,
                                d_model=d_model, num_heads=num_heads,dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=0.)
    model((tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))))
    # model.load_weights('logs/phase2_model/modelTAE_weights_phase2')
    model.load_weights('logs/20230425-162536/modelGAE_weights_phase2')
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, dataset=dataset, hp=str(200), seed=False),
                                              ratio=[0.8, 0.1, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    for key in datasets:
        datasets[key].apply(OnlyValidAccTransform())
        datasets[key].apply(OnlyFinalAcc())
        datasets[key].apply(LabelScale(scale=0.01))

    # Eval inverse
    datasets['train'] = datasets['train'][:350]
    datasets['valid'] = datasets['valid'][:50]
    x = []
    y = []
    invalid = 0
    loader = BatchLoader(datasets[plot_on_slit], batch_size=512, epochs=1)
    for _, label_acc in loader:
        ops_idx_lis, adj_list = inverse_from_acc(model, num_sample_z=1, x_dim=x_dim, z_dim=z_dim, to_inv_acc=label_acc[:, -1:])

        for ops_idx, adj, query_acc in zip(ops_idx_lis, adj_list, label_acc[:, -1]):
            try:
                ops_str_list = [OPS_by_IDX_201[i] for i in ops_idx]
                #print(adj)
                arch_str = ops_list_to_nb201_arch_str(ops_str_list)
                #print(arch_str)

                arch_idx = nb201api.query_index_by_arch(arch_str)

                acc_list = [float(query_acc)]
                data = nb201api.query_meta_info_by_index(arch_idx, hp='200').get_metrics(dataset, 'x-valid', iepoch=None, is_random=False)
                acc = data['accuracy'] / 100.

                x.append(float(query_acc))
                y.append(acc)
                print(acc_list)  # [query_acc, 777_acc, 888_acc]
            except:
                print('invalid')
                invalid += 1

    print('Number of invalid decode', invalid)
    fig, ax = plt.subplots()
    ax.axline((0, 0), slope=1, linewidth=0.2, color='black')
    plt.scatter(x, y, s=[1]*len(x))
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.savefig('inverse.png')
    plt.cla()

    # Eval regression
    loader = BatchLoader(datasets[plot_on_slit], batch_size=256, epochs=1)
    x = []
    y = []
    for arch, label_acc in loader:
        arch = (arch[0], to_undiredted_adj(arch[1]))
        ops_cls, adj_cls, kl_loss, reg, flat_encoding = model(arch)

        for true_acc, query_acc in zip(label_acc[:, -1], reg[:, -1]):
            x.append(float(query_acc))
            y.append(float(true_acc))

    print('Number of invalid decode', invalid)
    fig, ax = plt.subplots()
    ax.axline((0, 0), slope=1, linewidth=0.2, color='black')
    plt.scatter(x, y, s=[1] * len(x))
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.savefig('regresion.png')
    plt.cla()

    # Eval decending
    x = []
    y = []
    invalid = 0
    to_inv_acc = 0.00
    to_inv = []
    to_inv_repeat = 1
    while to_inv_acc <= 1.0:
        to_inv.append(to_inv_acc)
        to_inv_acc += 0.005

    to_inv = tf.repeat(tf.reshape(tf.constant(to_inv), [-1, 1]), to_inv_repeat, axis=0)
    ops_idx_lis, adj_list = inverse_from_acc(model, num_sample_z=1, x_dim=x_dim, z_dim=z_dim, to_inv_acc=to_inv)
    for ops_idx, adj, query_acc in zip(ops_idx_lis, adj_list, to_inv[:, -1]):
        try:
            ops = [OPS_by_IDX_201[i] for i in ops_idx]
            arch_str = ops_list_to_nb201_arch_str(ops)
            print(arch_str)

            idx = nb201api.query_index_by_arch(arch_str)

            data = nb201api.query_meta_info_by_index(idx, hp='200').get_metrics(dataset, 'x-valid', iepoch=None,
                                                                                     is_random=False)
            acc = data['accuracy'] / 100.
            print(data['accuracy'])
            x.append(query_acc)
            y.append(acc)
        except:
            print('invalid')
            invalid += 1

    print('Number of invalid decode', invalid)
    fig, ax = plt.subplots()
    ax.axline((0, 0), slope=1, linewidth=0.2, color='black')
    plt.scatter(x, y, s=[1] * len(x))
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.savefig('decending.png')
    plt.cla()

    invalid, avg_acc, best_acc, _ = eval_query_best(model, dataset, x_dim, z_dim)
    print('Number of invalid decode', invalid)
    print('Avg found acc', avg_acc)
    print('Best found acc', best_acc)