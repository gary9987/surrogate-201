import numpy as np
import tensorflow as tf
from datasets.nb201_dataset import NasBench201Dataset
from datasets.query_nb201 import OPS_by_IDX_201
from datasets.transformation import OnlyValidAccTransform, ReshapeYTransform, OnlyFinalAcc, LabelScale_NasBench101
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


def inverse_from_acc(model: tf.keras.Model, num_sample_z: int, x_dim: int, z_dim: int, to_inv_acc):
    batch_size = int(tf.shape(to_inv_acc)[0])
    y = tf.repeat(to_inv_acc, num_sample_z, axis=0)  # (batch_size * num_sample_z, 1)

    z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim),
                                      size=batch_size * num_sample_z)  # (num_sample_z, z_dim)
    y = np.concatenate([z, y], axis=-1).astype(np.float32)  # (num_sample_z, z_dim + 1)

    rev_latent = model.inverse(y)  # (num_sample_z, latent_dim)
    rev_latent = rev_latent[:, :x_dim]
    _, adj, ops_cls, adj_cls = model.decode(rev_latent)
    ops_cls = tf.reshape(ops_cls, (batch_size, num_sample_z, -1, model.num_ops))  # (batch_size, num_sample_z, 8, 7)
    ops_vote = tf.reduce_sum(ops_cls, axis=1).numpy()  # (batch_size, 1, 8 * 7)

    adj = tf.reshape(adj, (batch_size, num_sample_z, model.num_adjs))  # (batch_size, num_sample_z, 8 * 8)
    adj = tf.where(tf.reduce_mean(adj, axis=1) >= 0.5, x=1., y=0.).numpy()  # (batch_size, 8 * 8)
    adj = np.reshape(adj, (batch_size, int(adj.shape[-1] ** (1 / 2)), int(adj.shape[-1] ** (1 / 2))))

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

    d_model = 32
    dropout_rate = 0.0
    dff = 256
    num_layers = 3
    num_heads = 3

    latent_dim = 14
    x_dim = latent_dim  # 14
    z_dim = latent_dim * 2 - 1  # 27
    y_dim = 1
    tot_dim = y_dim + z_dim  # 28


    nvp_config = {
        'n_couple_layer': 4,
        'n_hid_layer': 4,
        'n_hid_dim': 256,
        'name': 'NVP',
        'inp_dim': tot_dim
    }

    model = GraphAutoencoderNVP(nvp_config=nvp_config, latent_dim=latent_dim, num_layers=num_layers,
                                d_model=d_model, num_heads=num_heads,dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=0.)

    # model.load_weights('logs/phase2_model/modelTAE_weights_phase2')
    model.load_weights('logs/20230414-011142/modelGAE_weights_phase2')
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(200), seed=777),
                                              ratio=[0.8, 0.1, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    for key in datasets:
        datasets[key].apply(OnlyValidAccTransform())
        datasets[key].apply(OnlyFinalAcc())
        datasets[key].apply(LabelScale_NasBench101(scale=0.01))

    nb201api = create(None, 'tss', fast_mode=True, verbose=False)

    # Eval inverse
    datasets['train'] = datasets['train'][:1000]
    datasets['valid'] = datasets['valid'][:100]
    x = []
    y = []
    invalid = 0
    loader = BatchLoader(datasets['train'], batch_size=512, epochs=1)
    for _, label_acc in loader:
        ops_idx_lis, adj_list = inverse_from_acc(model, num_sample_z=15, x_dim=x_dim, z_dim=z_dim, to_inv_acc=label_acc[:, -1:])

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

    print('Number of invalid decode', invalid)
    fig, ax = plt.subplots()
    ax.axline((0, 0), slope=1, linewidth=0.2, color='black')
    plt.scatter(x, y, s=[1]*len(x))
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.savefig('inverse.png')
    plt.cla()

    # Eval regression
    loader = BatchLoader(datasets['train'], batch_size=256, epochs=1)
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
            print(ops)
            print(adj)

            arch_str = ops_list_to_nb201_arch_str(ops)
            print(arch_str)

            idx = nb201api.query_index_by_arch(arch_str)

            acc = 0
            for seed in [777, 888]:
                data = nb201api.get_more_info(idx, 'cifar10-valid', iepoch=199, hp='200', is_random=seed)
                print(data['valid-accuracy'])
                acc += data['valid-accuracy'] / 100.

                #data = nb201api.get_more_info(idx, 'cifar10', iepoch=199, hp='200', is_random=seed)
                #print(data['test-accuracy'])
            acc /= 2
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
