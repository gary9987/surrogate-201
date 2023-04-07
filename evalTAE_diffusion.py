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


def inverse_from_acc(model: TransformerAutoencoderDiffusion, num_sample_z: int, to_inv_acc: float):
    num_ops = 7
    num_nodes = 8

    noise = tf.random.normal(shape=[1, 10, 12, 4])
    for i in range(model.diffusion_steps)[::-1]:
        acc = tf.constant(to_inv_acc, shape=[1, 1])
        t = tf.constant(i, shape=[1, 1])
        noise = model.denoise(noise, acc, t)

    latent = tf.reshape(noise, (1, -1, model.d_model))
    rev_x = model.decode(tf.reshape(noise, (1, -1, model.d_model)))  # (num_sample_z, input_size(120))

    ops_vote = tf.reduce_sum(rev_x[:, :num_ops * num_nodes], axis=0).numpy()  # 7 ops 8 nodes
    adj = tf.where(tf.reduce_mean(rev_x[:, num_ops * num_nodes:], axis=0) >= 0.5, x=1., y=0.).numpy()  # (1, 8 * 8)
    adj = np.reshape(adj, (int(adj.shape[-1]**(1/2)), int(adj.shape[-1]**(1/2))))
    ops_idx = []
    for i in range(num_nodes):
        ops_idx.append(np.argmax(ops_vote[i * num_ops: (i + 1) * num_ops], axis=-1))

    return ops_idx, adj, latent


if __name__ == '__main__':
    d_model = 4
    dropout_rate = 0.0
    dff = 512
    num_layers = 3
    num_heads = 3
    diffusion_steps = 500
    input_size = 120

    model = TransformerAutoencoderDiffusion(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                            input_size=input_size, diffusion_steps=diffusion_steps,
                                            dropout_rate=dropout_rate)

    model.load_weights('logs/trainTAE_diffusion/20230407-135542/modelTAE_weights')
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(200), seed=777),
                                              ratio=[0.9, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    for key in datasets:
        datasets[key].apply(OnlyValidAccTransform())

    x_valid, y_valid = to_latent_feature_data(datasets['train'], -1)

    invalid = 0
    x = []
    y = []
    for ly in y_valid:
        #try:
        ops_idx, adj, latent = inverse_from_acc(model, 1, to_inv_acc=ly[-1])
        true = model.encode(tf.constant([[x_valid[0]]]), training=False)
        latent = tf.reshape(latent, (1, -1))
        l = tf.keras.losses.MeanSquaredError()(true, latent)
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
        #except:
        #    invalid += 1
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


