import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datasets.nb201_dataset import NasBench201Dataset, OPS_by_IDX_201
from datasets.transformation import OnlyValidAccTransform, ReshapeYTransform, OnlyFinalAcc, LabelScale
from datasets.utils import train_valid_test_split_dataset, ops_list_to_nb201_arch_str
from models.GNN import GraphAutoencoderNVP
from nats_bench import create
from spektral.data import BatchLoader
from utils.tf_utils import to_undiredted_adj

random_seed = 0
np.random.seed(random_seed)
tf.random.set_seed(random_seed)


def prior_accuracy(model, loader):
    accuracy = 0
    encode_times = 10  # sample embedding 10 times for each Graph
    decode_times = 10  # decode each embedding 10 times

    for x, _ in tqdm(loader):
        undiredted_x = (x[0], to_undiredted_adj(x[1]))
        mean, log_var = model.encode(undiredted_x)
        for _ in range(encode_times):
            z = model.sample(mean, log_var, eps_scale=0.05)
            for _ in range(decode_times):
                ops_batch, adj_batch, _, _ = model.decode(z)
                for ops, adj, l_ops, l_adj in zip(ops_batch, adj_batch, tf.argmax(x[0], -1), x[1]):
                    adj = tf.reshape(adj, (int(adj.shape[-1] ** (1 / 2)), int(adj.shape[-1] ** (1 / 2))))
                    if tf.reduce_all(tf.equal(ops, l_ops)) and tf.reduce_all(tf.equal(adj, l_adj)):
                        accuracy += 1

    return accuracy / (len(loader.dataset) * encode_times * decode_times) * 100


def prior_validity(model, loader, n_latent_point, nb201api):
    from datasets.query_nb201 import ADJACENCY
    result = {}
    means = []
    for x, _ in tqdm(loader['train']):
        undiredted_x = (x[0], to_undiredted_adj(x[1]))
        mean, _ = model.encode(undiredted_x)
        means.extend(mean)

    means = tf.stack(means, axis=0)
    z_mean, z_std = tf.reduce_mean(means, axis=0), tf.math.reduce_std(means, axis=0)

    n_valid = 0
    amount = 0
    decode_times = 10
    g_valid = []
    for _ in tqdm(range(n_latent_point)):
        z = tf.random.normal((1, 8, model.latent_dim), dtype=tf.float32)
        z = z * z_std + z_mean  # move to train's latent range
        z = tf.repeat(z, decode_times, axis=0)
        #for _ in range(decode_times):
        ops_batch, adj_batch, ops_cls, adj_cls = model.decode(z)
        for ops, adj in zip(ops_batch, adj_batch):
            ops_str_list = [OPS_by_IDX_201[i] for i in ops.numpy()]
            adj = tf.reshape(adj, (int(adj.shape[-1] ** (1 / 2)), int(adj.shape[-1] ** (1 / 2))))
            arch_str = ops_list_to_nb201_arch_str(ops_str_list)
            try:
                idx = nb201api.query_index_by_arch(arch_str)
                if not tf.reduce_all(tf.equal(adj, ADJACENCY)):
                    continue
                g_valid.append(str(ops.numpy().tolist() + adj.numpy().tolist()))
                n_valid += 1
            except:
                continue
            amount += 1

    result['valid'] = n_valid / amount * 100
    result['unique'] = len(set(g_valid)) / len(g_valid) * 100

    arch_in_test_set = []
    for x, _ in loader['test']:
        for ops, adj in zip(x[0], x[1]):
            ops = np.argmax(ops, -1)
            arch_in_test_set.append(str(ops.tolist() + adj.tolist()))
    novelty = 0
    for i in g_valid:
        if i in arch_in_test_set:
            novelty += 1
    result['novelty'] = novelty / len(g_valid) * 100
    return result


if __name__ == '__main__':
    num_ops = 7
    num_nodes = 8
    num_adjs = 64

    d_model = 32
    dropout_rate = 0.0
    dff = 256
    num_layers = 3
    num_heads = 3

    latent_dim = 16
    x_dim = latent_dim * num_nodes  # 14
    z_dim = x_dim - 1  # 27
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
    model.load_weights('logs/20230418-152148/modelGAE_weights_phase1')
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(200), seed=777),
                                              ratio=[0.8, 0.1, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    for key in datasets:
        datasets[key].apply(OnlyValidAccTransform())
        datasets[key].apply(OnlyFinalAcc())
        datasets[key].apply(LabelScale(scale=0.01))

    nb201api = create(None, 'tss', fast_mode=True, verbose=False)

    # Eval prior
    accuracy = prior_accuracy(model, BatchLoader(datasets['test'], batch_size=2048, epochs=1))
    print(accuracy)
    loader = {
        'train': BatchLoader(datasets['train'], batch_size=2048, epochs=1),
        'test': BatchLoader(datasets['test'], batch_size=2048, epochs=1)
    }
    result = prior_validity(model, loader, 1000, nb201api)
    print(result)
