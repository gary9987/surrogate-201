import pickle
import numpy as np
import tensorflow as tf
import os
import random
from nats_bench import create
from spektral.data import PackedBatchLoader
from datasets.nb101_dataset import OP_PRIMITIVES_NB101, NasBench101Dataset
from datasets.nb201_dataset import OP_PRIMITIVES_NB201, NasBench201Dataset
from datasets.transformation import OnlyValidAccTransform, OnlyFinalAcc, LabelScale
from utils.tf_utils import to_undiredted_adj
from datasets.nb201_dataset import OPS_by_IDX_201, ops_list_to_nb201_arch_str
api = create(None, 'tss', fast_mode=True, verbose=False)
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def l2_norm(x, y):
    return np.sum(np.square(x - y))


if __name__ == '__main__':
    '''
    filename = 'logs/50_10_500_top5_finetuneFalse_rfinetuneFalse_rankTrue_ensemble_2NN_4*5*256/ImageNet16-120/20230731-163747/latent_in_each_round.pkl'
    with open(filename, 'rb') as f:
        latent_in_each_round = pickle.load(f)

    loss = tf.keras.metrics.RootMeanSquaredError()
    o2_dif_list = []
    for idx in range(1, len(latent_in_each_round)):
        
        #acc_last = latent_in_each_round[idx-1][0]
        #latent_last = [x for _, x in sorted(zip(acc_last, latent_in_each_round[idx-1][1]))]
        #acc_cur = latent_in_each_round[idx][0]
        #latent_cur = [x for _, x in sorted(zip(acc_cur, latent_in_each_round[idx][1]))]
        
        latent_last = latent_in_each_round[idx-1][1]
        latent_cur = latent_in_each_round[idx][1]

        dif_list = []
        for i in range(len(latent_last)):
            dif_list_tmp = []
            for j in range(len(latent_cur)):
                dif_list_tmp.append(float(loss(latent_last[i], latent_cur[j])))

            dif_list.append(min(dif_list_tmp))

        o2_dif_list.append(dif_list)
        #print(np.mean(dif_list), np.std(dif_list))

    for i in range(1, len(o2_dif_list)):
        dif_list = []
        for j in range(min(len(o2_dif_list[i]), len(o2_dif_list[i-1]))):
            dif_list.append(o2_dif_list[i][j] - o2_dif_list[i - 1][j])
        print(np.mean(dif_list), np.std(dif_list))
    '''
    with open('logs/50_10_500_top5_finetuneFalse_rfinetuneFalse_rankTrue_ensemble_2NN_4*5*256/ImageNet16-120/20230731-163747/latent_in_each_round.pkl', 'rb') as f:
        arch_ours = pickle.load(f)

    with open('/home/gary/CR-LSO/nas_bench_201_experiments/latent_each_iter.pkl', 'rb') as f:
        arch_crlso = pickle.load(f)

    dataset_name = 'cifar10-valid'
    random_seed = 0
    tf.random.set_seed(random_seed)
    random.seed(random_seed)

    if dataset_name == 'nb101':
        num_ops = len(OP_PRIMITIVES_NB101)  # 5
        num_nodes = 7
        num_adjs = num_nodes ** 2
        if os.path.exists('datasets/NasBench101Dataset.cache'):
            dataset = pickle.load(open('datasets/NasBench101Dataset.cache', 'rb'))
        else:
            dataset = NasBench101Dataset(start=0, end=423623)
            with open('datasets/NasBench101Dataset.cache', 'wb') as f:
                pickle.dump(dataset, f)
    else:
        # 15624
        num_ops = len(OP_PRIMITIVES_NB201)  # 7
        num_nodes = 8
        num_adjs = num_nodes ** 2
        label_epochs = 200
        if os.path.exists('datasets/NasBench201Dataset.cache'):
            dataset = pickle.load(open('datasets/NasBench201Dataset.cache', 'rb'))
        else:
            dataset = NasBench201Dataset(start=0, end=15624, dataset=dataset_name, hp=str(label_epochs), seed=False)
            with open('datasets/NasBench201Dataset.cache', 'wb') as f:
                pickle.dump(dataset, f)

    dataset.apply(OnlyValidAccTransform())
    dataset.apply(OnlyFinalAcc())
    if dataset_name != 'nb101':
        dataset.apply(LabelScale(scale=0.01))

    from models.GNN import GraphAutoencoderEnsembleNVP

    num_nvp = 10
    nvp_config = {
        'n_couple_layer': 4,
        'n_hid_layer': 5,
        'n_hid_dim': 256,
        'name': 'NVP',
        'num_couples': 2,
        'inp_dim': 128
    }
    eps_scale = 0.05  # 0.1
    d_model = 32
    dropout_rate = 0.0
    dff = 256
    num_layers = 3
    num_heads = 3
    latent_dim = 16
    model = GraphAutoencoderEnsembleNVP(num_nvp, nvp_config=nvp_config, latent_dim=latent_dim, num_layers=num_layers,
                                        d_model=d_model, num_heads=num_heads,
                                        dff=dff, num_ops=num_ops, num_nodes=num_nodes,
                                        num_adjs=num_adjs, dropout_rate=dropout_rate, eps_scale=eps_scale)
    model((tf.random.normal(shape=(1, num_nodes, num_ops)), tf.random.normal(shape=(1, num_nodes, num_nodes))))
    model.load_weights(
        'logs/50_10_500_top5_finetuneFalse_rfinetuneFalse_rankTrue_ensemble_2NN_4*5*256/ImageNet16-120/20230731-163747/modelGAE_weights_phase2')

    loader = PackedBatchLoader(dataset, batch_size=256, shuffle=False, epochs=1)
    # embedding_list = []
    # acc_list = []
    embedding_list = [0] * len(dataset)
    acc_list = [0] * len(dataset)
    for batch in loader:
        (x, a), y = batch

        ops_idxs = np.argmax(x, axis=-1).tolist()
        # print(ops_idxs)
        index_list = []
        '''
        for ops_idx in ops_idxs:
            new_x = np.zeros((8, 7))
            for i, ops in enumerate(ops_idx):
                new_x[i][ops] = 1

            embedding_list.append(new_x.flatten())

        acc_list.extend(np.squeeze(y).tolist())
        '''
        for ops_idx in ops_idxs:
            ops = [OPS_by_IDX_201[i] for i in ops_idx]
            arch_str = ops_list_to_nb201_arch_str(ops)
            index = api.archstr2index[arch_str]
            index_list.append(index)

        x = tf.constant(x, dtype=tf.float32)
        a = to_undiredted_adj(a)
        embedding = model.encoder((x, a))[0]
        embedding = tf.reshape(embedding, [tf.shape(embedding)[0], -1]).numpy().astype(np.float32).tolist()  # Flatten

        y = np.squeeze(y).tolist()
        for i in range(len(embedding)):
            embedding_list[index_list[i]] = embedding[i]
            acc_list[index_list[i]] = y[i]

    # acc to rank with descending order
    acc_list = np.argsort(np.flip(np.argsort(acc_list)))
    print(len(acc_list), len(embedding_list))

    embedding_idx = []

    for no, (reg, found_arch_list_set) in enumerate(arch_ours):
        for i in found_arch_list_set:
            ops_idx = np.argmax(i, axis=-1)
            ops = [OPS_by_IDX_201[i] for i in ops_idx]
            arch_str = ops_list_to_nb201_arch_str(ops)
            index = api.archstr2index[arch_str]
            if index not in embedding_idx:
                embedding_idx.append(index)
    print(len(embedding_idx))

    for no, (_, found_arch_strs) in enumerate(arch_crlso):
        for arch_str in found_arch_strs:
            index = api.archstr2index[arch_str]
            embedding_idx.append(index)

    print(len(embedding_idx))
    with open('/home/gary/embedding_idx.pkl', 'wb') as f:
        pickle.dump(embedding_idx, f)

    all_embedding = [embedding_list[i] for i in embedding_idx]

    max_dis = 0.
    loss = tf.keras.losses.mean_squared_error
    for i in all_embedding:
        for j in all_embedding:
            x = l2_norm(np.array(i), np.array(j))
            max_dis = max(max_dis, l2_norm(np.array(i), np.array(j)))

    print(max_dis)  # 1.2167107946909481
