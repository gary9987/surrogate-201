import copy
import random
import numpy as np
from spektral.data import Graph
from datasets.bananas_path_encoding_nb201 import Cell
from datasets.nb201_dataset import OPS_by_IDX_201


def train_valid_test_split_dataset(data, ratio=[0.8, 0.1, 0.1], shuffle=False, shuffle_seed=0):
    assert sum(ratio) <= 1.
    if shuffle:
        np.random.seed(shuffle_seed)
        idxs = np.random.permutation(len(data))
    else:
        idxs = np.array([i for i in range(len(data))])
    ret = {}

    if len(ratio) == 2:
        split_va = int(ratio[0] * len(data))
        idx_tr, idx_va = np.split(idxs, [split_va])
        ret['train'] = data[idx_tr]
        ret['valid'] = data[idx_va]
    elif len(ratio) == 3:
        split_va, split_te = int(ratio[0] * len(data)), int((ratio[0] + ratio[1]) * len(data))
        idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
        ret['train'] = data[idx_tr]
        ret['valid'] = data[idx_va]
        ret['test'] = data[idx_te]
    else:
        raise ValueError('len(ratio) should be 2 or 3')

    return ret


def to_NVP_data(graph_dataset, z_dim, reg_size, repeat=1):
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
        y = np.array([data.y[-1] / 100.0])
        y_list.append(y)

    y_list = np.array(y_list).astype(np.float32).repeat(repeat, axis=0)
    features = np.array(features).astype(np.float32).repeat(repeat, axis=0)
    z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim), y_list.shape[0])
    y_list = np.concatenate([z, y_list], axis=-1)

    to_nan_idx = to_nan_idx * repeat
    to_nan_idx_repeat = to_nan_idx
    for i in to_nan_idx:
        to_nan_idx_repeat = np.concatenate([to_nan_idx_repeat, i + np.arange(1, repeat)])
    y_list[to_nan_idx, :] = np.nan

    return np.array(features).astype(np.float32), np.array(y_list).astype(np.float32)


def to_NVP_data_path_encode(graph_dataset, z_dim, reg_size):
    features = []
    y_list = []
    if reg_size == -1:
        nan_size = 0
    else:
        nan_size = len(graph_dataset) - reg_size

    to_nan_idx = np.random.choice(range(len(graph_dataset)), nan_size, replace=False)

    ops_idx = []
    for data in graph_dataset:
        x = np.reshape(data.x, -1)
        for i in range(8):
            ops_idx.append(OPS_by_IDX_201[np.argmax(x[i * 7: (i + 1) * 7], axis=-1)])

        arch_str = ops_list_to_nb201_arch_str(ops_idx)
        encode_x = Cell(arch_str).encode_paths()
        features.append(encode_x)

        y = np.array([data.y[-1] / 100.0])
        y_list.append(y)

    y_list = np.array(y_list)
    z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim), y_list.shape[0])
    y_list = np.concatenate([z, y_list], axis=-1)
    y_list[to_nan_idx, :] = np.nan

    return np.array(features).astype(np.float32), np.array(y_list).astype(np.float32)


def to_latent_feature_data(graph_dataset, reg_size):
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

        y = np.array([data.y[-1] / 100.0])
        y_list.append(y)

    y_list = np.array(y_list)
    y_list[to_nan_idx, :] = np.nan

    return np.array(features).astype(np.float32), np.array(y_list).astype(np.float32)


def mask_graph_dataset(graph_dataset, reg_size: int, non_nan_repeat: int, random_seed=0):
    new_graph_dataset = copy.deepcopy(graph_dataset)
    if reg_size == -1:
        nan_size = 0
    else:
        nan_size = len(new_graph_dataset) - reg_size

    np.random.seed(random_seed)
    to_nan_idx = np.random.choice(range(len(new_graph_dataset)), nan_size, replace=False)

    for i in range(len(new_graph_dataset)):
        if i in to_nan_idx:
            new_graph_dataset[i].y = np.array([np.nan])
        else:
            for _ in range(non_nan_repeat - 1):
                new_graph_dataset += new_graph_dataset[i:i+1]

    return new_graph_dataset


def repeat_graph_dataset_element(graph_dataset, num_repeat: int):
    new_graph_dataset = copy.deepcopy(graph_dataset)
    for _ in range(num_repeat - 1):
        new_graph_dataset += graph_dataset
    return new_graph_dataset


def graph_to_str(graph):
    if isinstance(graph, dict):
        return str(graph['x'].astype(np.float32).tolist()) + str(graph['a'].astype(np.float32).tolist())
    elif isinstance(graph, Graph):
        return str(graph.x.astype(np.float32).tolist()) + str(graph.a.astype(np.float32).tolist())
    else:
        raise ValueError('graph type error')


def arch_list_to_set(arch_list):
    arch_list_set = []
    visited = []
    for i in arch_list:
        if graph_to_str(i) not in visited:
            arch_list_set.append(i)
            visited.append(graph_to_str(i))
    return arch_list_set


def get_rank_weight(y_true, num_repeat):
    outputs_argsort = np.argsort(-np.asarray(y_true))
    ranks = np.argsort(outputs_argsort)
    return 1 / (10e-3 * len(y_true) * num_repeat + (ranks * num_repeat))


def weighted_graph_dataset_element(graph_dataset, num_repeat: int):
    new_graph_dataset = copy.deepcopy(graph_dataset)
    y_true = [data.y[-1] for data in new_graph_dataset]
    weights = get_rank_weight(y_true, num_repeat)
    for idx, graph in enumerate(new_graph_dataset):
        new_graph_dataset[idx].y = np.array([weights[idx], new_graph_dataset[idx].y[-1]])
    new_graph_dataset.graphs = random.choices(new_graph_dataset.graphs, weights=weights, k=len(new_graph_dataset) * num_repeat)
    return new_graph_dataset
