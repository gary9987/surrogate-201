import numpy as np

from datasets.bananas_path_encoding_nb201 import Cell
from datasets.query_nb201 import OPS_by_IDX_201


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


def ops_list_to_nb201_arch_str(ops):
    # partial code from: https://github.com/jovitalukasik/SVGe/blob/main/datasets/NASBench201.py#L239
    steps_coding = ['0', '0', '1', '0', '1', '2']

    node_1 = '|' + ops[1] + '~' + steps_coding[0] + '|'
    node_2 = '|' + ops[2] + '~' + steps_coding[1] + '|' + ops[3] + '~' + steps_coding[2] + '|'
    node_3 = '|' + ops[4] + '~' + steps_coding[3] + '|' + ops[5] + '~' + steps_coding[4] + '|' + ops[
        6] + '~' + steps_coding[5] + '|'
    nodes_nb201 = node_1 + '+' + node_2 + '+' + node_3

    return nodes_nb201


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
        y = np.array([data.y[-1] / 100.0])
        y_list.append(y)

    y_list = np.array(y_list)
    z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim), y_list.shape[0])
    y_list = np.concatenate([z, y_list], axis=-1)
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
