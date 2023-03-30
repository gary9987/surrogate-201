import numpy as np


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