import logging
from pathlib import Path

from spektral.data import Dataset, Graph
import pickle
import numpy as np
import os

logger = logging.getLogger(__name__)


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


def transform_nb201_to_graph(records: dict, hp: str, seed: int):
    features_dict = {'INPUT': 0, 'none': 1, 'skip_connect': 2, 'nor_conv_1x1': 3, 'nor_conv_3x3': 4,
                     'avg_pool_3x3': 5, 'OUTPUT': 6}

    num_features = len(features_dict)
    file_path = f'NasBench201Dataset_hp{hp}_seed{seed}'
    Path(file_path).mkdir(exist_ok=True)

    for record, no in zip(records, range(len(records))):

        matrix, ops, metrics = np.array(record[0]), record[1], record[2]
        nodes = matrix.shape[0]

        # Labels Y
        y = np.zeros((3, int(hp)))  # (train_accuracy, validation_accuracy, test_accuracy) * epoch(12)
        metrics_list = ['train-accuracy', 'valid-accuracy']
        if hp == '12':
            metrics_list.append('test-accuracy')
        for i, j in enumerate(metrics_list):
            y[i] = np.array(metrics[j])

        # Node features X
        x = np.zeros((nodes, num_features), dtype=float)  # nodes * (features + metadata + num_layer)
        for i in range(len(ops)):
            x[i][features_dict[ops[i]]] = 1

        # Adjacency matrix A
        adj_matrix = np.array(matrix).astype(float)


        filename = os.path.join(file_path, f'graph_{no}.npz')
        np.savez(filename, a=adj_matrix, x=x, y=y)
        logger.info(f'graph_{no}.npz is saved.')
        print(f'graph_{no}.npz is saved.')


class NasBench201Dataset(Dataset):
    def __init__(self, start: int, end: int, hp: str, seed: int, **kwargs):
        self.file_path = f'NasBench201Dataset_hp{hp}_seed{seed}'
        self.start = start
        self.end = end
        super().__init__(**kwargs)

    def download(self):
        '''
        if not os.path.exists(self.file_path):
            print('Downloading...')
            file_name = wget.download('https://www.dropbox.com/s/40lrvb3lcgij5c8/NasBench101Dataset.zip?dl=1')
            print('Save dataset to {}'.format(file_name))
            os.system('unzip {}'.format(file_name))
            print(f'Unzip dataset finish.')
        '''
        pass

    def read(self):
        output = []
        filename_list = []

        for i in range(len(os.listdir(self.file_path))):
            #with np.load(os.path.join(path, f'graph_{i}.npz')) as npz:
            #    data = {'x': npz['x'], 'e': npz['e'], 'a': npz['a'], 'y': npz['y']}
            filename_list.append(os.path.join(self.file_path, f'graph_{i}.npz'))

        assert len(filename_list) > self.end
        assert self.start >= 0

        for i in range(self.start, self.end + 1):
            data = np.load(filename_list[i])
            output.append(Graph(x=data['x'], a=data['a'], y=data['y']))

        return output


if __name__ == '__main__':
    hp = '12' # hp = 12 or 200

    if hp == '12':
        seed_list = [111, 777]
    elif hp == '200':
        seed_list = [777, 888] # 999

    for seed in seed_list:
        output_dir = 'nb201_query_data'
        filename = f'hp{hp}_seed{seed}.pkl'
        with open(os.path.join(output_dir, filename), 'rb') as f:
            records = pickle.load(f)

        print(len(records))  # 15625
        transform_nb201_to_graph(records, hp=hp, seed=seed)
        datasets = NasBench201Dataset(0, 15355, hp=hp, seed=seed)
        print(datasets)
