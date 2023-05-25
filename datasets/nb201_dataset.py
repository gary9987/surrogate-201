from pathlib import Path
import spektral.data
import wget
from nats_bench import create
from spektral.data import Dataset, Graph
import pickle
import numpy as np
import os
from tqdm import tqdm


# Useful constants
OP_PRIMITIVES_NB201 = [
    'output',
    'input',
    'nor_conv_1x1',
    'nor_conv_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'none',
]

OPS_by_IDX_201 = {OP_PRIMITIVES_NB201.index(i):i for i in OP_PRIMITIVES_NB201}
OPS_201 = {i:OP_PRIMITIVES_NB201.index(i) for i in OP_PRIMITIVES_NB201}

ADJACENCY = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1 ,0 ,0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0]])


def ops_list_to_nb201_arch_str(ops):
    # partial code from: https://github.com/jovitalukasik/SVGe/blob/main/datasets/NASBench201.py#L239
    steps_coding = ['0', '0', '1', '0', '1', '2']

    node_1 = '|' + ops[1] + '~' + steps_coding[0] + '|'
    node_2 = '|' + ops[2] + '~' + steps_coding[1] + '|' + ops[3] + '~' + steps_coding[2] + '|'
    node_3 = '|' + ops[4] + '~' + steps_coding[3] + '|' + ops[5] + '~' + steps_coding[4] + '|' + ops[
        6] + '~' + steps_coding[5] + '|'
    nodes_nb201 = node_1 + '+' + node_2 + '+' + node_3

    return nodes_nb201


def convert_matrix_ops_to_graph(matrix, ops):
    features_dict = {'output': 0, 'input': 1, 'nor_conv_1x1': 2, 'nor_conv_3x3': 3, 'avg_pool_3x3': 4,
                     'skip_connect': 5, 'none': 6}

    num_features = len(features_dict)
    num_nodes = matrix.shape[0]

    # Node features X
    x = np.zeros((num_nodes, num_features), dtype=float)  # num_nodes * (features + metadata + num_layer)
    for i in range(len(ops)):
        x[i][features_dict[ops[i]]] = 1

    # Adjacency matrix A
    a = np.array(matrix).astype(float)

    return spektral.data.Graph(x=x, a=a)


def idx_to_metric(idx, dataset, nb201api):
    meta_info = nb201api.query_meta_info_by_index(idx, hp='200')
    data = meta_info.get_metrics(dataset, 'train', iepoch=None, is_random=False)
    train_acc = data['accuracy'] / 100
    data = meta_info.get_metrics(dataset, 'x-valid', iepoch=None, is_random=False)
    val_acc = data['accuracy'] / 100
    if dataset == 'cifar10-valid':
        data = meta_info.get_metrics('cifar10', 'ori-test', iepoch=None, is_random=False)
    else:
        data = meta_info.get_metrics(dataset, 'x-test', iepoch=None, is_random=False)
    test_acc = data['accuracy'] / 100
    return  np.array([[train_acc], [val_acc], [test_acc]], dtype=np.float32)


def transform_nb201_to_graph(records: dict, hp: str, seed: int, dataset: str):
    file_path = f'../NasBench201Dataset/{dataset}/NasBench201Dataset_hp{hp}_seed{seed}'
    Path(file_path).mkdir(exist_ok=True, parents=True)

    for record, no in zip(records, range(len(records))):
        matrix, ops, metrics = np.array(record[0]), record[1], record[2]

        # Labels Y
        if hp == '12':
            num_metrics = 3
        elif hp == '200':
            num_metrics = 2
        else:
            raise NotImplementedError('hp')

        y = np.zeros((num_metrics, int(hp)))  # (train_accuracy, validation_accuracy, test_accuracy) * epoch(12)
        metrics_list = ['train-accuracy', 'valid-accuracy']
        if hp == '12':
            metrics_list.append('test-accuracy')
        for i, j in enumerate(metrics_list):
            y[i] = np.array(metrics[j])

        graph = convert_matrix_ops_to_graph(matrix, ops)

        filename = os.path.join(file_path, f'graph_{no}.npz')
        np.savez(filename, a=graph.a, x=graph.x, y=y)
        print(f'graph_{no}.npz is saved.')


def generate_nb201_hash_to_metrics():
    file_path = '../NasBench201Dataset'
    api = create(None, 'tss', fast_mode=True, verbose=False)
    map_hash_to_metrics = {}
    for idx in tqdm(range(len(api))):
        arch_meta_info = api.query_meta_info_by_index(idx, hp='200')
        map_hash_to_metrics[arch_meta_info.arch_str] = {dataset: idx_to_metric(idx, dataset, api) for dataset in ['cifar10-valid', 'cifar100', 'ImageNet16-120']}

    with open(os.path.join(file_path, 'nb201_hash_to_metrics.pkl'), 'wb') as f:
        pickle.dump(map_hash_to_metrics, f)


def remove_last_subfolders(path):
    path = Path(path)
    parent_dir = path.parent
    folders = list(parent_dir.parts)
    new_folders = folders[:-1]
    new_path = Path(*new_folders)
    return str(new_path)


class NasBench201Dataset(Dataset):
    def __init__(self, start=0, end=15624, hp='200', seed=False, dataset='cifar10-valid', root='', **kwargs):
        """
        :param start:
        :param end:
        :param hp: 12 or 200 to set using 12 or 200 epochs dataset
        :param seed: if seed is False, will use average of all available trails
        :param dataset: 'cifar10-valid' or 'cifar100' or 'ImageNet16-120'
        :param root:
        :param kwargs:
        """
        self.file_path = os.path.join(root, 'NasBench201Dataset', dataset, f'NasBench201Dataset_hp{hp}_seed{seed}')
        self.start = start
        self.end = end
        self.hash_to_metrics = None
        super().__init__(**kwargs)

    def download(self):

        if not os.path.exists(self.file_path):
            print('Downloading...')
            file_name = wget.download('https://www.dropbox.com/s/925d04q9ko9wcgz/NasBench201Dataset.zip?dl=1')
            print('Save dataset to {}'.format(file_name))
            os.system('unzip {}'.format(file_name))
            print(f'Unzip dataset finish.')


    def read(self):
        with open(os.path.join(remove_last_subfolders(self.file_path), 'nb201_hash_to_metrics.pkl'), 'rb') as f:
            self.hash_to_metrics = pickle.load(f)

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
    hp = '200' # hp = 12 or 200
    datasets = ['cifar10-valid', 'cifar100', 'ImageNet16-120'] # cifar10-valid or cifar100 or ImageNet16-120
    generate_nb201_hash_to_metrics()
    for dataset in datasets:
        if hp == '12':
            seed_list = [111, 777]
        elif hp == '200':
            #seed_list = [777, 888, False] # 999
            seed_list = [False]

        for seed in seed_list:
            output_dir = os.path.join('../nb201_query_data', dataset)
            filename = f'hp{hp}_seed{seed}.pkl'
            with open(os.path.join(output_dir, filename), 'rb') as f:
                records = pickle.load(f)

            print(len(records))  # 15625
            transform_nb201_to_graph(records, hp=hp, seed=seed, dataset=dataset)
            datasets = NasBench201Dataset(0, 15624, dataset=dataset, hp=hp, seed=seed, root='../')
            print(datasets)
