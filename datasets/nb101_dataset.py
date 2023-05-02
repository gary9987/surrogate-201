# Partial code from: https://github.com/jovitalukasik/SVGe/blob/main/datasets/NASBench101.py
import logging
from pathlib import Path
from typing import Union
import spektral.data
import wget
from spektral.data import Dataset, Graph
import pickle
import numpy as np
import os

logger = logging.getLogger(__name__)

OP_PRIMITIVES_NB101 = [
    'output',
    'input',
    'conv1x1-bn-relu',
    'conv3x3-bn-relu',
    'maxpool3x3'
]

OPS_by_IDX_NB101 = {OP_PRIMITIVES_NB101.index(i):i for i in OP_PRIMITIVES_NB101}
OPS_NB101 = {i:OP_PRIMITIVES_NB101.index(i) for i in OP_PRIMITIVES_NB101}


def convert_matrix_ops_to_graph(matrix, ops):
    num_features = len(OP_PRIMITIVES_NB101)
    num_nodes = matrix.shape[0]

    # Node features X
    x = np.zeros((num_nodes, num_features), dtype=float)  # num_nodes * (features + metadata + num_layer)
    for i in range(len(ops)):
        x[i][OPS_NB101[ops[i]]] = 1

    # Adjacency matrix A
    a = np.array(matrix).astype(float)

    return spektral.data.Graph(x=x, a=a)


def transform_nb101_data_list_to_graph(records: dict):
    file_path = f'../NasBench101Dataset'
    Path(file_path).mkdir(exist_ok=True)
    epoch = 108
    for record, no in zip(records, range(len(records))):

        matrix, ops, metrics = np.array(record[0]), record[1], record[2]
        train_acc = 0.
        val_acc = 0.
        test_acc = 0.
        for repeat_index in range(len(metrics[epoch])):
            assert len(metrics[epoch]) == 3, 'len(computed_metrics[epoch]) should be 3'
            data_point = metrics[epoch][repeat_index]
            train_acc += data_point['final_train_accuracy']
            val_acc += data_point['final_validation_accuracy']
            test_acc += data_point['final_test_accuracy']

        train_acc = train_acc / 3.0
        val_acc = val_acc / 3.0
        test_acc = test_acc / 3.0

        y = np.array([[train_acc], [val_acc], [test_acc]], dtype=np.float32)

        graph = convert_matrix_ops_to_graph(matrix, ops)

        filename = os.path.join(file_path, f'graph_{no}.npz')
        np.savez(filename, a=graph.a, x=graph.x, y=y)
        logger.info(f'graph_{no}.npz is saved.')
        print(f'graph_{no}.npz is saved.')
        

class NasBench101Dataset(Dataset):
    def __init__(self, start=0, end=423623, root='', **kwargs):
        """
        :param start:
        :param end:
        :param root:
        :param kwargs:
        """
        self.file_path = os.path.join(root, 'NasBench101Dataset')
        self.start = start
        self.end = end
        super().__init__(**kwargs)

    def download(self):
        if not os.path.exists(self.file_path):
            print('Downloading...')
            file_name = wget.download('https://www.dropbox.com/s/luwbnie1vpsdvlv/NasBench101Dataset_new.zip?dl=1')
            print('Save dataset to {}'.format(file_name))
            os.system('unzip {}'.format(file_name))
            print(f'Unzip dataset finish.')

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
    output_dir = os.path.join('../nb101_query_data')
    filename = 'nb101_data_list.pkl'

    with open(os.path.join(output_dir, filename), 'rb') as f:
        records = pickle.load(f)

    print(len(records))  # 423624
    #transform_nb101_data_list_to_graph(records)
    datasets = NasBench101Dataset(root='../')
    print(datasets)
