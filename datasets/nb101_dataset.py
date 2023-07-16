# Partial code from: https://github.com/jovitalukasik/SVGe/blob/main/datasets/NASBench101.py
import copy
import logging
from pathlib import Path
import spektral.data
import wget
from spektral.data import Dataset, Graph
import pickle
import numpy as np
import os
from datasets.nasbench_lib.model_spec import ModelSpec

logger = logging.getLogger(__name__)

NB101_CANONICAL_OPS = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
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
    map_hash_to_metrics = {}
    for record, no in zip(records, range(len(records))):

        matrix, ops, metrics = np.array(record[0]), record[1], record[2]

        spec_hash = ModelSpec(matrix=matrix, ops=[i for i in ops]).hash_spec(NB101_CANONICAL_OPS)

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
        map_hash_to_metrics[spec_hash] = y

        graph = convert_matrix_ops_to_graph(matrix, ops)

        filename = os.path.join(file_path, f'graph_{no}.npz')
        np.savez(filename, a=graph.a, x=graph.x, y=y)
        logger.info(f'graph_{no}.npz is saved.')


        print(f'graph_{no}.npz is saved.')

    with open(os.path.join(file_path, 'nb101_hash_to_metrics.pkl'), 'wb') as f:
        pickle.dump(map_hash_to_metrics, f)


def mask_padding_vertex_for_model(a, x):
    """
    virtual node is padded with 0 for ops
    :param x: (nodes, num_ops)
    :param a: (nodes, nodes)
    :return: x: (nodes, num_ops), a: (nodes, nodes) if a and x are valid else None, None
    """
    try:
        ops = np.argmax(x, axis=-1).tolist()
        output_idx = ops.index(0)
        for i in range(output_idx+1, x.shape[0]):
            x[i] = np.zeros(x.shape[1])
        new_a = np.zeros(a.shape)
        new_a[:output_idx, :output_idx+1] = a[:output_idx, :output_idx+1]  # Set output node no connect to other nodes
    except:
        return None, None
    return new_a, x


def mask_padding_vertex_for_spec(a, x):
    """
    Return the ops and adj for nb101 ModelSpec
    :param x: (nodes, num_ops) or (nodes)
    :param a: (nodes, nodes)
    :return: a: (output_idx+1, output_idx+1), x: (output_idx+1)
    """
    if isinstance(x, list):
        ops = x
    else:
        ops = np.argmax(x, axis=-1).tolist()
    output_idx = ops.index(0)
    new_x = x[:output_idx+1]
    new_a = a[:output_idx+1, :output_idx+1]
    return new_a, new_x


def mask_for_model(arch):
    new_arch = copy.deepcopy(arch)
    new_arch['a'], new_arch['x'] = mask_padding_vertex_for_model(new_arch['a'], new_arch['x'])
    if new_arch['a'] is None:
        return None
    return new_arch


def mask_for_spec(arch):
    new_arch = copy.deepcopy(arch)
    new_arch['a'], new_arch['x'] = mask_padding_vertex_for_spec(new_arch['a'], new_arch['x'])
    return new_arch


def pad_nb101_graph(graph: spektral.data.Graph):
    new_graph = copy.deepcopy(graph)
    new_graph.a = np.zeros((7, 7))
    new_graph.x = np.zeros((7, 5))
    a_shape = graph.a.shape
    x_shape = graph.x.shape

    new_graph.a[:a_shape[0], :a_shape[1]] = graph.a
    new_graph.x[:x_shape[0], :x_shape[1]] = graph.x
    return new_graph


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
        self.hash_to_metrics = None
        super().__init__(**kwargs)

    def download(self):
        if not os.path.exists(self.file_path):
            print('Downloading...')
            file_name = wget.download('https://www.dropbox.com/s/luwbnie1vpsdvlv/NasBench101Dataset_new.zip?dl=1')
            print('Save dataset to {}'.format(file_name))
            os.system('unzip {}'.format(file_name))
            print(f'Unzip dataset finish.')

    def read(self):
        with open(os.path.join(self.file_path, 'nb101_hash_to_metrics.pkl'), 'rb') as f:
            self.hash_to_metrics = pickle.load(f)

        output = []
        filename_list = []

        for i in range(len(os.listdir(self.file_path)) - 1):  # include hash_to_metrics.pkl
            filename_list.append(os.path.join(self.file_path, f'graph_{i}.npz'))

        assert len(filename_list) > self.end
        assert self.start >= 0

        for i in range(self.start, self.end + 1):
            data = np.load(filename_list[i])
            output.append(Graph(x=data['x'], a=data['a'], y=data['y']))

        return output

    def get_spec_hash(self, matrix, ops):
        if isinstance(ops[0], int) or isinstance(ops, np.ndarray):
            ops = list(ops)
            ops = [OPS_by_IDX_NB101[i] for i in ops]
        matrix = np.array(matrix).astype(np.int8)
        spec_hash = ModelSpec(matrix=matrix, ops=ops).hash_spec(NB101_CANONICAL_OPS)
        return spec_hash

    def get_metrics(self, matrix, ops):
        """
        :param matrix: adj matrix for ModelSpec format
        :param ops: ops for ModelSpec format
        :return: dict: The metrics for the architecture corresponding to the given matrix and ops
        """
        return self.hash_to_metrics[self.get_spec_hash(matrix, ops)]


if __name__ == '__main__':
    output_dir = os.path.join('../nb101_query_data')
    filename = 'nb101_data_list.pkl'

    with open(os.path.join(output_dir, filename), 'rb') as f:
        records = pickle.load(f)

    print(len(records))  # 423624
    transform_nb101_data_list_to_graph(records)
    datasets = NasBench101Dataset(end=1000, root='../')
    matrix = np.array([[0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0]])

    ops = np.array([[0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0]])
    mask_padding_vertex_for_model(matrix, ops)
    matrix, ops = mask_padding_vertex_for_spec(matrix, ops)
    metrics = datasets.get_metrics(matrix, np.argmax(ops, axis=-1))
    print(metrics)

    # The following are identical
    matrix = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 0]])
    ops = ['input', 'conv3x3-bn-relu', 'output']
    metrics = datasets.get_metrics(matrix, ops)
    print(metrics)
    ops = [1, 3, 0]
    metrics = datasets.get_metrics(matrix, ops)
    print(metrics)
    matrix = np.array([[0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0]])
    ops = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'output']
    metrics = datasets.get_metrics(matrix, ops)
    print(metrics)
    matrix = np.array([[0, 0, 1, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0]])
    ops = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'output']
    metrics = datasets.get_metrics(matrix, ops)
    print(metrics)
    print(datasets)
    ops = [1, 2, 2, 3, 3, 3, 0]