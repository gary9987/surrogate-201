import logging
import random
from spektral.data import Dataset, Graph
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import os
import wget
from os import path
import re
import hashlib
from keras import backend as K
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.python.profiler.model_analyzer import profile
import csv

logging.basicConfig(filename='nb_201_dataset.log', level=logging.INFO)


def get_hash(id, layer):
    # calculate hash key of this model
    model_string = str(id) + '_' + str(layer)
    hash_sha256 = hashlib.sha256()
    hash_sha256.update(model_string.encode("utf-8"))
    hash = hash_sha256.hexdigest()

    return hash


def transform_nb201_to_graph(records: dict):
    features_dict = {'INPUT': 0, 'none': 1, 'skip_connect': 2, 'nor_conv_1x1': 3, 'nor_conv_3x3': 4,
                     'avg_pool_3x3': 5, 'OUTPUT': 6}

    num_features = len(features_dict)
    file_path = 'NasBench201Dataset'

    if not os.path.exists(file_path):
        os.mkdir(file_path)


    for record, no in zip(records, range(len(records))):

        matrix, ops, metrics = np.array(record[0]), record[1], record[2]
        nodes = matrix.shape[0]

        # Labels Y
        y = np.zeros((3, 12))  # (train_accuracy, validation_accuracy, test_accuracy) * epoch(12)
        for i, j in enumerate(['train-accuracy', 'valid-accuracy', 'test-accuracy']):
            y[i] = np.array(metrics[j])

        # Node features X
        x = np.zeros((nodes, num_features), dtype=float)  # nodes * (features + metadata + num_layer)
        for i in range(len(ops)):
            x[i][features_dict[ops[i]]] = 1

        # Adjacency matrix A
        adj_matrix = np.array(metrics)


        filename = os.path.join(file_path, f'graph_{no}.npz')
        np.savez(filename, a=adj_matrix, x=x, y=y)
        logging.info(f'graph_{no}.npz is saved.')


class NasBench201Dataset(Dataset):
    def __init__(self, start: int, end: int, matrix_size: int=None, matrix_size_list: list=None, record_dic: list=None,
                 shuffle_seed=0, shuffle=True, inputs_shape=None, num_classes=10,
                 preprocessed=False, repeat=1, mid_point: int=None, request_lower: bool=None, **kwargs):
        """
        :param start: The start index of data you want to query.
        :param end: The end index of data you want to query.
        :param record_dic: open('./nas-bench-101-data/nasbench_101_cell_list.pkl', 'rb')
        :param matrix_size: set the size of matrix to process
        :param matrix_size_list: set the list of size of matrix to load as dataset
        :param shuffle_seed: 0
        :param shuffle: shuffle when load the data, it prevents low acc data concentrated in frontend of data
        :param inputs_shape: (None, 32, 32, 3)
        :param num_classes: Number of the classes of the dataset
        :param preprocessed: Use the preprocessed dataset
        :param repeat: if repeat > 1 then mid_point is required to set, this repeat the data which acc lower than midpoint

        Direct use the dataset with set the start and end parameters,
        or if you want to preprocess again, unmark the marked download() function and set the all parameters.
        """
        self.nodes = 67
        self.features_dict = {'INPUT': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'OUTPUT': 4,
                              'Classifier': 5, 'maxpool2x2': 6, 'flops': 7, 'params': 8, 'num_layer': 9,
                              'input_shape_1': 10, 'input_shape_2': 11, 'input_shape_3': 12, 'output_shape_1': 13,
                              'output_shape_2': 14, 'output_shape_3': 15}

        self.num_features = len(self.features_dict)
        self.inputs_shape = inputs_shape
        self.num_classes = num_classes
        self.preprocessed = preprocessed
        if preprocessed:
            self.file_path_prefix = 'Preprocessed_NasBench101Dataset'
            self.file_path_suffix = 'Preprocessed_NasBench101Dataset_'
        else:
            self.file_path_prefix = 'NasBench101Dataset'
            self.file_path_suffix = 'NasBench101Dataset_'

        self.file_path = self.file_path_prefix + '/' + self.file_path_suffix + f'{matrix_size}'
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.cell_filename = f'./nas-bench-101-data/nasbench_101_cell_list_{matrix_size}.pkl'
        self.total_layers = 11
        self.record_dic = record_dic
        self.repeat = repeat
        if mid_point is not None and mid_point >= 10:
            raise Exception("the range of mid_point is < 10")
        self.mid_point = mid_point
        self.request_lower = request_lower
        self.matrix_size = matrix_size
        self.matrix_size_list = matrix_size_list
        self.start = start
        self.end = end

        if self.record_dic is not None:
            random.seed(shuffle_seed)
            random.shuffle(self.record_dic)

        super().__init__(**kwargs)


    def download(self):
        if not os.path.exists(self.file_path_prefix):
            print('Downloading...')
            if self.preprocessed:
                file_name = wget.download('https://www.dropbox.com/s/muetcgm9l1e01mc/Preprocessed_NasBench101Dataset.zip?dl=1')
            else:
                file_name = wget.download('https://www.dropbox.com/s/40lrvb3lcgij5c8/NasBench101Dataset.zip?dl=1')
            print('Save dataset to {}'.format(file_name))
            os.system('unzip {}'.format(file_name))
            print(f'Unzip dataset finish.')


    def read(self):
        if self.repeat > 1 or self.request_lower is not None:
            if self.mid_point is None:
                raise Exception("mid_point is not set")

        output = []
        filename_list = []

        if self.matrix_size_list is not None:
            matrix_size_list = self.matrix_size_list
        else:
            matrix_size_list = [self.matrix_size]

        for size in matrix_size_list:
            path = self.file_path_prefix + '/' + self.file_path_suffix + f'{size}'
            for i in range(len(os.listdir(path))):
                #with np.load(os.path.join(path, f'graph_{i}.npz')) as npz:
                #    data = {'x': npz['x'], 'e': npz['e'], 'a': npz['a'], 'y': npz['y']}
                filename_list.append(os.path.join(path, f'graph_{i}.npz'))

        if self.shuffle:
            random.seed(self.shuffle_seed)
            random.shuffle(filename_list)

        for i in range(self.start, self.end + 1):
            if i >= len(filename_list):
                print(f'The len of data is {len(filename_list)}')
                break

            data = np.load(filename_list[i])

            if self.preprocessed:
                if np.isnan(data['y'][0][0]) and np.isnan(data['y'][1][0]) and np.isnan(data['y'][2][0]):
                    continue

            highest_valid_acc = find_highest_valid_data(data['y'])

            # request split mode
            if self.request_lower is not None:
                # valid acc < mid_point
                if self.request_lower:
                    if highest_valid_acc <= self.mid_point:
                        output.append(Graph(x=data['x'], e=data['e'], a=data['a'], y=data['y']))
                else:
                    if highest_valid_acc > self.mid_point:
                        output.append(Graph(x=data['x'], e=data['e'], a=data['a'], y=data['y']))

            # normal mode
            else:
                if self.repeat > 1:
                    # valid acc < mid_point
                    if highest_valid_acc <= self.mid_point:
                        for _ in range(self.repeat):
                            output.append(Graph(x=data['x'], e=data['e'], a=data['a'], y=data['y']))
                    else:
                        output.append(Graph(x=data['x'], e=data['e'], a=data['a'], y=data['y']))
                else:
                    output.append(Graph(x=data['x'], e=data['e'], a=data['a'], y=data['y']))

        return output


if __name__ == '__main__':
    with open('model_label.pkl', 'rb') as f:
        records = pickle.load(f)

    print(len(records))
    transform_nb201_to_graph(records)
