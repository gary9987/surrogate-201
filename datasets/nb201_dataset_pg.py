import sys, pathlib
import os, glob, json
import torch
import numpy as np
import itertools
import tqdm
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from collections import OrderedDict
from datasets.nb201_dataset import NasBench201Dataset
from datasets.utils_data import prep_data

class Dataset:

    ##########################################################################
    def __init__(self, batch_size: int, hp: str, nb201_seed: int):

        self.hp = hp
        if __name__ == "__main__":
            path = os.path.join(".", "nasbench201")
        else:
            path = os.path.join(".", "datasets", "nasbench201")  # for debugging

        pathlib.Path(path).mkdir(exist_ok=True)

        file_cache_train = os.path.join(path, "cache_train")
        file_cache_test = os.path.join(path, "cache_test")
        file_cache = os.path.join(path, "cache")
        ############################################

        if not os.path.isfile(file_cache):
            nasbench = NasBench201Dataset(start=0, end=15624, hp=hp, seed=nb201_seed)
            self.data = []
            for graph in tqdm.tqdm(nasbench):
                self.data.append(self.map_item(graph))
                map_network = Dataset.map_network(graph)
                self.data[-1].edge_index = map_network[0]
                self.data[-1].node_atts = map_network[1]
                self.data[-1].num_nodes = graph.x.shape[0]

            print(f"Saving data to cache: {file_cache}")
            torch.save(self.data, file_cache)

        ############################################
        if not os.path.isfile(file_cache_train):
            print(f"Loading data from cache: {file_cache}")
            self.data = torch.load(file_cache)

            self.train_data, self.test_data = Dataset.sample(self.data)
            print('prepare data for Autoencoder Training')
            self.train_data = prep_data(self.train_data, max_num_nodes=8, NB201=True)

            print(f"Saving train data to cache: {file_cache_train}")
            torch.save(self.train_data, file_cache_train)

            print(f"Saving test data to cache: {file_cache_test}")
            torch.save(self.test_data, file_cache_test)

        else:
            print(f"Loading train data from cache: {file_cache_train}")
            self.train_data = torch.load(file_cache_train)

            print(f"Loading test data from cache: {file_cache_test}")
            self.test_data = torch.load(file_cache_test)

        ############################################

        self.length = len(self.train_data) + len(self.test_data)

        self.train_dataloader = DataLoader(
            self.train_data,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            batch_size=batch_size
        )

        self.test_dataloader = DataLoader(
            self.test_data,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            batch_size=batch_size
        )

        # self.dataloader = DataLoader(
        #     self.data,
        #     shuffle = True,
        #     num_workers = 4,
        #     pin_memory = True,
        #     batch_size = batch_size
        # )

    ##########################################################################
    def map_item(self, graph):
        train_acc = graph.y[0, :]
        valid_acc = graph.y[1, :]

        train_acc = torch.FloatTensor(train_acc / 100.0)
        valid_acc = torch.FloatTensor(valid_acc / 100.0)

        if self.hp == '12':
            test_acc = graph.y[2, :]
            test_acc = torch.FloatTensor([test_acc / 100.0])
            return Data(train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)

        return Data(train_acc=train_acc, valid_acc=valid_acc)

    ##########################################################################
    @staticmethod
    def map_network(graph):

        ops = np.argmax(graph.x, axis=1)
        node_attr = torch.LongTensor(ops)
        edge_index = torch.tensor(np.nonzero(graph.a))

        return edge_index.long(), node_attr


    ##########################################################################
    @staticmethod
    def sample(dataset,
               seed=999
               ):
        random_shuffle = np.random.permutation(range(len(dataset)))

        train_data = [dataset[i] for i in random_shuffle[:int(len(dataset) * 0.9)]]
        test_data = [dataset[i] for i in random_shuffle[int(len(dataset) * 0.9):]]

        return train_data, test_data

    ##########################################################################
    '''
    @staticmethod
    def pg_graph_to_nb201(pg_graph):
        # first tensor node attributes, second is the edge list
        ops = [OPS_by_IDX_201[i] for i in pg_graph.x.cpu().numpy()]
        matrix = np.array(to_dense_adj(pg_graph.edge_index)[0].cpu().numpy())
        try:
            if (matrix == ADJACENCY).all():
                steps_coding = ['0', '0', '1', '0', '1', '2']

                node_1 = '|' + ops[1] + '~' + steps_coding[0] + '|'
                node_2 = '|' + ops[2] + '~' + steps_coding[1] + '|' + ops[3] + '~' + steps_coding[2] + '|'
                node_3 = '|' + ops[4] + '~' + steps_coding[3] + '|' + ops[5] + '~' + steps_coding[4] + '|' + ops[
                    6] + '~' + steps_coding[5] + '|'
                nodes_nb201 = node_1 + '+' + node_2 + '+' + node_3
                index = nasbench.query_index_by_arch(nodes_nb201)
                acc = Dataset.map_item(index).acc
            else:
                acc = torch.zeros(1)
        except:
            acc = torch.zeros(1)

        return acc
    '''


##############################################################################
#
#                              Debugging
#
##############################################################################

if __name__ == "__main__":

    def print_keys(d, k=None, lvl=0):
        if k is not None:
            print(f"{'---' * (lvl)}{k}")
        if type(d) == list and len(d) == 1:
            d = d[0]
        if type(d) == dict:
            for k in d:
                print_keys(d[k], k, lvl + 1)


    ds = Dataset(10, '200', 777)
    for batch in ds.train_dataloader:
        print(batch)
        break