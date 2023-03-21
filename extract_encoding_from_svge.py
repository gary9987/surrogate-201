import logging
import pickle

logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm
import numpy as np
import json
import random
import os
import sys
import time
from scipy.io import loadmat
import torch
import torch.nn as nn
from torch.nn import MSELoss
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader
from ConfigSpace.read_and_write import json as config_space_json_r_w
from utils import util
from datasets.utils_data import prep_data
from models.SVGe import SVGE_acc, SVGE, BPRLoss

import argparse

parser = argparse.ArgumentParser(description='SVGE Encoding')
parser.add_argument('--model', type=str, default='SVGE_acc')
parser.add_argument('--name', type=str, default='Train_PP')
parser.add_argument('--data_search_space', choices=['NB101', 'NB201'],
                    help='which search space for learning autoencoder', default='NB201')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--path_state_dict', type=str, help='directory to saved model', default='state_dicts/SVGE_NB201/')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--checkpoint', type=int, default=300, help='Which checkpoint of trained model to load')
parser.add_argument('--on_valid', type=int, default=1, help='if predict on valid acc')
args = parser.parse_args()

seed = args.seed
print(f"Random Seed: {seed}")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
device = args.device

##############################################################################
#
#                              Runfolder
#
##############################################################################
# Create Log Directory
now = datetime.now()
runfolder = now.strftime("%Y_%m_%d_%H_%M_%S")
runfolder = f"{args.data_search_space}/{args.model}/{args.data_search_space}/{runfolder}_{args.name}"

FOLDER_EXPERIMENTS = os.path.join(os.getcwd(), 'Experiments/Surrogate/')
log_dir = os.path.join(FOLDER_EXPERIMENTS, runfolder)
os.makedirs(log_dir)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(log_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main(args):
    ##############################################################################
    #
    #                           Dataset Config
    #
    ##############################################################################

    if args.data_search_space == 'NB101':
        data_config_path = 'configs/data_configs/NB101_configspace.json'
    elif args.data_search_space == 'NB201':
        data_config_path = 'configs/data_configs/NB201_configspace.json'
    else:
        raise TypeError("Unknow Seach Space : {:}".format(args.data_search_space))
    # Get Data specific configs
    data_config = json.load(open(data_config_path, 'r'))

    ##############################################################################
    #
    #                           Model Config
    #
    ##############################################################################
    # Get Model configs
    model_config_path = 'configs/model_configs/svge_configspace.json'
    model_configspace = config_space_json_r_w.read(open(model_config_path, 'r').read())
    model_config = model_configspace.get_default_configuration().get_dictionary()

    ##############################################################################
    #
    #                              Dataset
    #
    ##############################################################################
    print("Creating Dataset.")
    if args.data_search_space == 'NB201':
        from datasets.nb201_dataset_pg import Dataset as Dataset201
        dataset = Dataset201(batch_size=data_config['batch_size'], hp=data_config['hp'],
                             nb201_seed=data_config['nb201_seed'])
        print("Prepare Test Set")
        train_data = dataset.train_data
        val_data = dataset.test_data
        val_data = prep_data(val_data, data_config['max_num_nodes'], NB201=True)
    elif args.data_search_space == 'NB101':
        from datasets.NASBench101 import Dataset as Dataset101
        dataset = Dataset101(batch_size=data_config['batch_size'])
        print("Prepare Test Set")
        val_data = dataset.test_data
        val_data = prep_data(val_data, data_config['max_num_nodes'], NB101=True)
    else:
        raise TypeError("Unknow Dataset: {:}".format(args.data_search_space))

    ##############################################################################
    #
    #                              Model
    #
    ##############################################################################
    model = eval(args.model)(model_config=model_config, data_config=data_config, dim_target=200).to(device)
    path_state_dict = args.path_state_dict
    checkpoint = args.checkpoint
    model.load_state_dict(
        torch.load(os.path.join(path_state_dict, f"model_checkpoint{checkpoint}.obj"), map_location=device))

    logging.info("param size = %fMB", util.count_parameters_in_MB(model))

    ##############################################################################
    #
    #                              Infer
    #
    ##############################################################################
    infer(train_data, model, device, data_config, log_dir, f'{args.data_search_space}_train.pkl')
    infer(val_data, model, device, data_config, log_dir, f'{args.data_search_space}_test.pkl')


def infer(data, model, device, data_config, output_dir, filename):
    encodings = []
    labels = []

    model.eval()
    data_loader = DataLoader(data, shuffle=False, num_workers=data_config['num_workers'],
                             batch_size=data_config['batch_size'])
    for graph_batch in tqdm(data_loader):
        with torch.no_grad():
            for i in range(len(graph_batch)):
                graph_batch[i].to(device)

            mean, _ = model.Encoder(graph_batch[0].edge_index,
                                    graph_batch[0].node_atts,
                                    graph_batch[0].batch)

            if args.on_valid:
                acc = torch.reshape(graph_batch[0].valid_acc,
                                    (graph_batch[0].valid_acc.size(0) // data_config['hp'], data_config['hp']))
            else:
                acc = torch.reshape(graph_batch[0].train_acc,
                                    (graph_batch[0].train_acc.size(0) // data_config['hp'], data_config['hp']))

            encoding = mean.detach().cpu().numpy()
            label = acc.detach().cpu().numpy()
            print(encoding.shape, label.shape)
            encodings.extend(encoding)
            labels.extend(label)

    with open(os.path.join(output_dir, filename), 'wb') as file:
        pickle.dump({'encodings': encodings, 'labels': labels}, file)


if __name__ == '__main__':
    main(args)
