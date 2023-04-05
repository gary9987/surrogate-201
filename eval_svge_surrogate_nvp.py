import logging

logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import os
import sys
import torch
from datetime import datetime
from torch_geometric.data import Data, DataLoader
from ConfigSpace.read_and_write import json as config_space_json_r_w
from utils import util
from datasets.utils_data import prep_data
from datasets.query_nb201 import OPS_by_IDX_201
from datasets.utils import ops_list_to_nb201_arch_str
from nats_bench import create
from models.SVGe import SVGE_nvp
import argparse
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


parser = argparse.ArgumentParser(description='Surrogate-Model-training')
parser.add_argument('--model', type=str, default='SVGE_nvp')
parser.add_argument('--name', type=str, default='Train_PP')
parser.add_argument('--data_search_space', choices=['NB101', 'NB201'],
                    help='which search space for learning autoencoder', default='NB201')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--path_state_dict', type=str, help='directory to saved model', default='Experiments/Surrogate/NB201/SVGE_nvp/14061/2023_04_04_13_51_36_Train_PP')
parser.add_argument('--save_interval', type=int, default=50, help='how many epochs to wait to save model')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--checkpoint', type=int, default=150, help='Which checkpoint of trained model to load')
parser.add_argument('--on_valid', type=int, default=1, help='if predict on valid acc')
parser.add_argument('--sample_amount', type=int, default=14061,
                    help='fine tuning VAE and surrogate on 14061 training data')


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
runfolder = f"{args.data_search_space}/{args.model}/{args.sample_amount}/{runfolder}_{args.name}"

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
        dataset = Dataset201(batch_size=data_config['batch_size'], hp=data_config['hp'], nb201_seed=data_config['nb201_seed'])
        print("Prepare Test Set")
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

    # Sample train Data
    data = dataset.train_data
    random_shuffle = np.random.permutation(range(len(data)))
    train_data = [data[i] for i in random_shuffle[:args.sample_amount]]
    print(f"Dataset size: {len(train_data)}")

    torch.save(random_shuffle[:args.sample_amount], os.path.join(log_dir, 'sampled_train_idx.pth'))

    ##############################################################################
    #
    #                              Model
    #
    ##############################################################################
    model = eval(args.model)(model_config=model_config, data_config=data_config, dim_target=1, device=device).to(device)
    model_dict = model.state_dict()

    path_state_dict = args.path_state_dict
    checkpoint = args.checkpoint
    m = torch.load(os.path.join(path_state_dict, f"model_checkpoint{checkpoint}.obj"), map_location=device)

    m = {k: v for k, v in m.items() if k in model_dict}

    model_dict.update(m)
    model.load_state_dict(model_dict)

    logging.info("param size = %fMB", util.count_parameters_in_MB(model))

    ##############################################################################
    #
    #                              Training
    #
    ##############################################################################

    invert(train_data, model, device, data_config)


def invert(train_data, model, device, data_config):
    # TRAINING
    x = []
    y = []
    model.eval()
    nb201api = create(None, 'tss', fast_mode=True, verbose=False)

    data_loader = DataLoader(train_data, shuffle=True, num_workers=data_config['num_workers'], pin_memory=False,
                             batch_size=256)

    for step, graph_batch in enumerate(data_loader):

        for i in range(len(graph_batch)):
            graph_batch[i].to(device)

        try:
            edges, node_atts, edge_list = model.invert(graph_batch, num_samples_z=1000)
            label_acc = torch.reshape(graph_batch[0].valid_acc, (node_atts.size(0), -1))[:, -1].cpu().numpy().tolist()
            ops_idxs = node_atts.cpu().numpy().tolist()
            for ops_idx, q_acc in zip(ops_idxs, label_acc):
                ops = [OPS_by_IDX_201[i] for i in ops_idx]
                arch_str = ops_list_to_nb201_arch_str(ops)

                idx = nb201api.query_index_by_arch(arch_str)
                acc_l = [q_acc]
                l_acc = 0
                for seed in [777]:
                    data = nb201api.get_more_info(idx, 'cifar10-valid', iepoch=199, hp='200', is_random=seed)
                    # print(data['valid-accuracy'])
                    l_acc += data['valid-accuracy'] / 100.
                    acc_l.append(data['valid-accuracy'] / 100.)
                    # print(data['test-accuracy'])

                x.append(q_acc)
                y.append(l_acc)
                print(acc_l)
        except:
            print('invalid')

    plt.scatter(x, y, s=[1] * len(x))
    plt.xlim(0.0, 1.)
    plt.ylim(0.0, 1.)
    plt.show()
    plt.savefig('scatter0.png')
    plt.cla()

    x = []
    y = []
    acc = 0.9500
    while acc >= 0.0500:
        try:
            edges, node_atts, edge_list = model.invert_from_acc(acc + 0.001 * random.random(), num_samples_z=1000)
            q_acc = acc
            ops_idxs = node_atts.cpu().numpy().tolist()
            for ops_idx in ops_idxs:
                ops = [OPS_by_IDX_201[i] for i in ops_idx]
                arch_str = ops_list_to_nb201_arch_str(ops)

                idx = nb201api.query_index_by_arch(arch_str)
                acc_l = [q_acc]
                l_acc = 0
                for seed in [777]:
                    data = nb201api.get_more_info(idx, 'cifar10-valid', iepoch=199, hp='200', is_random=seed)
                    # print(data['valid-accuracy'])
                    l_acc += data['valid-accuracy'] / 100.
                    acc_l.append(data['valid-accuracy'] / 100.)
                    # print(data['test-accuracy'])

                x.append(q_acc)
                y.append(l_acc)
                print(acc_l)
        except:
            print('invalid')
        acc -= 0.005

    plt.scatter(x, y, s=[1] * len(x))
    plt.xlim(0.0, 1.)
    plt.ylim(0.0, 1.)
    plt.show()
    plt.savefig('scatter1.png')


if __name__ == '__main__':
    main(args)