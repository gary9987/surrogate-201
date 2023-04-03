import logging

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
from models.SVGe import SVGE_nvp, SVGE, BPRLoss


import argparse

parser = argparse.ArgumentParser(description='Surrogate-Model-training')
parser.add_argument('--model', type=str, default='SVGE_nvp')
parser.add_argument('--name', type=str, default='Train_PP')
parser.add_argument('--data_search_space', choices=['NB101', 'NB201'],
                    help='which search space for learning autoencoder', default='NB201')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--path_state_dict', type=str, help='directory to saved model', default='state_dicts/SVGE_NB201/')
parser.add_argument('--save_interval', type=int, default=50, help='how many epochs to wait to save model')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--checkpoint', type=int, default=300, help='Which checkpoint of trained model to load')
parser.add_argument('--on_valid', type=int, default=1, help='if predict on valid acc')
parser.add_argument('--finetune_SVGE', action='store_true', help='if fine tuning SVGE pretrained model', default=False)
parser.add_argument('--sample_amount', type=int, default=14061,
                    help='fine tuning VAE and surrogate on 14061 training data')
parser.add_argument('--criterion', type=str, default='MSELoss',
                    help='criterion function MSELoss or BPRLoss')
parser.add_argument('--criterion_reduction', type=str, default='sum',
                    help='criterion reduction mean or sum')

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
    model = eval(args.model)(model_config=model_config, data_config=data_config, dim_target=1).to(device)
    model_dict = model.state_dict()

    path_state_dict = args.path_state_dict
    checkpoint = args.checkpoint
    m = torch.load(os.path.join(path_state_dict, f"model_checkpoint{checkpoint}.obj"), map_location=device)

    m = {k: v for k, v in m.items() if k in model_dict}

    model_dict.update(m)
    model.load_state_dict(model_dict)

    if not args.finetune_SVGE:
        for name, param in model.named_parameters():
            if name in m:
                logging.info("Freeze the weight of %s", name)
                param.required_grad = False

    logging.info("param size = %fMB", util.count_parameters_in_MB(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['regression_learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    alpha = model_config['regression_loss_proportion']
    if args.criterion == 'MSELoss':
        criterion = eval(args.criterion)(reduction=args.criterion_reduction)
    elif args.criterion == 'BPRLoss':
        criterion = eval(args.criterion)(device, reduction=args.criterion_reduction)
    else:
        criterion = eval(args.criterion)()

    ##############################################################################
    #
    #                              Training
    #
    ##############################################################################

    budget = model_config['regression_epochs']
    for epoch in range(1, int(budget) + 1):
        logging.info('epoch: %s', epoch)

        # training
        _ = train(train_data, model, criterion, optimizer, epoch, device, alpha, data_config, log_dir)

        if epoch % args.save_interval == 0:
            logger.info('save model checkpoint {}  '.format(epoch))
            model_name = os.path.join(log_dir, 'model_checkpoint{}.obj'.format(epoch))
            torch.save(model.state_dict(), model_name)
            optimizer_name = os.path.join(log_dir, 'optimizer_checkpoint{}.obj'.format(epoch))
            torch.save(optimizer.state_dict(), optimizer_name)
            scheduler_name = os.path.join(log_dir, 'scheduler_checkpoint{}.obj'.format(epoch))
            torch.save(model.state_dict(), scheduler_name)

            # validation
        if epoch % 5 == 0:
            valid_obj = infer(val_data, model, criterion, optimizer, epoch, device, alpha, data_config,
                                             log_dir)


def train(train_data, model, criterion, optimizer, epoch, device, alpha, data_config, log_dir):
    objs = util.AvgrageMeter()
    vae_objs = util.AvgrageMeter()
    f_loss = util.AvgrageMeter()
    b_objs = util.AvgrageMeter()
    # TRAINING


    model.train()

    data_loader = DataLoader(train_data, shuffle=True, num_workers=data_config['num_workers'], pin_memory=True,
                             batch_size=data_config['batch_size'])
    for step, graph_batch in enumerate(data_loader):
        for i in range(len(graph_batch)):
            graph_batch[i].to(device)

        optimizer.zero_grad()
        vae_loss, recon_loss, kl_loss, f_loss, _ = model(graph_batch)
        loss = vae_loss + f_loss
        torch.mean(torch.squeeze(loss), 0)
        print(loss.shape)
        loss.backward()

        b_loss = model.backward(graph_batch)
        b_loss.backward()

        optimizer.step()
        n = graph_batch[0].num_graphs
        objs.update(loss.data.item() + b_loss.data.item(), n)
        vae_objs.update(vae_loss.data.item(), n)
        f_loss.update(f_loss.data.item(), n)
        b_objs.update(b_loss.data.item(), n)

    config_dict = {
        'epoch': epoch,
        'vae_loss': vae_objs.avg,
        'f_loss': f_loss.avg,
        'b_loss': b_objs.avg,
        'loss': objs.avg,
    }

    with open(os.path.join(log_dir, 'loss.txt'), 'a') as file:
        json.dump(str(config_dict), file)
        file.write('\n')

    logging.info('train %03d %.5f', step, objs.avg)
    return objs.avg


def infer(val_data, model, criterion, optimizer, epoch, device, alpha, data_config, log_dir):
    objs = util.AvgrageMeter()
    vae_objs = util.AvgrageMeter()
    acc_objs = util.AvgrageMeter()
    b_loss_objs = util.AvgrageMeter()

    # VALIDATION
    preds = []
    targets = []

    model.eval()
    data_loader = DataLoader(val_data, shuffle=False, num_workers=data_config['num_workers'],
                             batch_size=data_config['batch_size'])
    for step, graph_batch in enumerate(data_loader):
        with torch.no_grad():
            for i in range(len(graph_batch)):
                graph_batch[i].to(device)
            vae_loss, recon_loss, kl_loss, f_loss, _ = model(graph_batch)

            loss = vae_loss + f_loss
            b_loss = model.backward(graph_batch)

        n = graph_batch[0].num_graphs
        objs.update(loss.data.item(), n)
        vae_objs.update(vae_loss.data.item(), n)
        acc_objs.update(f_loss.data.item(), n)
        b_loss_objs.update(b_loss.data.item(), n)

    config_dict = {
        'epoch': epoch,
        'vae_loss_val': vae_objs.avg,
        'acc_loss_val': acc_objs.avg,
        'b_loss_val': b_loss_objs.avg,
        'loss-val': objs.avg,
    }

    with open(os.path.join(log_dir, 'loss.txt'), 'a') as file:
        json.dump(str(config_dict), file)
        file.write('\n')

    logging.info('val %03d %.5f', step, objs.avg)
    return objs.avg


if __name__ == '__main__':
    main(args)