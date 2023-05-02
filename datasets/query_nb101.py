# nasbench should be running on tf1.x
import os
import pickle
import subprocess
from pathlib import Path
from nasbench import api
from tqdm import tqdm


##############################################################################
# NAS-Bench-101 Data STRUCTURE .tfrecord
##############################################################################
# ---nasbench.hash_iterator() : individual hash for each graph in the whole .tfrecord dataset
# ------nasbench.get_metrics_from_hash(unique_hash): metrics of data sample given by the hash
# ---------fixed_metrics: {'module_adjacency': array([[0, 1, 0, 0, ...type=int8),
#                         'module_operations': ['input', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'output'],
#                         'trainable_parameters': 8555530}
# ---------computed_metrics : dict including the epoch and all metrics which are computed for each architecture{108: [{...}, {...}, {...}]}
# ------------computed_metrics[108]: [{'final_test_accuracy': 0.9211738705635071,
#                                  'final_train_accuracy': 1.0,
#                                  'final_training_time': 1769.1279296875,
#                                  'final_validation_accuracy': 0.9241786599159241,
#                                  'halfway_test_accuracy': 0.7740384340286255,
#                                  'halfway_train_accuracy': 0.8282251358032227,
#                                  'halfway_training_time': 883.4580078125,
#                                  'halfway_validation_accuracy': 0.7776442170143127},
#                                  {...},{...}]
##############################################################################


def query_by_hash_iter():
    nasbench_data_dir = Path('nb101_query_data')
    nasbench_only108_filename = Path('nasbench_only108.tfrecord')

    # Auto download the nasbench_only108.tfrecord
    if not os.path.exists(nasbench_data_dir / nasbench_only108_filename):
        os.makedirs(nasbench_data_dir)
        subprocess.check_output(
            f'wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord; mv {nasbench_only108_filename.name} {nasbench_data_dir}',
            shell=True)

    nasbench = api.NASBench(str(nasbench_data_dir / nasbench_only108_filename))

    data_list = []
    for unique_hash in tqdm(nasbench.hash_iterator()):
        fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
        adj = fixed_metrics['module_adjacency']
        ops = fixed_metrics['module_operations']
        data_list.append([adj, ops, computed_metrics])

    with open(nasbench_data_dir / Path(f'nb101_data_list.pkl'), 'wb') as f:
        pickle.dump(data_list, f)

    print(f'Finish')


if __name__ == '__main__':
    query_by_hash_iter()




