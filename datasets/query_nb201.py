import os.path
import pickle
from typing import List, Tuple
from nats_bench import create
import numpy as np
from pathlib import Path
from datasets.nb201_dataset import ADJACENCY


# Partial reference from https://github.com/automl/SVGe/blob/main/datasets/NASBench201.py
def convert_arch_str_to_martrix_ops(arch_str: str) -> Tuple[np.ndarray, List[str]]:

    nodes = ['input']
    steps = arch_str.split('+')
    steps_coding = ['0', '0', '1', '0', '1', '2']
    cont = 0
    for step in steps:
        step = step.strip('|').split('|')
        for node in step:
            n, idx = node.split('~')
            assert idx == steps_coding[cont]
            cont += 1
            nodes.append(n)
    nodes.append('output')

    return ADJACENCY, nodes


if __name__ == '__main__':
    # cifar10 optimal valid 91.60666665039064

    on_dataset = 'cifar10-valid'  # cifar10-valid cifar100 ImageNet16-120
    output_dir = os.path.join('../nb201_query_data', on_dataset)
    Path(output_dir).mkdir(exist_ok=True)

    hp = '200'  # can be 12 or 200 for cifar-10
    # is_random For hp=12 seed={111, 777}
    # is_random For hp=200 seed={777, 888, 999}
    # seed 999 data is not completed
    if hp == '12':
        seed_list = [111, 777]
    elif hp == '200':
        #seed_list = [777, 888, False] # 999
        seed_list = [False]

    # Create the API instance for the topology search space in NATS
    api = create(None, 'tss', fast_mode=True, verbose=False)
    for is_random in seed_list:
        final = []
        metrics = [
            'train-accuracy',
            'train-loss',
            'valid-accuracy',
            'valid-loss'
        ]
        if hp == '12':
            metrics += ['test-accuracy', 'test-loss']

        count = 0
        for idx in range(len(api)):
            print('start model NO. {}'.format(idx))
            record = {metric: [] for metric in metrics}

            arch_meta_info = api.query_meta_info_by_index(idx, hp=hp)
            total_train_epo = arch_meta_info.get_total_epoch(on_dataset)  # 12 for cifar10 training

            for epoch in range(total_train_epo):
                try:
                    metric = arch_meta_info.get_metrics(on_dataset, 'train', iepoch=epoch, is_random=is_random)
                    record['train-accuracy'].append(metric['accuracy'])
                    record['train-loss'].append(metric['loss'])
                    if on_dataset != 'cifar10-valid':
                        epoch = None
                    metric = arch_meta_info.get_metrics(on_dataset, 'x-valid', iepoch=epoch, is_random=is_random)
                    record['valid-accuracy'].append(metric['accuracy'])
                    record['valid-loss'].append(metric['loss'])
                    count += 1
                except:
                    print(f'no data for idx {idx}')
                    break

            adj_matrix, ops_list = convert_arch_str_to_martrix_ops(arch_meta_info.arch_str)
            final.append([adj_matrix, ops_list, record])

        print(f'count = {count / total_train_epo}')
        filename = f'hp{hp}_seed{is_random}.pkl'
        with open(os.path.join(output_dir, filename), 'wb') as file:
            pickle.dump(final, file)

