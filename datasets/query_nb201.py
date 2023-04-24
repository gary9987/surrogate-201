import os.path
import pickle
from typing import List, Tuple

from nats_bench import create
import numpy as np
from pathlib import Path

# Useful constants
OP_PRIMITIVES_201 = [
    'output',
    'input',
    'nor_conv_1x1',
    'nor_conv_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'none',
]

OPS_by_IDX_201 = {OP_PRIMITIVES_201.index(i):i for i in OP_PRIMITIVES_201}
OPS_201 = {i:OP_PRIMITIVES_201.index(i) for i in OP_PRIMITIVES_201}

ADJACENCY = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1 ,0 ,0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0]])

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
    # optimal valid 91.60666665039064
    output_dir = '../nb201_query_data'
    Path(output_dir).mkdir(exist_ok=True)

    hp = '200'  # can be 12 or 200 for cifar-10
    # is_random For hp=12 seed={111, 777}
    # is_random For hp=200 seed={777, 888, 999}
    # seed 999 data is not completed
    if hp == '12':
        seed_list = [111, 777]
    elif hp == '200':
        seed_list = [777, 888, False] # 999

    for is_random in seed_list:
        # Create the API instance for the topology search space in NATS
        api = create(None, 'tss', fast_mode=True, verbose=False)
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

            arch = api.query_meta_info_by_index(idx, hp=hp)
            total_train_epo = arch.get_total_epoch('cifar10-valid')  # 12 for cifar10 training

            for epoch in range(total_train_epo):
                try:
                    info = api.get_more_info(idx, 'cifar10-valid', iepoch=epoch, hp=hp, is_random=is_random)
                    count += 1
                    for metric in metrics:
                        record[metric].append(info[metric])
                except:
                    print(f'no data for idx {idx}')
                    break

            #arch_str = api.query_info_str_by_arch(idx)
            arch_str = api.query_meta_info_by_index(idx).arch_str
            adj_matrix, ops_list = convert_arch_str_to_martrix_ops(arch_str)
            final.append([adj_matrix, ops_list, record])

        print(f'count = {count / total_train_epo}')
        filename = f'hp{hp}_seed{is_random}.pkl'
        with open(os.path.join(output_dir, filename), 'wb') as file:
            pickle.dump(final, file)

