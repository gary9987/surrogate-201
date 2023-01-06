import pickle
from nats_bench import create
import numpy as np


if __name__ == '__main__':
    hp = '12' # can be 12 or 200 for cifar-10
    # For hp=12 seed={111, 777}
    # For hp=200 seed={777, 888, 999}

    is_random = 111
    api = create(None, 'tss', fast_mode=True, verbose=False)
    # Create the API instance for the topology search space in NATS
    template_array = np.zeros((8, 8), dtype=int)
    template_array[0][1] = template_array[0][2] = template_array[0][3] = 1
    template_array[1][4] = template_array[1][6] = 1
    template_array[2][5] = 1
    template_array[3][7] = 1
    template_array[4][6] = 1
    template_array[5][7] = template_array[6][7] = 1

    final = []

    metrics = [
        'train-accuracy',
        'train-loss',
        'valid-accuracy',
        'valid-loss',
        'test-accuracy',
        'test-loss'
    ]

    for idx in range(len(api)):
        print('start model NO. {}'.format(idx))
        record = {metric: [] for metric in metrics}

        arch = api.query_meta_info_by_index(idx, hp=hp)
        total_train_epo = arch.get_total_epoch('cifar10-valid')  # 12 for cifar10 training

        for epoch in range(total_train_epo):
            info = api.get_more_info(idx, 'cifar10-valid', iepoch=epoch, hp=hp, is_random=is_random)
            for metric in metrics:
                record[metric].append(info[metric])

        arch_str = api.query_info_str_by_arch(idx).split('\n')[0]
        tmp_list = api.str2lists(arch_str)

        arch_list = ['INPUT']
        for j in range(3):
            for k in range(3):
                if j < len(tmp_list[k]):
                    arch_list.append(tmp_list[k][j][0])
        arch_list.append('OUTPUT')
        now_array = template_array.copy()
        final.append([now_array, arch_list, record])

    filename = 'model_label.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(final, file)

