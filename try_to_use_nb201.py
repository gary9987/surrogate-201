import pickle
from nats_bench import create
import numpy as np


if __name__ == '__main__':
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

    for idx in range(len(api)):
        print('start model NO. {}'.format(idx))
        record = {}
        arch = api.query_meta_info_by_index(idx)
        total_train_epo = arch.get_total_epoch('cifar10-valid')  # 12 for cifar10 training

        for epoch in total_train_epo:
            info = api.get_more_info(idx, 'cifar10', iepoch=epoch)
            validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(idx,
                                                                                                       dataset='cifar10',
                                                                                                       iepoch=epoch)
            info['train-loss']
            info['test-loss']
            info['test-accuracy']
            info['train-accuracy']

        train_met = arch.get_metrics('cifar10-valid', 'train')
        record['train_accuracy'] = train_met['accuracy']
        record['train_loss'] = train_met['loss']

        valid_met = arch.get_metrics('cifar10-valid', 'x-valid')
        record['valid_accuracy'] = valid_met['accuracy']
        record['valid_loss'] = valid_met['loss']

        test_met = arch.get_metrics('cifar10-valid', 'ori-test')
        record['test_accuracy'] = test_met['accuracy']
        record['test_loss'] = test_met['loss']

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

