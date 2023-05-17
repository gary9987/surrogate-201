import numpy as np
from trainGAE_ensemble import main


if __name__ == '__main__':
    invalid_list = []
    avg_acc_list = []
    best_acc_list = []
    avg_test_acc_list = []
    best_test_acc_list = []
    for i in range(10):
        avg_acc, best_acc, avg_test_acc, best_test_acc = main(seed=i, dataset_name='cifar10-valid', train_sample_amount=250, valid_sample_amount=50, query_budget=400)
        avg_acc_list.append(avg_acc)
        best_acc_list.append(best_acc)
        avg_test_acc_list.append(avg_test_acc)
        best_test_acc_list.append(best_test_acc)

    print(avg_acc_list)
    print(best_acc_list)
    print(f'avg_valid_acc avg: {sum(avg_acc_list) / len(avg_acc_list)}, std: {np.std(avg_acc_list)}')
    print(f'best_valid_acc avg: {sum(best_acc_list) / len(best_acc_list)}, std: {np.std(best_acc_list)}')
    print(f'max valid acc: {max(best_acc_list)}')
    print(avg_test_acc_list)
    print(best_test_acc_list)
    print(f'avg_test_acc avg: {sum(avg_test_acc_list) / len(avg_test_acc_list)}, std: {np.std(avg_test_acc_list)}')
    print(f'best_test_acc avg: {sum(best_test_acc_list) / len(best_test_acc_list)}, std: {np.std(best_test_acc_list)}')
    print(f'max test acc: {max(best_test_acc_list)}')

