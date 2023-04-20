import numpy as np
from trainGAE_two_phase import main


if __name__ == '__main__':
    invalid_list = []
    avg_acc_list = []
    best_acc_list = []
    for i in range(10):
        avg_acc, best_acc = main(seed=i, train_sample_amount=350, valid_sample_amount=50)
        avg_acc_list.append(avg_acc)
        best_acc_list.append(best_acc)

    print(avg_acc_list)
    print(best_acc_list)
    print('avg_acc avg: %f, std: %f', sum(avg_acc_list) / len(avg_acc_list), np.std(avg_acc_list))
    print('best_acc avg: %f, std: %f', sum(best_acc_list) / len(best_acc_list), np.std(best_acc_list))
