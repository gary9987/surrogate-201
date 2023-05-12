import pickle
import matplotlib.pyplot as plt
from math import sqrt


if __name__ == '__main__':

    #filename = 'logs/cifar10-valid/aggregate_log/log.pkl'
    filename = 'logs/cifar10-valid/20230509-150626/log_0.pkl'
    with open(filename, 'rb') as f:
        log = pickle.load(f)

    log['train_size_list'] = log['train_size_list'][:-1]
    log['bound_list'] = log['bound_list'][:-1]
    print(log)
    iter = [i for i in range(len(log['train_size_list']))]
    x_data = log['train_size_list']
    #x_data = iter

    #log['bound_list'] = [sqrt(bound) for bound in log['bound_list']]
    plt.plot(x_data, log['bound_list'], label='Bound^2')
    plt.plot(x_data, log['mse_list_valid'], label='mse_valid')
    plt.ylabel('Bound^2')
    plt.legend()
    plt.xlabel('Train size')
    plt.show()

    log['bound_list'] = [sqrt(bound) for bound in log['bound_list']]
    plt.plot(x_data, log['bound_list'], label='Bound')
    plt.xlabel('Train size')
    plt.ylabel('Bound')
    plt.show()

    plt.plot(x_data, log['best_list'])
    plt.xlabel('Train size')
    plt.ylabel('Best accuracy')
    plt.show()

    #plt.plot(x_data, log['mse_list_train'], label='mse_train')
    plt.plot(x_data, log['mse_list_valid'], label='mse_valid')
    plt.xlabel('Train size')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
