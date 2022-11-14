from nats_bench import create

if __name__ == '__main__':
    api = create(None, 'tss', fast_mode=True, verbose=True)
    architecture_str = api.arch(12)
    print(architecture_str)

    lc = []
    for i in range(12):
        info = api.get_more_info(1, 'cifar10', iepoch=i)
        validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(1, dataset='cifar10', iepoch=i)
        info['train-loss']
        info['test-loss']
        info['test-accuracy']
        info['train-accuracy']
        lc.append(validation_accuracy)

    print(lc)

