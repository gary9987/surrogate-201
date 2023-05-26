import numpy as np
from trainGAE_ensemble import main
import tensorflow as tf
import gc
import os
from datetime import datetime
import pickle


if __name__ == '__main__':
    train_sample_list = [50]
    valid_sample_list = [10]
    budget_list = [192]
    dataset_name = 'nb101'
    now_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = now_time + dataset_name +'_experiments'
    os.makedirs(logdir, exist_ok=True)

    for train_sample, valid_sample, budget in zip(train_sample_list, valid_sample_list, budget_list):
        best_acc_list = []
        best_test_acc_list = []
        record_list = []
        for i in range(10):
            best_acc, best_test_acc, record = main(seed=i, dataset_name=dataset_name,
                                                    train_sample_amount=train_sample,
                                                    valid_sample_amount=valid_sample,
                                                    query_budget=budget)
            best_acc_list.append(best_acc)
            best_test_acc_list.append(best_test_acc)
            record_list.append(record)
            tf.keras.backend.clear_session()
            gc.collect()

        print(best_acc_list)
        print(f'best_valid_acc avg: {sum(best_acc_list) / len(best_acc_list)}, std: {np.std(best_acc_list)}')
        print(f'max valid acc: {max(best_acc_list)}')
        print(best_test_acc_list)
        print(f'best_test_acc avg: {sum(best_test_acc_list) / len(best_test_acc_list)}, std: {np.std(best_test_acc_list)}')
        print(f'max test acc: {max(best_test_acc_list)}')

        with open(os.path.join(logdir, f'{dataset_name}_{train_sample}_{valid_sample}_{budget}.txt'), 'w') as f:
            f.write(f'best_valid_acc avg: {sum(best_acc_list) / len(best_acc_list)}, std: {np.std(best_acc_list)}\n')
            f.write(f'max valid acc: {max(best_acc_list)}\n')
            f.write(f'best_test_acc avg: {sum(best_test_acc_list) / len(best_test_acc_list)}, std: {np.std(best_test_acc_list)}\n')
            f.write(f'max test acc: {max(best_test_acc_list)}\n')
        
        with open(os.path.join(logdir, f'{dataset_name}_{train_sample}_{valid_sample}_{budget}_record.pkl'), 'wb') as f:
            pickle.dump(record_list, f)


