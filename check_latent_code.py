import pickle
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    filename = 'logs/50_10_192_top5_finetuneFalse_rfinetuneFalse_rankTrue_ensemble_2NN_4*5*256/cifar10-valid/20230718-203255/latent_in_each_round.pkl'
    with open(filename, 'rb') as f:
        latent_in_each_round = pickle.load(f)

    for batch in latent_in_each_round:
        dif = 0
        for i in range(batch.shape[0]):
            for j in range(batch.shape[0]):
                dif += tf.losses.mse(batch[i], batch[j])
        dif /= batch.shape[0]
        print(dif)