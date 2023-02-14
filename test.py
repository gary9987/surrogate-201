import os
import keras
from transformation import ReshapeYTransform
import logging
from nb201_dataset import NasBench201Dataset, train_valid_test_split_dataset
from spektral.data import BatchLoader
from metrics import get_avg_kt, get_avg_r2, get_final_epoch_kt, get_final_epoch_r2

log_filename = 'test.log'
if os.path.exists(log_filename):
    os.remove(log_filename)

logging.basicConfig(filename=log_filename, level=logging.INFO, force=True)

if __name__ == '__main__':

    model = keras.models.load_model('model')
    hp = '200'
    # hp=12 end=15624
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=hp, seed=777),
                                              ratio=[0.8, 0.1, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    for key in datasets:
        datasets[key].apply(ReshapeYTransform())

    test_loader = BatchLoader(datasets['test'], batch_size=128, shuffle=False, epochs=1)
    loss = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)
    logging.info(f'MSE loss: {loss}')

    model.compile('adam', 'mae')
    test_loader = BatchLoader(datasets['test'], batch_size=128, shuffle=False, epochs=1)
    loss = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)
    logging.info(f'MAE loss: {loss}')


    test_loader = BatchLoader(datasets['test'], batch_size=128, shuffle=False, epochs=1)

    partition = ['train', 'valid', 'test']
    pred_dict = {i: [] for i in partition}
    label_dict = {i: [] for i in partition}

    for data in test_loader:
        preds = model.predict(data[0])
        for label, pred in zip(data[1], preds):

            for key, ep in zip(partition, range(0, 35, 12)):
                #logging.info(f'\n{i[ep: ep+12]}\n{j[ep: ep+12]}')
                pred_dict[key].append(label[ep: ep + 12])
                label_dict[key].append(pred[ep: ep+12])

    for key in partition:
        kt, _ = get_avg_kt(pred_dict[key], label_dict[key])
        final_kt, _ = get_final_epoch_kt(pred_dict[key], label_dict[key])
        r2 = get_avg_r2(pred_dict[key], label_dict[key])
        final_r2 = get_final_epoch_r2(pred_dict[key], label_dict[key])
        logging.info(f'{key} avg KT: {kt}')
        logging.info(f'{key} final KT: {final_kt}')
        logging.info(f'{key} avg r2: {r2}')
        logging.info(f'{key} final r2: {final_r2}')
