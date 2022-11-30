import os.path

import keras
from transformation import ReshapeYTransform
import logging
from nb201_dataset import NasBench201Dataset, train_valid_test_split_dataset
from spektral.data import BatchLoader

log_filename = 'test.log'
if os.path.exists(log_filename):
    os.remove(log_filename)

logging.basicConfig(filename=log_filename, level=logging.INFO, force=True)

if __name__ == '__main__':

    model = keras.models.load_model('model')
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624),
                                              ratio=[0.8, 0.1, 0.1])
    for key in datasets:
        datasets[key].apply(ReshapeYTransform())

    test_loader = BatchLoader(datasets['test'], batch_size=128, shuffle=False, epochs=1)
    for data in test_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            logging.info('\n')
            for ep in range(0, 35, 12):
                logging.info(f'\n{i[ep: ep+12]}\n{j[ep: ep+12]}')
