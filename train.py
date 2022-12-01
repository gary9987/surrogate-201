from keras.callbacks import EarlyStopping, CSVLogger
from transformation import ReshapeYTransform
from model import Graph_Model
import tensorflow as tf
import logging
import sys
from nb201_dataset import NasBench201Dataset, train_valid_test_split_dataset
from spektral.data import BatchLoader


logging.basicConfig(filename='train.log', level=logging.INFO, force=True, filemode='w')
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


if __name__ == '__main__':
    n_hidden = 256 * 12
    mlp_hidden = [64, 64, 64, 64]
    model_activation = 'relu'
    model_dropout = 0.1
    label_epochs = 12
    batch_size = 16
    train_epochs = 150
    patience = 20

    model = Graph_Model(n_hidden, mlp_hidden, model_activation, label_epochs, model_dropout)

    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624),
                                              ratio=[0.8, 0.1, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    for key in datasets:
        datasets[key].apply(ReshapeYTransform())

    model.compile('adam', 'mse')

    loader = {key: BatchLoader(datasets[key], batch_size=batch_size, shuffle=True if key != 'test' else False) for key in datasets}
    model.fit(loader['train'].load(), steps_per_epoch=loader['train'].steps_per_epoch,
              validation_data=loader['valid'].load(), validation_steps=loader['valid'].steps_per_epoch,
              epochs=train_epochs,
              callbacks=[EarlyStopping(patience=patience, restore_best_weights=True),
                         CSVLogger(f"learning_curve.log")]
              )
    model.save('model')

    logger.info(f'{model.summary()}')
    loss = model.evaluate(loader['test'].load(), steps=loader['test'].steps_per_epoch)
    logger.info('Test loss: {}'.format(loss))

    test_loader = BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)

    cot = 0
    for data in test_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            logger.info(f'{cot}')
            cot += 1
            for ep in range(0, 35, 12):
                logger.info(f'\n{i[ep: ep + 12]}\n{j[ep: ep + 12]}')
