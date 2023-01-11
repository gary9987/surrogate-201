from keras.callbacks import EarlyStopping, CSVLogger
from transformation import ReshapeYTransform
from model import Graph_Model
import tensorflow as tf
import logging
import sys
from nb201_dataset import NasBench201Dataset, train_valid_test_split_dataset
from spektral.data import BatchLoader
from metrics import get_avg_kt, get_avg_r2, get_final_epoch_kt, get_final_epoch_r2


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
    label_epochs = 200
    batch_size = 16
    train_epochs = 150
    patience = 20

    model = Graph_Model(n_hidden, mlp_hidden, model_activation, label_epochs, model_dropout)

    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(label_epochs), seed=777),
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

    partition = ['train', 'valid', 'test']
    pred_dict = {i: [] for i in partition}
    label_dict = {i: [] for i in partition}

    for data in test_loader:
        preds = model.predict(data[0])
        for label, pred in zip(data[1], preds):

            for key, ep in zip(partition, range(0, 35, 12)):
                # logging.info(f'\n{i[ep: ep+12]}\n{j[ep: ep+12]}')
                pred_dict[key].append(label[ep: ep + 12])
                label_dict[key].append(pred[ep: ep + 12])

    kt_list = []
    final_kt_list = []
    r2_list = []
    final_r2_list = []

    for key in partition:
        kt, _ = get_avg_kt(pred_dict[key], label_dict[key])
        final_kt, _ = get_final_epoch_kt(pred_dict[key], label_dict[key])
        r2 = get_avg_r2(pred_dict[key], label_dict[key])
        final_r2 = get_final_epoch_r2(pred_dict[key], label_dict[key])
        kt_list.append(kt)
        final_kt_list.append(final_kt)
        r2_list.append(r2)
        final_r2_list.append(final_r2)
        logging.info(f'{key} avg KT: {kt}')
        logging.info(f'{key} final KT: {final_kt}')
        logging.info(f'{key} avg r2: {r2}')
        logging.info(f'{key} final r2: {final_r2}')

    logging.info(f'Avg of {partition} avg KT: {sum(kt_list)/len(kt_list)}')
    logging.info(f'Avg of {partition} final KT: {sum(final_kt_list) / len(final_kt_list)}')
    logging.info(f'Avg of {partition} avg r2: {sum(r2_list) / len(r2_list)}')
    logging.info(f'Avg of {partition} final r2: {sum(final_r2_list) / len(final_r2_list)}')