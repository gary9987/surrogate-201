from keras.callbacks import EarlyStopping, CSVLogger
from transformation import ReshapeYTransform, OnlyValidAccTransform
from model import Graph_Model
import tensorflow as tf
import logging
import sys
from datasets.nb201_dataset import NasBench201Dataset, train_valid_test_split_dataset
from spektral.data import BatchLoader
from metrics import get_avg_kt, get_avg_r2, get_final_epoch_kt, get_final_epoch_r2


logging.basicConfig(filename='train.log', level=logging.INFO, force=True, filemode='w')
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


'''
mae
n_hidden = 256 * 64
2023-02-17 02:47:14,167 - __main__ - INFO - Test loss: 3.1751654148101807
2023-02-17 02:47:15,362 - __main__ - INFO - valid avg KT: 0.720009150446175
2023-02-17 02:47:15,362 - __main__ - INFO - valid final KT: 0.818033679082239
2023-02-17 02:47:15,362 - __main__ - INFO - valid avg r2: 0.8894060437871442
2023-02-17 02:47:15,362 - __main__ - INFO - valid final r2: 0.8888513897650342

mae
n_hidden = 256 * 128
2023-02-17 12:13:43,230 - __main__ - INFO - Test loss: 3.0976991653442383
2023-02-17 12:13:44,430 - __main__ - INFO - valid avg KT: 0.7270281661162866
2023-02-17 12:13:44,430 - __main__ - INFO - valid final KT: 0.834623220828212
2023-02-17 12:13:44,430 - __main__ - INFO - valid avg r2: 0.8981137365661757
2023-02-17 12:13:44,430 - __main__ - INFO - valid final r2: 0.9273346105833765

n_hidden = 256 * 96
98/98 [==============================] - 0s 2ms/step - loss: 18.6336
2023-02-16 14:25:58,505 - __main__ - INFO - Test loss: 18.633615493774414
2023-02-16 14:26:01,556 - __main__ - INFO - valid avg KT: 0.7117642347113109
2023-02-16 14:26:01,556 - __main__ - INFO - valid final KT: 0.7964802185411368
2023-02-16 14:26:01,556 - __main__ - INFO - valid avg r2: 0.9240731107323961
2023-02-16 14:26:01,556 - __main__ - INFO - valid final r2: 0.9873605223246051

n_hidden = 256 * 32
98/98 [==============================] - 0s 2ms/step - loss: 19.0196
2023-02-16 02:18:34,430 - __main__ - INFO - Test loss: 19.01958656311035
2023-02-16 02:18:37,522 - __main__ - INFO - valid avg KT: 0.706678500194417
2023-02-16 02:18:37,523 - __main__ - INFO - valid final KT: 0.7763533566947862
2023-02-16 02:18:37,523 - __main__ - INFO - valid avg r2: 0.9223269308711685
2023-02-16 02:18:37,523 - __main__ - INFO - valid final r2: 0.9821228606288467
2023-02-16 02:18:37,523 - __main__ - INFO - Avg of ['valid'] avg KT: 0.706678500194417
2023-02-16 02:18:37,523 - __main__ - INFO - Avg of ['valid'] final KT: 0.7763533566947862
2023-02-16 02:18:37,523 - __main__ - INFO - Avg of ['valid'] avg r2: 0.9223269308711685
2023-02-16 02:18:37,523 - __main__ - INFO - Avg of ['valid'] final r2: 0.9821228606288467
'''

tf.random.set_seed(777)

if __name__ == '__main__':
    is_only_validation_data = True
    mlp_hidden = [64, 64, 64, 64]
    model_activation = 'relu'
    model_dropout = 0.1
    label_epochs = 200
    #n_hidden = 256 * 12
    n_hidden = 256 * 128
    batch_size = 64
    train_epochs = 150
    patience = 20

    model = Graph_Model(n_hidden, mlp_hidden, model_activation, label_epochs, model_dropout, is_only_validation_data)

    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(label_epochs), seed=777),
                                              ratio=[0.8, 0.1, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    for key in datasets:
        if is_only_validation_data:
            datasets[key].apply(OnlyValidAccTransform())
        else:
            datasets[key].apply(ReshapeYTransform())


    model.compile('adam', 'mae')

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

    partition = ['train', 'valid']
    if is_only_validation_data:
        partition = ['valid']
    elif label_epochs == 12:
        partition.append('test')

    pred_dict = {i: [] for i in partition}
    label_dict = {i: [] for i in partition}

    for data in test_loader:
        preds = model.predict(data[0])
        for label, pred in zip(data[1], preds):

            for key, ep in zip(partition, range(0, label_epochs * 3 - 1, label_epochs)):
                # logging.info(f'\n{i[ep: ep+12]}\n{j[ep: ep+12]}')
                pred_dict[key].append(pred[ep: ep + label_epochs])
                label_dict[key].append(label[ep: ep + label_epochs])

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
        logger.info(f'{key} avg KT: {kt}')
        logger.info(f'{key} final KT: {final_kt}')
        logger.info(f'{key} avg r2: {r2}')
        logger.info(f'{key} final r2: {final_r2}')

    logger.info(f'Avg of {partition} avg KT: {sum(kt_list)/len(kt_list)}')
    logger.info(f'Avg of {partition} final KT: {sum(final_kt_list) / len(final_kt_list)}')
    logger.info(f'Avg of {partition} avg r2: {sum(r2_list) / len(r2_list)}')
    logger.info(f'Avg of {partition} final r2: {sum(final_r2_list) / len(final_r2_list)}')