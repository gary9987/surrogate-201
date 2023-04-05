import argparse
from keras.callbacks import EarlyStopping, CSVLogger
from datasets.transformation import ReshapeYTransform, OnlyValidAccTransform
from models.GNN import Graph_Model, bpr_loss
import tensorflow as tf
import logging
import sys
from datasets.nb201_dataset import NasBench201Dataset
from datasets.utils import train_valid_test_split_dataset
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
n_hidden = 256 * 32
mae
2023-02-18 15:37:28,767 - __main__ - INFO - valid avg KT: 0.7374774040702665
2023-02-18 15:37:28,767 - __main__ - INFO - valid final KT: 0.8485214035621704
2023-02-18 15:37:28,767 - __main__ - INFO - valid avg r2: 0.9082094481908604
2023-02-18 15:37:28,767 - __main__ - INFO - valid final r2: 0.9412924297748704
mse
023-02-18 16:01:45,576 - __main__ - INFO - Test loss: 20.011558532714844
2023-02-18 16:01:46,632 - __main__ - INFO - valid avg KT: 0.7281491125656102
2023-02-18 16:01:46,632 - __main__ - INFO - valid final KT: 0.8070790871274783
2023-02-18 16:01:46,632 - __main__ - INFO - valid avg r2: 0.9175857863487716
2023-02-18 16:01:46,632 - __main__ - INFO - valid final r2: 0.9655637727304788

n_hidden = 256 * 64
mae
2023-02-18 15:09:46,856 - __main__ - INFO - valid avg KT: 0.7404437502248676
2023-02-18 15:09:46,856 - __main__ - INFO - valid final KT: 0.8501183796364733
2023-02-18 15:09:46,856 - __main__ - INFO - valid avg r2: 0.9091128082621964
2023-02-18 15:09:46,856 - __main__ - INFO - valid final r2: 0.9365387193297008
mse
2023-02-18 16:16:00,580 - __main__ - INFO - valid avg KT: 0.7195785664790253
2023-02-18 16:16:00,580 - __main__ - INFO - valid final KT: 0.7901186903768046
2023-02-18 16:16:00,580 - __main__ - INFO - valid avg r2: 0.9171485771807407
2023-02-18 16:16:00,581 - __main__ - INFO - valid final r2: 0.9669182222014336

n_hidden = 256 * 128
mae
2023-02-18 15:02:27,791 - __main__ - INFO - Test loss: 2.8402063846588135
2023-02-18 15:02:28,874 - __main__ - INFO - valid avg KT: 0.728318550718445
2023-02-18 15:02:28,874 - __main__ - INFO - valid final KT: 0.8319821470949826
2023-02-18 15:02:28,874 - __main__ - INFO - valid avg r2: 0.9072266789625367
2023-02-18 15:02:28,874 - __main__ - INFO - valid final r2: 0.9308830909549481
mse
2023-02-18 16:28:51,590 - __main__ - INFO - valid avg KT: 0.7181505848911058
2023-02-18 16:28:51,590 - __main__ - INFO - valid final KT: 0.792515369205859
2023-02-18 16:28:51,590 - __main__ - INFO - valid avg r2: 0.9177202080362106
2023-02-18 16:28:51,590 - __main__ - INFO - valid final r2: 0.9707941779490004

'''

parser = argparse.ArgumentParser(description='train GIN')
parser.add_argument('--train_sample_amount', type=int, default=900, help='Number of samples to train (default: 900)')
parser.add_argument('--valid_sample_amount', type=int, default=100, help='Number of samples to train (default: 100)')
parser.add_argument('--criterion', type=str, default='mse', help='loss function for training (mse, bpr, mae)')
args = parser.parse_args()


tf.random.set_seed(777)

if __name__ == '__main__':
    is_only_validation_data = True
    mlp_hidden = [64, 64, 64, 64]
    model_activation = 'relu'
    model_dropout = 0.1
    label_epochs = 200
    #n_hidden = 256 * 12
    n_hidden = 256 * 64
    batch_size = 64
    train_epochs = 150
    patience = 20

    model = Graph_Model(n_hidden, mlp_hidden, model_activation, label_epochs, model_dropout, is_only_validation_data)

    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(label_epochs), seed=777),
                                              ratio=[0.8, 0.1, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    datasets['train'] = datasets['train'][:args.train_sample_amount]
    datasets['valid'] = datasets['valid'][:args.valid_sample_amount]

    for key in datasets:
        if is_only_validation_data:
            datasets[key].apply(OnlyValidAccTransform())
        else:
            datasets[key].apply(ReshapeYTransform())


    if args.criterion == 'mse':
        criterion = tf.keras.losses.MeanSquaredError()
    elif args.criterion == 'mae':
        criterion = tf.keras.losses.MeanAbsoluteError()
    elif args.criterion == 'bpr':
        criterion = bpr_loss
    else:
        raise ValueError(f'args.criterion {args.criterion} is not supported')

    model.compile(tf.keras.optimizers.Adam(), loss=criterion)

    loader = {key: BatchLoader(datasets[key], batch_size=batch_size, shuffle=True if key != 'test' else False) for key in datasets}

    model.fit(loader['train'].load(), steps_per_epoch=loader['train'].steps_per_epoch,
              validation_data=loader['valid'].load(), validation_steps=loader['valid'].steps_per_epoch,
              epochs=train_epochs,
              callbacks=[EarlyStopping(patience=patience, restore_best_weights=True),
                         CSVLogger(f"learning_curve.log")]
              )
    model.save(f'model_{args.criterion}_train{args.train_sample_amount}_valid{args.valid_sample_amount}')

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