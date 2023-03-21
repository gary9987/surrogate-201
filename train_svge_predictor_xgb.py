import os.path
import pickle
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
import logging
from eval_svge_xgb import eval_svge_xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import kendalltau


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_output_dir', type=str, default='ensemble_model')
    parser.add_argument('--train_dataset_path', type=str,
                        default='Experiments/Surrogate/NB201/SVGE_acc/NB201/2023_03_21_16_43_06_Train_PP/NB201_train.pkl')
    parser.add_argument('--test_dataset_path', type=str,
                        default='Experiments/Surrogate/NB201/SVGE_acc/NB201/2023_03_21_16_43_06_Train_PP/NB201_test.pkl')
    return parser.parse_args()


args = parse_args()


def train(model_output_dir, run: int, data_size: int):
    hp = {
        'n_estimators': 20000,
        'max_depth': 13,
        'min_child_weight': 39,
        'colsample_bylevel': 0.6909,
        'colsample_bytree': 0.2545,
        'reg_lambda': 31.3933,
        'reg_alpha': 0.2417,
        'learning_rate': 0.00824,
        'booster': 'gbtree',
        'early_stopping_rounds': 100,
        'random_state': 0,
        'objective': 'reg:squarederror',
        'eval_metric': ['rmse', 'mae'],
        'tree_method': 'gpu_hist'  # GPU
    }

    model = XGBRegressor(**hp)

    weight_file_dir = f'xgb_size{data_size}'
    weight_filename = f'{weight_file_dir}_{run}'

    Path(os.path.join(model_output_dir, weight_file_dir)).mkdir(parents=True, exist_ok=True)
    weight_full_name = os.path.join(model_output_dir, weight_file_dir, weight_filename)

    print(weight_full_name)

    log_dir = f'{model_output_dir}_log'
    Path(os.path.join(log_dir, 'valid_log')).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, 'valid_log', f'{weight_filename}.log'), level=logging.INFO, force=True, filemode='w')
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    random.seed(hp['random_state'])
    datasets = {'train': {}, 'test': {}}
    with open(args.train_dataset_path, 'rb') as f:
        data = pickle.load(f)
        ids = random.sample(range(0, len(data['encodings'])), k=data_size)
        datasets['train']['x'] = np.array(data['encodings'])[ids]
        datasets['train']['y'] = np.array(data['labels'])[ids]
        #datasets['train']['y'] = datasets['train']['y'][:, -1]
    with open(args.test_dataset_path, 'rb') as f:
        data = pickle.load(f)
        datasets['test']['x'] = np.array(data['encodings'])
        datasets['test']['y'] = np.array(data['labels'])
        #datasets['test']['y'] = datasets['test']['y'][:, -1]

    x_train, x_valid, y_train, y_valid = train_test_split(datasets['train']['x'], datasets['train']['y'], test_size=0.1, random_state=hp['random_state'])

    model.fit(X=x_train, y=y_train, eval_set=[(x_valid, y_valid)])

    logging.info(f'Model will save to {weight_full_name}')
    model.save_model(weight_full_name)

    pred = model.predict(datasets['test']['x'])
    loss = mean_squared_error(datasets['test']['y'], pred)
    logging.info('Test MSE: {}'.format(loss))

    return eval_svge_xgb(os.path.join(log_dir, 'test_result'), weight_full_name, datasets['test'], model)


def train_n_runs(model_output_dir: str, n: int, data_size: int):
    metrics = ['mse', 'rmse', 'avg KT', 'final KT', 'avg r2', 'final r2']
    results = {i: [] for i in metrics}

    for i in range(n):
        # {'MSE': mse, 'MAE': mae, 'KT': kt, 'P': p}
        metrics = train(model_output_dir, i, data_size)
        print(metrics)
        for m in metrics:
            results[m].append(metrics[m])

    logger = logging.getLogger()

    for key in results:
        logger.info(f'{key} mean: {sum(results[key])/len(results[key])}')
        logger.info(f'{key} min: {min(results[key])}')
        logger.info(f'{key} max: {max(results[key])}')
        logger.info(f'{key} std: {np.std(results[key])}')


if __name__ == '__main__':
    Path(args.model_output_dir).mkdir(exist_ok=True)
    #train(args.model_output_dir, 0, 1000)
    range_list = [
        [1000, 10501, 500],
        [11500, 20501, 1000],
        [25500, 170501, 5000]
    ]
    for r in range_list:
        for i in range(r[0], r[1], r[2]):
            train_n_runs(args.model_output_dir, n=5, data_size=i)
