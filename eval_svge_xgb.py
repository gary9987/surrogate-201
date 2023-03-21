import os.path
import pickle
from pathlib import Path
import logging
from scipy.stats import kendalltau
from xgboost import XGBRegressor
from utils.test_metric import randon_select_data, mAP
from sklearn.metrics import ndcg_score, mean_squared_error, mean_absolute_error
import numpy as np
from utils.util import evaluate_metrics


def eval_svge_xgb(log_dir, weight_path, test_dataset, model):
    if not os.path.exists(log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_path = os.path.join(log_dir, f'{Path(weight_path).name}_test.log')

    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(filename=log_path, level=logging.INFO, force=True)

    # model = XGBRegressor()
    # model.load_model(weight_path)

    pred = model.predict(test_dataset['x'])
    mse = mean_squared_error(test_dataset['y'], pred)
    print('Test MSE loss: {}'.format(mse))
    logging.info('Test MSE loss: {}'.format(mse))

    mae = mean_absolute_error(test_dataset['y'], pred)
    print('Test MAE loss: {}'.format(mae))
    logging.info('Test MAE loss: {}'.format(mae))

    metrics = evaluate_metrics(test_dataset['y'], pred, prediction_is_first_arg=False)
    print(metrics)
    logging.info(f'{metrics}')
    return metrics


if __name__ == '__main__':
    model = XGBRegressor()
    model.load_model('ensemble_model/xgb_size500/xgb_size500_0')
    datasets = {}
    with open('Experiments/Surrogate/NB201/SVGE/NB201/2023_03_21_14_24_44_Train_PP/NB201_test.pkl', 'rb') as f:
        data = pickle.load(f)
        datasets['x'] = np.array(data['encodings'])
        datasets['y'] = np.array(data['labels'])

    eval_svge_xgb('ensemble_model_log', 'xgb_size500_0', datasets, model)
