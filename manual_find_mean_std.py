import pickle
import random
from tqdm import tqdm
from models.GNN import GraphAutoencoder, GraphAutoencoderNVP, weighted_mse, get_rank_weight
import tensorflow as tf
from utils.tf_utils import to_undiredted_adj


def main():
    random_seed = 0
    tf.random.set_seed(random_seed)
    random.seed(random_seed)


    model_file = 'logs/cifar10-valid/test50/model.pkl'
    datasets_file = 'logs/cifar10-valid/test50/datasets.pkl'
    with open(model_file, 'rb') as f:
        model: GraphAutoencoderNVP = pickle.load(f)
    with open(datasets_file, 'rb') as f:
        datasets = pickle.load(f)

    print('dataset', datasets['train_1'])

    tf.config.run_functions_eagerly(False)
    def get_pred_list(model, dataset):
        x = tf.stack([tf.constant(data.x) for data in dataset])
        a = tf.stack([tf.constant(data.a) for data in dataset])
        xa = (x, to_undiredted_adj(a))
        _, _, _, y_out, _ = model(xa, training=False)
        pred_list = [float(y[-1]) for y in y_out]
        return pred_list

    def check_all_smaller_than_eps(pred_list, noise_pred_list, eps):
        for i in range(len(pred_list)):
            #print(pred_list[i], noise_pred_list[i])
            if abs(pred_list[i] - noise_pred_list[i]) > eps:
                return False
        return True


    std_list = []
    weight_list = model.nvp.get_weights()
    '''
    for l in range(len(model.nvp.get_weights()) - 2):
        print('layer', l, f'/{len(weight_list)} start')
        std = 1
        shape = model.nvp.get_weights()[l].shape
        weight = np.reshape(model.nvp.get_weights()[l], -1)
        pred_list = get_pred_list(model, datasets['train_1'])
        while True:
            print('check std', std)
            flag = True
            noise = np.random.normal(loc=[0. for _ in range(weight.shape[0])],
                                         scale=[std for _ in range(weight.shape[0])],
                                         size=(weight.shape[0]))
            weight_list[l] = np.reshape(noise + weight, shape)
            model.nvp.set_weights(weight_list)
            noise_pred_list = get_pred_list(model, datasets['train_1'])
            if check_all_smaller_than_eps(pred_list, noise_pred_list, 0.01):
                std_list.append(std)
                print('found std', std)
                break
            std /= 10

        weight_list[l] = np.reshape(weight, shape)
        model.nvp.set_weights(weight_list)

    print(std_list)
    
    '''

    pred_list = get_pred_list(model, datasets['train_1'])
    # 0.00020849203426585205
    # 0.00015682785664062503
    l = 0.000156772
    r = 0.00015683
    ori_weight = model.nvp.get_weights()
    while l < r:
        std = l + (r - l) / 2
        print('check std', std)
        result_list = []
        for x in tqdm(range(1000)):
            noise_weight = []
            for layer_num in range(len(ori_weight) - 2):
                noise_weight.append(ori_weight[layer_num] + tf.random.normal(tf.shape(ori_weight[layer_num]), mean=0.0, stddev=std))
            noise_weight.append(ori_weight[-2])
            noise_weight.append(ori_weight[-1])
            model.nvp.set_weights(noise_weight)
            noise_pred_list = get_pred_list(model, datasets['train_1'])
            result_list.append(check_all_smaller_than_eps(pred_list, noise_pred_list, 0.01))
            #print(x, result_list[-1])

        print(all(result_list))
        if all(result_list):
            l = std + 1e-10
        else:
            r = std

    print(f'final std: r {r} l {l}')


if __name__ == '__main__':
    main()
