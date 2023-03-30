import numpy as np
import tensorflow as tf
from datasets.query_nb201 import OPS_by_IDX_201
from models.TransformerAE import TransformerAutoencoderNVP
from nats_bench import create

random_seed = 0
np.random.seed(random_seed)
tf.random.set_seed(random_seed)


def inverse_from_acc(model: tf.keras.Model, num_sample_z: int, z_dim: int, to_inv_acc: float):
    num_ops = 7
    num_nodes = 8
    y = np.array([to_inv_acc] * num_sample_z).reshape((num_sample_z, -1))  # (num_sample_z, 1)
    z = np.random.multivariate_normal([1.] * z_dim, np.eye(z_dim), num_sample_z)  # (num_sample_z, z_dim)
    y = np.concatenate([z, y], axis=-1).astype(np.float32)  # (num_sample_z, x_dim)

    rev_latent = model.inverse(y)  # (num_sample_z, x_dim)
    rev_x = model.decode(tf.reshape(rev_latent, (num_sample_z, -1, model.d_model)))  # (num_sample_z, input_size(120))

    ops_vote = tf.reduce_sum(rev_x[:, :num_ops * num_nodes], axis=0).numpy()  # 7 ops 8 nodes
    adj = tf.where(tf.reduce_mean(rev_x[:, num_ops * num_nodes:], axis=0) >= 0.5, x=1., y=0.).numpy()  # (1, 8 * 8)
    adj = np.reshape(adj, (int(adj.shape[-1]**(1/2)), int(adj.shape[-1]**(1/2))))
    ops_idx = []
    for i in range(num_nodes):
        ops_idx.append(np.argmax(ops_vote[i * num_ops: (i + 1) * num_ops], axis=-1))

    return ops_idx, adj

def ops_list_to_nb201_arch_str(ops):
    # partial code from: https://github.com/jovitalukasik/SVGe/blob/main/datasets/NASBench201.py#L239
    steps_coding = ['0', '0', '1', '0', '1', '2']

    node_1 = '|' + ops[1] + '~' + steps_coding[0] + '|'
    node_2 = '|' + ops[2] + '~' + steps_coding[1] + '|' + ops[3] + '~' + steps_coding[2] + '|'
    node_3 = '|' + ops[4] + '~' + steps_coding[3] + '|' + ops[5] + '~' + steps_coding[4] + '|' + ops[
        6] + '~' + steps_coding[5] + '|'
    nodes_nb201 = node_1 + '+' + node_2 + '+' + node_3

    return nodes_nb201


if __name__ == '__main__':
    d_model = 4
    dropout_rate = 0.0
    dff = 512
    num_layers = 3
    num_heads = 3
    input_size = 120
    nvp_config = {
        'n_couple_layer': 3,
        'n_hid_layer': 3,
        'n_hid_dim': 128,
        'name': 'NVP'
    }
    model = TransformerAutoencoderNVP(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                      input_size=input_size, nvp_config=nvp_config)

    model.load_weights('modelTAE_weights')

    to_inv_acc = 0.9999

    ops_idx, adj = inverse_from_acc(model, num_sample_z=10000, z_dim=120 * d_model-1, to_inv_acc=to_inv_acc)
    ops = [OPS_by_IDX_201[i] for i in ops_idx]
    print(ops)
    print(adj)

    arch_str = ops_list_to_nb201_arch_str(ops)
    print(arch_str)

    nb201api = create(None, 'tss', fast_mode=True, verbose=True)
    idx = nb201api.query_index_by_arch(arch_str)
    for seed in [777, 888]:
        data = nb201api.get_more_info(idx, 'cifar10-valid', iepoch=199, hp='200', is_random=seed)
        print(data['valid-accuracy'])

        data = nb201api.get_more_info(idx, 'cifar10', iepoch=199, hp='200', is_random=seed)
        print(data['test-accuracy'])


