import argparse
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tqdm import tqdm
from datasets.transformation import ReshapeYTransform, OnlyValidAccTransform
from models.TransformerAE import TransformerAutoencoderReg, CustomSchedule
import tensorflow as tf
import logging
import sys
from datasets.nb201_dataset import NasBench201Dataset
from datasets.utils_data import train_valid_test_split_dataset
import numpy as np


logging.basicConfig(filename='train.log', level=logging.INFO, force=True, filemode='w')
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


parser = argparse.ArgumentParser(description='train TAE')
parser.add_argument('--train_sample_amount', type=int, default=900, help='Number of samples to train (default: 900)')
parser.add_argument('--valid_sample_amount', type=int, default=100, help='Number of samples to train (default: 100)')
args = parser.parse_args()


random_seed = 0
tf.random.set_seed(random_seed)

def to_np_data(graph_dataset):
    features = []
    y_list = []
    for data in graph_dataset:
        x = np.reshape(data.x, -1)
        a = np.reshape(data.a, -1)
        features.append(np.concatenate([x, a]))
        y_list.append([data.y[-1] / 100.0])

    return np.array(features), np.array(y_list)


def train(model, train_dataset, rec_loss_fn, reg_loss_fn, optimizer, alpha=0.5):
    total_rec_loss = 0
    total_reg_loss = 0
    total_loss = 0

    # Iterate over the batches of the dataset.
    for (x_batch_train, y_batch_train) in tqdm(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            rec_logits, reg_logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            rec_loss = rec_loss_fn(x_batch_train, rec_logits)
            reg_loss = reg_loss_fn(y_batch_train, reg_logits)
            loss_value = alpha * rec_loss + (1 - alpha) * reg_loss

            total_loss += rec_loss + reg_loss
            total_rec_loss += rec_loss
            total_reg_loss += reg_loss

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))


    print("Training rec_loss: %.4f, reg_loss: %.4f, total_loss: %.4f" % (float(total_rec_loss / len(train_dataset)),
                                                                       float(total_reg_loss / len(train_dataset)),
                                                                       float(total_loss / len(train_dataset))))


def eval(model, test_dataset, rec_loss_fn, reg_loss_fn):
    total_rec_loss = 0
    total_reg_loss = 0

    # Iterate over the batches of the dataset.
    for (x, y) in tqdm(test_dataset):
        rec_logits, reg_logits = model(x, training=False)  # Logits for this minibatch

        # Compute the loss value for this minibatch.
        rec_loss = rec_loss_fn(x, rec_logits)
        reg_loss = reg_loss_fn(y, reg_logits)

        total_rec_loss += rec_loss
        total_reg_loss += reg_loss

    print("Testing rec_loss: %.4f, reg_loss: %.4f" % (float(total_rec_loss / len(test_dataset)),
                                                      float(total_reg_loss / len(test_dataset))))


if __name__ == '__main__':
    is_only_validation_data = True
    label_epochs = 200

    d_model=64
    dff = 128
    num_layers = 3
    num_heads = 3

    batch_size = 32
    train_epochs = 300
    patience = 20

    # 15624
    datasets = train_valid_test_split_dataset(NasBench201Dataset(start=0, end=15624, hp=str(label_epochs), seed=777),
                                              ratio=[0.9, 0.1],
                                              shuffle=True,
                                              shuffle_seed=0)

    datasets['train'] = datasets['train'][:args.train_sample_amount]
    datasets['valid'] = datasets['valid'][:args.valid_sample_amount]

    for key in datasets:
        if is_only_validation_data:
            datasets[key].apply(OnlyValidAccTransform())
        else:
            datasets[key].apply(ReshapeYTransform())

    x_train, y_train = to_np_data(datasets['train'])
    x_valid, y_valid = to_np_data(datasets['valid'])

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    rec_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)
    reg_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

    model = TransformerAutoencoderReg(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, input_size=x_train.shape[-1], dropout_rate=0)


    loader = {'train': tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(batch_size=batch_size),
              'valid': tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size=batch_size)}


    for epoch in range(train_epochs):
        print('Epoch: {}'.format(epoch))
        train(model, loader['train'], rec_loss_fn, reg_loss_fn, optimizer)

    eval(model, loader['valid'], rec_loss_fn, reg_loss_fn)
