import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from spektral.layers import GINConvBatch, GlobalSumPool, GlobalMaxPool, GlobalAvgPool, DiffPool
import tensorflow as tf


class Graph_Model(Model):

    def __init__(self, n_hidden, mlp_hidden, activation: str, epochs, dropout=0., is_only_validation_data=False):
        super(Graph_Model, self).__init__()
        self.graph_conv = GINConvBatch(n_hidden, mlp_hidden=mlp_hidden, mlp_activation=activation, mlp_batchnorm=True,
                                       activation=activation)
        self.bn = tensorflow.keras.layers.BatchNormalization()
        self.pool = GlobalMaxPool()
        self.dropout = tensorflow.keras.layers.Dropout(dropout)
        if is_only_validation_data:
            self.dense = Dense(epochs)
        elif epochs == 12:
            self.dense = Dense(3 * epochs)  # (train_acc, valid_acc, test_acc) * 12 epochs
        elif epochs == 200:
            self.dense = Dense(2 * epochs)  # (train_acc, valid_acc) * 200 epochs
        else:
            raise NotImplementedError('epochs')

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.bn(out)
        out = self.pool(out)
        out = self.dropout(out)
        out = self.dense(out)
        return out


def bpr_loss(y_true, y_pred):
    total_loss = []

    N = y_true.shape[0]  # y_true.shape[0] = batch size
    lc_length = y_true.shape[1]

    for i in range(lc_length):
        loss_value = 0
        for j in range(N):
            loss_value += tf.reduce_sum(tf.keras.backend.switch(y_true[:, i] > y_true[j, i],
                                                                -tf.math.log(tf.sigmoid(y_pred[:, i] - y_pred[j, i])),
                                                                0))
        total_loss.append(loss_value)

    total_loss = tf.stack(total_loss, 0)

    return total_loss / N ** 2