import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from spektral.layers import GINConvBatch, GlobalSumPool, GlobalMaxPool, GlobalAvgPool, DiffPool


class Graph_Model(Model):

    def __init__(self, n_hidden, mlp_hidden, activation: str, epochs, dropout=0.):
        super(Graph_Model, self).__init__()
        self.graph_conv = GINConvBatch(n_hidden, mlp_hidden=mlp_hidden, mlp_activation=activation, mlp_batchnorm=True,
                                       activation=activation)
        self.bn = tensorflow.keras.layers.BatchNormalization()
        self.pool = GlobalMaxPool()
        self.dropout = tensorflow.keras.layers.Dropout(dropout)
        if epochs == 12:
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
