import tensorflow as tf
import os


class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, every_epoch=50):
        super(SaveModelCallback, self).__init__()
        self.every_epoch = every_epoch
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_epoch == 0:
            self.model.save_weights(os.path.join(self.save_dir, 'model_{:04d}.ckpt'.format(epoch + 1)))


def to_undiredted_adj(adj):
    undirected_adj = tf.cast(tf.cast(adj, tf.int32) | tf.cast(tf.transpose(adj, perm=[0, 2, 1]), tf.int32), tf.float32)
    return undirected_adj