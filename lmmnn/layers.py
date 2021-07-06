from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


class NLL(Layer):
    """Negative Log Likelihood Loss Layer"""

    def __init__(self, sig2e, sig2bs, Z_non_linear=False):
        super(NLL, self).__init__(dynamic=False)
        self.sig2e = tf.Variable(
            sig2e, name='sig2e', constraint=lambda x: tf.clip_by_value(x, 1e-5, np.infty))
        self.sig2bs = tf.Variable(
            sig2bs, name='sig2bs', constraint=lambda x: tf.clip_by_value(x, 1e-5, np.infty))
        # for k, sig2b in enumerate(sig2bs):
        #     locals()['sig2b' + str(k)] = tf.Variable(
        #         sig2b, name='sig2b' + str(k), constraint=lambda x: tf.clip_by_value(x, 1e-5, np.infty))
        self.Z_non_linear = Z_non_linear

    def get_vars(self):
        return self.sig2e.numpy(), self.sig2bs.numpy()

    def get_indices(self, N, Z_idx):
        return tf.stack([tf.range(N, dtype=tf.int64), Z_idx], axis=1)

    def getZ(self, N, Z_idx):
        if self.Z_non_linear:
            return Z_idx
        Z_idx = K.squeeze(Z_idx, axis=1)
        indices = self.get_indices(N, Z_idx)
        return tf.sparse.to_dense(tf.sparse.SparseTensor(indices, tf.ones(N), (N, tf.reduce_max(Z_idx) + 1)))

    def custom_loss(self, y_true, y_pred, Z_idxs):
        N = K.shape(y_true)[0]
        V = self.sig2e * tf.eye(N)
        for k, Z_idx in enumerate(Z_idxs):
            Z = self.getZ(N, Z_idx)
            V += self.sig2bs[k] * K.dot(Z, K.transpose(Z))
        V_inv = tf.linalg.inv(V)
        loss2 = K.dot(K.transpose(y_true - y_pred),
                      K.dot(V_inv, y_true - y_pred))
        loss1 = tf.math.log(tf.linalg.det(V))
        total_loss = 0.5 * K.cast(N, tf.float32) * \
            np.log(2 * np.pi) + 0.5 * loss1 + 0.5 * loss2
        return total_loss

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, y_true, y_pred, Z_idxs):
        self.add_loss(self.custom_loss(y_true, y_pred, Z_idxs))
        return y_pred
