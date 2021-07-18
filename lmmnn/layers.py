from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


class NLL(Layer):
    """Negative Log Likelihood Loss Layer"""

    def __init__(self, mode, sig2e, sig2bs, rhos = [], est_cors = [], Z_non_linear=False):
        super(NLL, self).__init__(dynamic=False)
        self.sig2bs = tf.Variable(
            sig2bs, name='sig2bs', constraint=lambda x: tf.clip_by_value(x, 1e-5, np.infty))
        self.Z_non_linear = Z_non_linear
        self.mode = mode
        if self.mode == 'intercepts' or self.mode == 'slopes':
            self.sig2e = tf.Variable(
                sig2e, name='sig2e', constraint=lambda x: tf.clip_by_value(x, 1e-5, np.infty))
        if self.mode == 'slopes':
            self.rhos = tf.Variable(
                rhos, name='rhos', constraint=lambda x: tf.clip_by_value(x, -1.0, 1.0))
            self.est_cors = est_cors
        elif self.mode == 'glmm':
            self.nGQ = 5
            self.x_ks, self.w_ks = np.polynomial.hermite.hermgauss(self.nGQ)

    def get_vars(self):
        if self.mode == 'intercepts':
            return self.sig2e.numpy(), self.sig2bs.numpy(), []
        if self.mode == 'glmm':
            return None, self.sig2bs.numpy(), []
        return self.sig2e.numpy(), self.sig2bs.numpy(), self.rhos.numpy()

    def get_indices(self, N, Z_idx):
        return tf.stack([tf.range(N, dtype=tf.int64), Z_idx], axis=1)

    def getZ(self, N, Z_idx):
        if self.Z_non_linear:
            return Z_idx
        Z_idx = K.squeeze(Z_idx, axis=1)
        indices = self.get_indices(N, Z_idx)
        return tf.sparse.to_dense(tf.sparse.SparseTensor(indices, tf.ones(N), (N, tf.reduce_max(Z_idx) + 1)))

    def custom_loss_lm(self, y_true, y_pred, Z_idxs):
        N = K.shape(y_true)[0]
        V = self.sig2e * tf.eye(N)
        if self.mode == 'intercepts':
            for k, Z_idx in enumerate(Z_idxs):
                Z = self.getZ(N, Z_idx)
                V += self.sig2bs[k] * K.dot(Z, K.transpose(Z))
        elif self.mode == 'slopes':
            Z0 = self.getZ(N, Z_idxs[0])
            Z_list = [Z0]
            for k in range(1, len(self.sig2bs)):
                T = tf.linalg.tensor_diag(K.squeeze(Z_idxs[1], axis=1) ** k)
                Z = K.dot(T, Z0)
                Z_list.append(Z)
            for k in range(len(self.sig2bs)):
                for j in range(len(self.sig2bs)):
                    if k == j:
                        sig = self.sig2bs[k] 
                    else:
                        rho_symbol = ''.join(map(str, sorted([k, j])))
                        if rho_symbol in self.est_cors:
                            rho = self.rhos[self.est_cors.index(rho_symbol)]
                            sig = rho * tf.math.sqrt(self.sig2bs[k]) * tf.math.sqrt(self.sig2bs[j])
                        else:
                            continue
                    V += sig * K.dot(Z_list[j], K.transpose(Z_list[k]))
        V_inv = tf.linalg.inv(V)
        loss2 = K.dot(K.transpose(y_true - y_pred),
                      K.dot(V_inv, y_true - y_pred))
        loss1 = tf.math.log(tf.linalg.det(V))
        total_loss = 0.5 * K.cast(N, tf.float32) * \
            np.log(2 * np.pi) + 0.5 * loss1 + 0.5 * loss2
        return total_loss

    def custom_loss_glm(self, y_true, y_pred, Z_idxs):
        Z_idx = K.squeeze(Z_idxs[0], axis=1)
        a, _ = tf.unique(Z_idx)
        i_sum = tf.zeros(shape=(1,1))
        for i in a:
            y_i = y_true[Z_idx == i]
            f_i = y_pred[Z_idx == i]
            yf = K.dot(K.transpose(y_i), f_i)
            k_sum = tf.zeros(shape=(1,1))
            for k in range(self.nGQ):
                sqrt2_sigb_xk = np.sqrt(2) * tf.sqrt(self.sig2bs[0]) * self.x_ks[k]
                y_sum_x = K.sum(y_i) * sqrt2_sigb_xk
                log_gamma_sum = K.sum(K.log(1 + K.exp(f_i + sqrt2_sigb_xk)))
                k_sum = k_sum + K.exp(yf + y_sum_x - log_gamma_sum) * self.w_ks[k] / np.sqrt(np.pi)
            i_sum = i_sum + K.log(k_sum)
        return -i_sum
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, y_true, y_pred, Z_idxs):
        if self.mode == 'glmm':
            self.add_loss(self.custom_loss_glm(y_true, y_pred, Z_idxs))
        else:
            self.add_loss(self.custom_loss_lm(y_true, y_pred, Z_idxs))
        return y_pred
