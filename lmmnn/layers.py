from os import name
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


class NLL(Layer):
    """Negative Log Likelihood Loss Layer"""

    def __init__(self, mode, sig2e, sig2bs, rhos = [], weibull_init = [], est_cors = [], Z_non_linear=False, dist_matrix=None):
        super(NLL, self).__init__(dynamic=False)
        self.sig2bs = tf.Variable(
            sig2bs, name='sig2bs', constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))
        self.Z_non_linear = Z_non_linear
        self.mode = mode
        if self.mode in ['intercepts', 'slopes', 'spatial', 'spatial_embedded', 'spatial_and_categoricals']:
            self.sig2e = tf.Variable(
                sig2e, name='sig2e', constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))
            if self.mode in ['spatial', 'spatial_and_categoricals']:
                self.dist_matrix = dist_matrix
                self.max_loc = dist_matrix.shape[1] - 1
                self.spatial_delta = int(0.0 * dist_matrix.shape[1])
        if self.mode == 'slopes':
            if len(est_cors) > 0:
                self.rhos = tf.Variable(
                    rhos, name='rhos', constraint=lambda x: tf.clip_by_value(x, -1.0, 1.0))
            self.est_cors = est_cors
        if self.mode == 'glmm':
            self.nGQ = 5
            self.x_ks, self.w_ks = np.polynomial.hermite.hermgauss(self.nGQ)
        if self.mode == 'survival':
            self.weibull_lambda = tf.Variable(
                weibull_init[0], name='weibull_lambda', constraint=lambda x: tf.clip_by_value(x, 1e-5, np.infty))
            self.weibull_nu = tf.Variable(
                weibull_init[1], name='weibull_nu', constraint=lambda x: tf.clip_by_value(x, 1e-5, np.infty))

    def get_vars(self):
        if self.mode in ['intercepts', 'spatial', 'spatial_embedded', 'spatial_and_categoricals']:
            return self.sig2e.numpy(), self.sig2bs.numpy(), [], []
        if self.mode == 'glmm':
            return None, self.sig2bs.numpy(), [], []
        if self.mode == 'survival':
            return None, self.sig2bs.numpy(), [], [self.weibull_lambda.numpy(), self.weibull_nu.numpy()]
        if hasattr(self, 'rhos'):
            return self.sig2e.numpy(), self.sig2bs.numpy(), self.rhos.numpy(), []
        else:
            return self.sig2e.numpy(), self.sig2bs.numpy(), [], []

    def get_table(self, Z_idx):
        Z_unique, _ = tf.unique(Z_idx)
        Z_mapto = tf.range(tf.shape(Z_unique)[0], dtype=tf.int64)
        table = tf.lookup.StaticVocabularyTable(
                tf.lookup.KeyValueTensorInitializer(
                    Z_unique,
                    Z_mapto,
                    key_dtype=tf.int64,
                    value_dtype=tf.int64,
                ),
                num_oov_buckets=1,
            )
        return table
    
    def get_indices(self, N, Z_idx, min_Z):
        return tf.stack([tf.range(N, dtype=tf.int64), Z_idx - min_Z], axis=1)

    def get_indices_v1(self, N, Z_idx):
        return tf.stack([tf.range(N, dtype=tf.int64), Z_idx], axis=1)

    def getZ(self, N, Z_idx, min_Z, max_Z):
        if self.Z_non_linear:
            return Z_idx
        Z_idx = K.squeeze(Z_idx, axis=1)
        indices = self.get_indices(N, Z_idx, min_Z)
        return tf.sparse.to_dense(tf.sparse.SparseTensor(indices, tf.ones(N), (N, max_Z - min_Z + 1)))
    
    def getZ_v1(self, N, Z_idx):
        if self.Z_non_linear:
            return Z_idx
        Z_idx = K.squeeze(Z_idx, axis=1)
        indices = self.get_indices_v1(N, Z_idx)
        return tf.sparse.to_dense(tf.sparse.SparseTensor(indices, tf.ones(N), (N, tf.reduce_max(Z_idx) + 1)))

    def getD(self, min_Z, max_Z):
        a = tf.range(min_Z, max_Z + 1)
        d = tf.shape(a)[0]
        ix_ = tf.reshape(tf.stack([tf.repeat(a, d), tf.tile(a, [d])], 1), [d, d, 2])
        M = tf.gather_nd(self.dist_matrix, ix_)
        M = tf.cast(M, tf.float32)
        D = self.sig2bs[0] * tf.math.exp(-M / (2 * self.sig2bs[1]))
        return D
    
    def custom_loss_lm(self, y_true, y_pred, Z_idxs):
        N = K.shape(y_true)[0]
        V = self.sig2e * tf.eye(N)
        if self.mode in ['intercepts', 'spatial_embedded', 'spatial_and_categoricals']:
            categoricals_loc = 0
            if self.mode == 'spatial_and_categoricals':
                categoricals_loc = 1
            for k, Z_idx in enumerate(Z_idxs[categoricals_loc:]):
                min_Z = tf.reduce_min(Z_idx)
                max_Z = tf.reduce_max(Z_idx)
                Z = self.getZ(N, Z_idx, min_Z, max_Z)
                # Z = self.getZ_v1(N, Z_idx)
                sig2bs_loc = k
                if self.mode == 'spatial_and_categoricals': # first 2 sig2bs go to kernel
                    sig2bs_loc += 2
                V += self.sig2bs[sig2bs_loc] * K.dot(Z, K.transpose(Z))
        if self.mode == 'slopes':
            min_Z = tf.reduce_min(Z_idxs[0])
            max_Z = tf.reduce_max(Z_idxs[0])
            Z0 = self.getZ(N, Z_idxs[0], min_Z, max_Z)
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
        if self.mode in ['spatial', 'spatial_and_categoricals']:
            # for expanded kernel experiments
            # min_Z = tf.maximum(tf.reduce_min(Z_idxs[0]) - self.spatial_delta, 0)
            # max_Z = tf.minimum(tf.reduce_max(Z_idxs[0]) + self.spatial_delta, self.max_loc)
            min_Z = tf.reduce_min(Z_idxs[0])
            max_Z = tf.reduce_max(Z_idxs[0])
            D = self.getD(min_Z, max_Z)
            Z = self.getZ(N, Z_idxs[0], min_Z, max_Z)
            V += K.dot(Z, K.dot(D, K.transpose(Z)))
        if self.Z_non_linear:
            V_inv = tf.linalg.inv(V)
            V_inv_y = K.dot(V_inv, y_true - y_pred)
        else:
            V_inv_y = tf.linalg.solve(V, y_true - y_pred)
        loss2 = K.dot(K.transpose(y_true - y_pred), V_inv_y)
        # loss1 = tf.math.log(tf.linalg.det(V))
        _, loss1 = tf.linalg.slogdet(V)
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
    
    def custom_loss_survival(self, y_true, y_pred, Z_idxs):
        N = K.shape(y_true)[0]
        min_Z = tf.reduce_min(Z_idxs[0])
        max_Z = tf.reduce_max(Z_idxs[0])
        Z = self.getZ(N, Z_idxs[0], min_Z, max_Z)
        event = Z_idxs[1]
        Z_idx = K.squeeze(Z_idxs[0], axis=1)
        event_sums = tf.math.segment_sum(event, Z_idx - min_Z)
        Hs = self.weibull_lambda * tf.math.pow(y_true, self.weibull_nu)
        hs = self.weibull_lambda * self.weibull_nu * tf.math.pow(y_true, self.weibull_nu - 1)
        sum_exps = K.dot(K.transpose(Z), tf.multiply(Hs, tf.math.exp(y_pred)))
        l1 = tf.reduce_sum(event_sums * tf.math.log(self.sig2bs[0]) - tf.math.lgamma(1 / self.sig2bs[0]) + tf.math.lgamma(1 / self.sig2bs[0] + event_sums))
        l2 = tf.reduce_sum(tf.multiply(-(1 / self.sig2bs[0] + event_sums), tf.math.log(sum_exps * self.sig2bs[0] + 1)))
        l3 = tf.reduce_sum(K.dot(K.transpose(Z), (y_pred + tf.math.log(hs)) * event))
        return -(l1 + l2 + l3)
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, y_true, y_pred, Z_idxs):
        if self.mode == 'glmm':
            self.add_loss(self.custom_loss_glm(y_true, y_pred, Z_idxs))
        elif self.mode == 'survival':
            self.add_loss(self.custom_loss_survival(y_true, y_pred, Z_idxs))
        else:
            self.add_loss(self.custom_loss_lm(y_true, y_pred, Z_idxs))
        return y_pred
