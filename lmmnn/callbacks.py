import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import Callback


class PrintSigmas(Callback):
    """Print sigmas at each epoch end"""
    def __init__(self):
        super(PrintSigmas, self).__init__()
        self.sig_hist = {'sig2e': [], 'sig2b': []}

    def on_epoch_end(self, epoch, logs=None):
        sig2e, sig2b = self.model.layers[-1].get_vars()
        self.sig_hist['sig2e'].append(sig2e)
        self.sig_hist['sig2b'].append(sig2b)
        if epoch > 0 and epoch % 10 == 0:
            print(' sig2e: %.2f, sig2b: %.2f' % (sig2e, sig2b))


class PrintBestLoss(Callback):
    """Print best loss so far at each epoch end"""
    def __init__(self):
        super(PrintBestLoss, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and epoch % 10 == 0:
            print(' best_loss: %.2f' % min(self.model.history.history['val_loss']))


class EarlyStoppingWithSigmasConvergence(Callback):
    def __init__(self, patience=0, auto_norm_thresh=True, norm_thresh=0.01, ma_lag=5):
        super(EarlyStoppingWithSigmasConvergence, self).__init__()
        self.patience = patience
        self.auto_norm_thresh = auto_norm_thresh
        self.norm_thresh = norm_thresh
        self.ma_lag = ma_lag
        self.sig_hist = {'sig2e': [], 'sig2b': []}
        if not auto_norm_thresh and norm_thresh is None:
            raise ValueError('If auto_norm_thresh is False you must set norm_thresh.')
    
    def record_sigmas(self):
        sig2e_est, sig2b_est = self.model.layers[-1].get_vars()
        self.sig_hist['sig2e'].append(sig2e_est)
        self.sig_hist['sig2b'].append(sig2b_est)

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss hasn't decreased.
        self.wait_loss = 0
        # The number of epoch it has waited when sigmas norm hasn't converged.
        self.wait_norm = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Record initial sigmas
        self.record_sigmas()
        # Initialize sigmas MA norm.
        self.sig2e_norm, self.sig2b_norm = self.get_sigmas_norm()
        # Initialize best loss
        self.best_loss = np.Inf

    def get_sigmas_norm(self):
        sig2e_ma = np.mean(self.sig_hist['sig2e'][-self.ma_lag:])
        sig2b_ma = np.mean(self.sig_hist['sig2b'][-self.ma_lag:])
        return sig2e_ma, sig2b_ma

    def check_stop_model(self, epoch):
        if self.wait_loss >= self.patience and self.wait_norm >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
    
    def on_epoch_end(self, epoch, logs=None):
        self.record_sigmas()
        current_loss = logs.get('val_loss')
        current_sig2e_norm, current_sig2b_norm = self.get_sigmas_norm()
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait_loss = 0
        else:
            self.wait_loss += 1
            self.check_stop_model(epoch)
        
        if self.auto_norm_thresh:
            sig2e_nt, sig2b_nt = current_sig2e_norm / 100, current_sig2b_norm / 100
        else:
            sig2e_nt, sig2b_nt = self.norm_thresh, self.norm_thresh

        if np.abs(current_sig2e_norm - self.sig2e_norm) > sig2e_nt or np.abs(current_sig2b_norm - self.sig2b_norm) > sig2b_nt:
            self.sig2e_norm, self.sig2b_norm = current_sig2e_norm, current_sig2b_norm
            self.wait_norm = 0
        else:
            self.wait_norm += 1
            self.check_stop_model(epoch)