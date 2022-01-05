import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import Callback


class LogEstParams(Callback):
    def __init__(self, idx):
        super(LogEstParams, self).__init__()
        self.idx = idx

    def on_epoch_end(self, epoch, logs):
        sig2e_est, sig2bs_est, rhos_est, weibull_est = self.model.layers[-1].get_vars()
        logs['experiment'] = self.idx
        logs['sig2e_est'] = sig2e_est
        for k, sig2b_est in enumerate(sig2bs_est):
            logs['sig2b_est' + str(k)] = sig2b_est
        for k, rho_est in enumerate(rhos_est):
            logs['rho_est' + str(k)] = rho_est
        for k, weibull_est in enumerate(weibull_est):
            logs['weibull_est' + str(k)] = weibull_est


class PrintSigmas(Callback):
    """Print sigmas at each epoch end"""
    def __init__(self, print_steps = 10):
        super(PrintSigmas, self).__init__()
        self.sig_hist = {'sig2e': [], 'sig2b': []}
        self.print_steps = print_steps

    def on_epoch_end(self, epoch, logs=None):
        sig2e, sig2bs, _ = self.model.layers[-1].get_vars()
        self.sig_hist['sig2e'].append(sig2e)
        self.sig_hist['sig2b'].append(sig2bs[0])
        if (epoch + 1) % self.print_steps == 0:
            print(' sig2e: %.2f, sig2b: %.2f' % (sig2e, sig2bs[0]))


class PrintBestLoss(Callback):
    """Print best loss so far at each epoch end"""
    def __init__(self):
        super(PrintBestLoss, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and epoch % 10 == 0:
            print(' best_loss: %.2f' % min(self.model.history.history['val_loss']))


class EarlyStoppingWithSigmasConvergence(Callback):
    def __init__(self, patience=0, auto_ma_thresh=True, ma_thresh=0.01, ma_lag=10):
        super(EarlyStoppingWithSigmasConvergence, self).__init__()
        self.patience = patience
        self.auto_ma_thresh = auto_ma_thresh
        self.ma_thresh = ma_thresh
        self.ma_lag = ma_lag
        self.sqrt_lag = np.sqrt(self.ma_lag)
        self.sig_hist = {'sig2e': [], 'sig2b': []}
        if not auto_ma_thresh and ma_thresh is None:
            raise ValueError('If auto_ma_thresh is False you must set ma_thresh.')
    
    def record_sigmas(self):
        sig2e_est, sig2b_ests, _, _ = self.model.layers[-1].get_vars()
        if sig2e_est is None:
            sig2e_est = 0
        self.sig_hist['sig2e'].append(sig2e_est)
        self.sig_hist['sig2b'].append(sig2b_ests.sum())

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss hasn't decreased.
        self.wait_loss = 0
        # The number of epoch it has waited when sigmas MA hasn't converged.
        self.wait_ma = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Record initial sigmas
        self.record_sigmas()
        # Initialize sigmas MA.
        self.sig2e_ma, _, self.sig2b_ma, _ = self.get_sigmas_ma_sd()
        # Initialize best loss
        self.best_loss = np.Inf

    def get_sigmas_ma_sd(self):
        previous_sig2e = self.sig_hist['sig2e'][-self.ma_lag:]
        previous_sig2b = self.sig_hist['sig2b'][-self.ma_lag:]
        sig2e_ma, sig2e_sd = np.mean(previous_sig2e), np.std(previous_sig2e)
        sig2b_ma, sig2b_sd = np.mean(previous_sig2b), np.std(previous_sig2b)
        return sig2e_ma, sig2e_sd, sig2b_ma, sig2b_sd

    def check_stop_model(self, epoch):
        if self.wait_loss >= self.patience and self.wait_ma >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
    
    def on_epoch_end(self, epoch, logs=None):
        self.record_sigmas()
        current_loss = logs.get('val_loss')
        current_sig2e_ma, current_sig2e_sd, current_sig2b_ma, current_sig2b_sd = self.get_sigmas_ma_sd()
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait_loss = 0
        else:
            self.wait_loss += 1
            self.check_stop_model(epoch)
        
        if self.auto_ma_thresh:
            sig2e_nt, sig2b_nt = 2 * current_sig2e_sd / self.sqrt_lag, 2 * current_sig2b_sd / self.sqrt_lag
            sig2e_nt, sig2b_nt = np.max([sig2e_nt, 0.01]), np.max([sig2b_nt, 0.01])
        else:
            sig2e_nt, sig2b_nt = self.ma_thresh, self.ma_thresh

        if np.abs(current_sig2e_ma - self.sig2e_ma) > sig2e_nt or np.abs(current_sig2b_ma - self.sig2b_ma) > sig2b_nt:
            self.sig2e_ma, self.sig2b_ma = current_sig2e_ma, current_sig2b_ma
            self.wait_ma = 0
        else:
            self.wait_ma += 1
            self.check_stop_model(epoch)