import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from tensorflow._api.v2 import random
try:
    from lifelines.utils import concordance_index
except Exception:
    pass

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Concatenate, Reshape, Layer, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras import Model

from lmmnn.utils import NNResult, get_cov_mat, get_dummies
from lmmnn.callbacks import LogEstParams, EarlyStoppingWithSigmasConvergence
from lmmnn.layers import NLL
from lmmnn.menet import menet_fit, menet_predict


def add_layers_sequential(model, n_neurons, dropout, activation, input_dim):
    if len(n_neurons) > 0:
        model.add(Dense(n_neurons[0], input_dim=input_dim, activation=activation))
        if dropout is not None and len(dropout) > 0:
            model.add(Dropout(dropout[0]))
        for i in range(1, len(n_neurons) - 1):
            model.add(Dense(n_neurons[i], activation=activation))
            if dropout is not None and len(dropout) > i:
                model.add(Dropout(dropout[i]))
        if len(n_neurons) > 1:
            model.add(Dense(n_neurons[-1], activation=activation))


def add_layers_functional(X_input, n_neurons, dropout, activation, input_dim):
    if len(n_neurons) > 0:
        x = Dense(n_neurons[0], input_dim=input_dim, activation=activation)(X_input)
        if dropout is not None and len(dropout) > 0:
            x = Dropout(dropout[0])(x)
        for i in range(1, len(n_neurons) - 1):
            x = Dense(n_neurons[i], activation=activation)(x)
            if dropout is not None and len(dropout) > i:
                x = Dropout(dropout[i])(x)
        if len(n_neurons) > 1:
            x = Dense(n_neurons[-1], activation=activation)(x)
    return x


def process_one_hot_encoding(X_train, X_test, x_cols):
    z_cols = X_train.columns[X_train.columns.str.startswith('z')]
    X_train_new = X_train[x_cols]
    X_test_new = X_test[x_cols]
    for z_col in z_cols:
        X_train_ohe = pd.get_dummies(X_train[z_col])
        X_test_ohe = pd.get_dummies(X_test[z_col])
        X_test_cols_in_train = set(X_test_ohe.columns).intersection(X_train_ohe.columns)
        X_train_cols_not_in_test = set(X_train_ohe.columns).difference(X_test_ohe.columns)
        X_test_comp = pd.DataFrame(np.zeros((X_test.shape[0], len(X_train_cols_not_in_test))),
            columns=X_train_cols_not_in_test, dtype=np.uint8, index=X_test.index)
        X_test_ohe_comp = pd.concat([X_test_ohe[X_test_cols_in_train], X_test_comp], axis=1)
        X_test_ohe_comp = X_test_ohe_comp[X_train_ohe.columns]
        X_train_ohe.columns = list(map(lambda c: z_col + '_' + str(c), X_train_ohe.columns))
        X_test_ohe_comp.columns = list(map(lambda c: z_col + '_' + str(c), X_test_ohe_comp.columns))
        X_train_new = pd.concat([X_train_new, X_train_ohe], axis=1)
        X_test_new = pd.concat([X_test_new, X_test_ohe_comp], axis=1)
    return X_train_new, X_test_new


def get_D_est(qs, sig2bs):
    D_hat = np.eye(np.sum(qs))
    np.fill_diagonal(D_hat, np.repeat(sig2bs, qs))    
    return D_hat


def calc_b_hat(X_train, y_train, y_pred_tr, qs, sig2e, sig2bs, Z_non_linear, model, ls, mode, rhos, est_cors, dist_matrix, weibull_ests):
    if mode == 'intercepts':
        if Z_non_linear or len(qs) > 1:
            gZ_trains = []
            for k in range(len(sig2bs)):
                gZ_train = get_dummies(X_train['z' + str(k)].values, qs[k])
                if Z_non_linear:
                    W_est = model.get_layer('Z_embed' + str(k)).get_weights()[0]
                    gZ_train = gZ_train @ W_est
                gZ_trains.append(gZ_train)
            if Z_non_linear:
                if X_train.shape[0] > 10000:
                    samp = np.random.choice(X_train.shape[0], 10000, replace=False)
                else:
                    samp = np.arange(X_train.shape[0])
                gZ_train = np.hstack(gZ_trains)[samp]
                n_cats = ls
            else:
                gZ_train = sparse.csr_matrix(np.hstack(gZ_trains))
                n_cats = qs
                samp = np.arange(X_train.shape[0])
            D_inv = get_D_est(n_cats, 1 / sig2bs)
            A = gZ_train.T @ gZ_train / sig2e + D_inv
            b_hat = np.linalg.inv(A) @ gZ_train.T / sig2e @ (y_train.values[samp] - y_pred_tr[samp])
            b_hat = np.asarray(b_hat).reshape(gZ_train.shape[1])
        else:
            b_hat = []
            for i in range(qs[0]):
                i_vec = X_train['z0'] == i
                n_i = i_vec.sum()
                if n_i > 0:
                    y_bar_i = y_train[i_vec].mean()
                    y_pred_i = y_pred_tr[i_vec].mean()
                    # BP(b_i) = (n_i * sig2b / (sig2a + n_i * sig2b)) * (y_bar_i - y_pred_bar_i)
                    b_i = n_i * sig2bs[0] * (y_bar_i - y_pred_i) / (sig2e + n_i * sig2bs[0])
                else:
                    b_i = 0
                b_hat.append(b_i)
            b_hat = np.array(b_hat)
    elif mode == 'slopes':
        q = qs[0]
        Z0 = sparse.csr_matrix(get_dummies(X_train['z0'], q))
        t = X_train['t'].values
        N = X_train.shape[0]
        Z_list = [Z0]
        for k in range(1, len(sig2bs)):
            Z_list.append(sparse.spdiags(t ** k, 0, N, N) @ Z0)
        gZ_train = sparse.hstack(Z_list)
        cov_mat = get_cov_mat(sig2bs, rhos, est_cors)
        D = np.kron(cov_mat, np.eye(q)) + sig2e * np.eye(q * len(sig2bs))
        D_inv = np.linalg.inv(D)
        A = gZ_train.T @ gZ_train / sig2e + D_inv
        b_hat = np.linalg.inv(A) @ gZ_train.T / sig2e @ (y_train.values - y_pred_tr)
        b_hat = np.asarray(b_hat).reshape(gZ_train.shape[1])
    elif mode == 'glmm':
        nGQ = 5
        x_ks, w_ks = np.polynomial.hermite.hermgauss(nGQ)
        a = np.unique(X_train['z0'])
        b_hat_numerators = []
        b_hat_denominators = []
        q = qs[0]
        for i in range(q):
            if i in a:
                i_vec = X_train['z0'] == i
                y_i = y_train.values[i_vec]
                f_i = y_pred_tr[i_vec]
                yf = np.dot(y_i, f_i)
                k_sum_num = 0
                k_sum_den = 0
                for k in range(nGQ):
                    sqrt2_sigb_xk = np.sqrt(2) * np.sqrt(sig2bs[0]) * x_ks[k]
                    y_sum_x = y_i.sum() * sqrt2_sigb_xk
                    log_gamma_sum = np.sum(np.log(1 + np.exp(f_i + sqrt2_sigb_xk)))
                    k_exp = np.exp(yf + y_sum_x - log_gamma_sum) * w_ks[k] / np.sqrt(np.pi)
                    k_sum_num = k_sum_num + sqrt2_sigb_xk * k_exp
                    k_sum_den = k_sum_den + k_exp
                b_hat_numerators.append(k_sum_num)
                if k_sum_den == 0.0:
                    b_hat_denominators.append(1)
                else:
                    b_hat_denominators.append(k_sum_den)
            else:
                b_hat_numerators.append(0)
                b_hat_denominators.append(1)
        b_hat = np.array(b_hat_numerators) / np.array(b_hat_denominators)
    elif mode == 'spatial':
        gZ_train = get_dummies(X_train['z0'].values, qs[0])
        gZ_train = sparse.csr_matrix(gZ_train)
        D = sig2bs[0] * np.exp(-dist_matrix / (2 * sig2bs[1]))
        N = gZ_train.shape[0]
        if X_train.shape[0] > 10000:
            samp = np.random.choice(X_train.shape[0], 10000, replace=False)
        else:
            samp = np.arange(X_train.shape[0])
        gZ_train = gZ_train[samp]
        V = gZ_train @ D @ gZ_train.T + np.eye(gZ_train.shape[0]) * sig2e
        V_inv_y = np.linalg.inv(V) @ (y_train.values[samp] - y_pred_tr[samp])
        b_hat = D @ gZ_train.T @ V_inv_y
        # A = gZ_train.T @ gZ_train / sig2e + D_inv
        # A_inv_Zt = np.linalg.inv(A) @ gZ_train.T
        # b_hat = A_inv_Zt / sig2e @ (y_train.values[samp] - y_pred_tr[samp])
        # b_hat = np.asarray(b_hat).reshape(gZ_train.shape[1])
    elif mode == 'spatial_embedded':
        loc_df = X_train[['D1', 'D2']]
        last_layer = Model(inputs = model.input[2], outputs = model.layers[-2].output)
        gZ_train = last_layer.predict([loc_df])
        if X_train.shape[0] > 10000:
            samp = np.random.choice(X_train.shape[0], 10000, replace=False)
        else:
            samp = np.arange(X_train.shape[0])
        gZ_train = gZ_train[samp]
        n_cats = ls
        D_inv = get_D_est(n_cats, 1 / sig2bs)
        A = gZ_train.T @ gZ_train / sig2e + D_inv
        b_hat = np.linalg.inv(A) @ gZ_train.T / sig2e @ (y_train.values[samp] - y_pred_tr[samp])
        b_hat = np.asarray(b_hat).reshape(gZ_train.shape[1])
    elif mode == 'survival':
        Hs = weibull_ests[0] * (y_train ** weibull_ests[1])
        b_hat = []
        for i in range(qs[0]):
            i_vec = X_train['z0'] == i
            D_i = X_train['C0'][i_vec].sum()
            A_i = 1 / sig2bs[0] + D_i
            C_i = 1 / sig2bs[0] + np.sum(Hs[i_vec] * np.exp(y_pred_tr[i_vec]))
            b_i = A_i / C_i
            b_hat.append(b_i)
        b_hat = np.array(b_hat)
    return b_hat


def reg_nn_ohe_or_ignore(X_train, X_test, y_train, y_test, qs, x_cols, batch_size, epochs,
        patience, n_neurons, dropout, activation, mode, n_sig2bs, est_cors, verbose=False, ignore_RE=False):
    if mode == 'glmm':
        loss = 'binary_crossentropy'
        last_layer_activation = 'sigmoid'
    else:
        loss = 'mse'
        last_layer_activation = 'linear'
    if ignore_RE:
        X_train, X_test = X_train[x_cols], X_test[x_cols]
    else:
        X_train, X_test = process_one_hot_encoding(X_train, X_test, x_cols)

    model = Sequential()
    add_layers_sequential(model, n_neurons, dropout, activation, X_train.shape[1])
    model.add(Dense(1, activation=last_layer_activation))

    model.compile(loss=loss, optimizer='adam')

    callbacks = [EarlyStopping(
        monitor='val_loss', patience=epochs if patience is None else patience)]
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_split=0.1, callbacks=callbacks, verbose=verbose)
    y_pred = model.predict(X_test).reshape(X_test.shape[0])
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_rhos = [None for _ in range(len(est_cors))]
    none_weibull = [None, None]
    return y_pred, (None, none_sigmas), none_rhos, none_weibull, len(history.history['loss'])


def reg_nn_lmm(X_train, X_test, y_train, y_test, qs, x_cols, batch_size, epochs, patience, n_neurons, dropout, activation,
        mode, n_sig2bs, est_cors, dist_matrix, spatial_embed_neurons, verbose=False, Z_non_linear=False, Z_embed_dim_pct=10,
        log_params=False, idx=0):
    if mode == 'spatial' or mode == 'spatial_embedded':
        x_cols = [x_col for x_col in x_cols if x_col not in ['D1', 'D2']]
    if mode == 'survival':
        x_cols = [x_col for x_col in x_cols if x_col not in ['C0']]
    # dmatrix_tf = tf.constant(dist_matrix)
    dmatrix_tf = dist_matrix
    X_input = Input(shape=(X_train[x_cols].shape[1],))
    y_true_input = Input(shape=(1,))
    if mode in ['intercepts', 'glmm', 'spatial']:
        z_cols = X_train.columns[X_train.columns.str.startswith('z')].tolist()
        Z_inputs = []
        n_RE_inputs = len(qs)
        if mode == 'spatial':
            n_sig2bs_init = n_sig2bs
        else:
            n_sig2bs_init = len(qs)
        for _ in range(n_RE_inputs):
            Z_input = Input(shape=(1,), dtype=tf.int64)
            Z_inputs.append(Z_input)
    elif mode == 'slopes':
        z_cols = ['z0', 't']
        n_RE_inputs = 2
        n_sig2bs_init = n_sig2bs
        Z_input = Input(shape=(1,), dtype=tf.int64)
        t_input = Input(shape=(1,))
        Z_inputs = [Z_input, t_input]
    elif mode == 'spatial_embedded':
        Z_inputs = [Input(shape=(2,))]
        n_sig2bs_init = 1
    elif mode == 'survival':
        z_cols = ['z0', 'C0']
        Z_input = Input(shape=(1,), dtype=tf.int64)
        event_input = Input(shape=(1,))
        Z_inputs = [Z_input, event_input]
        n_sig2bs_init = 1
    
    out_hidden = add_layers_functional(X_input, n_neurons, dropout, activation, X_train[x_cols].shape[1])
    y_pred_output = Dense(1)(out_hidden)
    if Z_non_linear and (mode in ['intercepts', 'glmm', 'survival']):
        Z_nll_inputs = []
        ls = []
        for k, q in enumerate(qs):
            l = int(q * Z_embed_dim_pct / 100.0)
            Z_embed = Embedding(q, l, input_length=1, name='Z_embed' + str(k))(Z_inputs[k])
            Z_embed = Reshape(target_shape=(l, ))(Z_embed)
            Z_nll_inputs.append(Z_embed)
            ls.append(l)
    elif mode == 'spatial_embedded':
        Z_embed = add_layers_functional(Z_inputs[0], spatial_embed_neurons, dropout=None, activation='relu', input_dim=2)
        Z_nll_inputs = [Z_embed]
        ls = [spatial_embed_neurons[-1]]
        Z_non_linear = True
    else:
        Z_nll_inputs = Z_inputs
        ls = None
    sig2bs_init = np.ones(n_sig2bs_init, dtype=np.float32)
    rhos_init = np.zeros(len(est_cors), dtype=np.float32)
    weibull_init = np.ones(2, dtype=np.float32)
    nll = NLL(mode, 1.0, sig2bs_init, rhos_init, weibull_init, est_cors, Z_non_linear, dmatrix_tf)(
        y_true_input, y_pred_output, Z_nll_inputs)
    model = Model(inputs=[X_input, y_true_input] + Z_inputs, outputs=nll)

    model.compile(optimizer='adam')

    patience = epochs if patience is None else patience
    if Z_non_linear and mode == 'intercepts':
        # in complex scenarios such as non-linear g(Z) consider training "more", until var components norm has converged
        callbacks = [EarlyStoppingWithSigmasConvergence(patience=patience)]
        # callbacks = [EarlyStopping(patience=patience, monitor='val_loss')]
    else:
        callbacks = [EarlyStopping(patience=patience, monitor='val_loss')]
    if log_params:
        callbacks.extend([LogEstParams(idx), CSVLogger('res_params.csv', append=True)])
    if not Z_non_linear:
        X_train.sort_values(by=z_cols, inplace=True)
        y_train = y_train[X_train.index]
    if mode == 'spatial_embedded':
        X_train_z_cols = [X_train[['D1', 'D2']]]
        X_test_z_cols = [X_test[['D1', 'D2']]]
    else:
        X_train_z_cols = [X_train[z_col] for z_col in z_cols]
        X_test_z_cols = [X_test[z_col] for z_col in z_cols]
    history = model.fit([X_train[x_cols], y_train] + X_train_z_cols, None,
                        batch_size=batch_size, epochs=epochs, validation_split=0.1,
                        callbacks=callbacks, verbose=verbose, shuffle=False)

    sig2e_est, sig2b_ests, rho_ests, weibull_ests = model.layers[-1].get_vars()
    y_pred_tr = model.predict(
        [X_train[x_cols], y_train] + X_train_z_cols).reshape(X_train.shape[0])
    b_hat = calc_b_hat(X_train, y_train, y_pred_tr, qs, sig2e_est, sig2b_ests,
                Z_non_linear, model, ls, mode, rho_ests, est_cors, dist_matrix, weibull_ests)
    dummy_y_test = np.random.normal(size=y_test.shape)
    if mode in ['intercepts', 'glmm', 'spatial']:
        if Z_non_linear or len(qs) > 1:
            Z_tests = []
            for k, q in enumerate(qs):
                Z_test = get_dummies(X_test['z' + str(k)], q)
                if Z_non_linear:
                    W_est = model.get_layer('Z_embed' + str(k)).get_weights()[0]
                    Z_test = Z_test @ W_est
                Z_tests.append(Z_test)
            Z_test = np.hstack(Z_tests)
            y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols).reshape(
                X_test.shape[0]) + Z_test @ b_hat
        else:
            y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols).reshape(
                X_test.shape[0]) + b_hat[X_test['z0']]
        if mode == 'glmm':
            y_pred = np.exp(y_pred)/(1 + np.exp(y_pred))
    elif mode == 'slopes':
        q = qs[0]
        Z0 = sparse.csr_matrix(get_dummies(X_test['z0'], q))
        t = X_test['t'].values
        N = X_test.shape[0]
        Z_list = [Z0]
        for k in range(1, len(sig2b_ests)):
            Z_list.append(sparse.spdiags(t ** k, 0, N, N) @ Z0)
        Z_test = sparse.hstack(Z_list)
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols).reshape(
                X_test.shape[0]) + Z_test @ b_hat
    elif mode == 'spatial_embedded':
        last_layer = Model(inputs = model.input[2], outputs = model.layers[-2].output)
        gZ_test = last_layer.predict(X_test_z_cols)
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols).reshape(
                X_test.shape[0]) + gZ_test @ b_hat
        sig2b_ests = np.concatenate([sig2b_ests, [np.nan]])
    elif mode == 'survival':
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols).reshape(
                X_test.shape[0])
        y_pred = y_pred + np.log(b_hat[X_test['z0']])
    return y_pred, (sig2e_est, list(sig2b_ests)), list(rho_ests), list(weibull_ests), len(history.history['loss'])


def reg_nn_embed(X_train, X_test, y_train, y_test, qs, x_cols, batch_size, epochs, patience,
        n_neurons, dropout, activation, mode, n_sig2bs, est_cors, verbose=False):
    if mode == 'glmm':
        loss = 'binary_crossentropy'
        last_layer_activation = 'sigmoid'
    else:
        loss = 'mse'
        last_layer_activation = 'linear'
    embed_dim = 10

    X_input = Input(shape=(X_train[x_cols].shape[1],))
    Z_inputs = []
    embeds = []
    for q in qs:
        Z_input = Input(shape=(1,))
        embed = Embedding(q, embed_dim, input_length=1)(Z_input)
        embed = Reshape(target_shape=(embed_dim,))(embed)
        Z_inputs.append(Z_input)
        embeds.append(embed)
    concat = Concatenate()([X_input] + embeds)
    out_hidden = add_layers_functional(concat, n_neurons, dropout, activation, X_train[x_cols].shape[1] + embed_dim * len(qs))
    output = Dense(1, activation=last_layer_activation)(out_hidden)
    model = Model(inputs=[X_input] + Z_inputs, outputs=output)

    model.compile(loss=loss, optimizer='adam')

    callbacks = [EarlyStopping(
        monitor='val_loss', patience=epochs if patience is None else patience)]
    X_train_z_cols = [X_train[z_col] for z_col in X_train.columns[X_train.columns.str.startswith('z')]]
    X_test_z_cols = [X_test[z_col] for z_col in X_train.columns[X_train.columns.str.startswith('z')]]
    history = model.fit([X_train[x_cols]] + X_train_z_cols, y_train,
                        batch_size=batch_size, epochs=epochs, validation_split=0.1,
                        callbacks=callbacks, verbose=verbose)
    y_pred = model.predict([X_test[x_cols]] + X_test_z_cols,
                           ).reshape(X_test.shape[0])
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_rhos = [None for _ in range(len(est_cors))]
    none_weibull = [None, None]
    return y_pred, (None, none_sigmas), none_rhos, none_weibull, len(history.history['loss'])


def reg_nn_menet(X_train, X_test, y_train, y_test, q, x_cols, batch_size, epochs, patience,
        n_neurons, dropout, activation, n_sig2bs, est_cors, verbose=False):
    clusters_train, clusters_test = X_train['z0'].values, X_test['z0'].values
    X_train, X_test = X_train[x_cols].values, X_test[x_cols].values
    y_train, y_test = y_train.values, y_test.values

    model = Sequential()
    add_layers_sequential(model, n_neurons, dropout, activation, X_train.shape[1])
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    model, b_hat, sig2e_est, n_epochs, _ = menet_fit(model, X_train, y_train, clusters_train, q, batch_size, epochs,
                                                patience, verbose=verbose)
    y_pred = menet_predict(model, X_test, clusters_test, q, b_hat)
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_rhos = [None for _ in range(len(est_cors))]
    none_weibull = [None, None]
    return y_pred, (sig2e_est, none_sigmas), none_rhos, none_weibull, n_epochs


def reg_nn(X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience, n_neurons, dropout, activation, reg_type,
        Z_non_linear, Z_embed_dim_pct, mode, n_sig2bs, est_cors, dist_matrix, spatial_embed_neurons, verbose, log_params, idx):
    start = time.time()
    if reg_type == 'ohe':
        y_pred, sigmas, rhos, weibull, n_epochs = reg_nn_ohe_or_ignore(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, n_sig2bs, est_cors, verbose)
    elif reg_type == 'lmm':
        y_pred, sigmas, rhos, weibull, n_epochs = reg_nn_lmm(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode,
            n_sig2bs, est_cors, dist_matrix, spatial_embed_neurons, verbose, Z_non_linear, Z_embed_dim_pct, log_params, idx)
    elif reg_type == 'ignore':
        y_pred, sigmas, rhos, weibull, n_epochs = reg_nn_ohe_or_ignore(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, n_sig2bs, est_cors, verbose, ignore_RE=True)
    elif reg_type == 'embed':
        y_pred, sigmas, rhos, weibull, n_epochs = reg_nn_embed(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, n_sig2bs, est_cors, verbose)
    elif reg_type == 'menet':
        y_pred, sigmas, rhos, weibull, n_epochs = reg_nn_menet(
            X_train, X_test, y_train, y_test, qs[0], x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, n_sig2bs, est_cors, verbose)
    else:
        raise ValueError(reg_type + 'is an unknown reg_type')
    end = time.time()
    if mode == 'glmm':
        metric = roc_auc_score(y_test, y_pred)
    elif mode == 'survival':
        if np.any(np.isnan(y_pred)):
            metric = np.nan
        else:
            metric = concordance_index(y_test, -y_pred, X_test['C0'])
    else:
        metric = np.mean((y_pred - y_test)**2)
    return NNResult(metric, sigmas, rhos, weibull, n_epochs, end - start)
