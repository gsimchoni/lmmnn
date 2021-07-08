import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Concatenate, Reshape, Layer, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model

from lmmnn.utils import NNResult, get_dummies
from lmmnn.callbacks import EarlyStoppingWithSigmasConvergence
from lmmnn.layers import NLL
from lmmnn.menet import menet_fit, menet_predict


def add_shallow_layers_sequential(model, input_dim):
    model.add(Dense(10, input_dim=input_dim))


def add_deep_layers_sequential(model, input_dim):
    model.add(Dense(100, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(25, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(12, activation='relu'))


def add_shallow_layers_functional(X_input):
    return Dense(10)(X_input)


def add_deep_layers_functional(X_input, input_dim):
    hidden1 = Dense(100, input_dim=input_dim, activation='relu')(X_input)
    drop1 = Dropout(0.25)(hidden1)
    hidden2 = Dense(50, activation='relu')(drop1)
    drop2 = Dropout(0.25)(hidden2)
    hidden3 = Dense(25, activation='relu')(drop2)
    drop3 = Dropout(0.25)(hidden3)
    out_hidden = Dense(12, activation='relu')(drop3)
    return out_hidden


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


def calc_b_hat(X_train, y_train, y_pred_tr, qs, sig2e, sig2bs, Z_non_linear, model, ls):
    if Z_non_linear or len(qs) > 1:
        gZ_trains = []
        for k, sig2b in enumerate(sig2bs):
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
            gZ_train = csr_matrix(np.hstack(gZ_trains))
            n_cats = qs
        D_inv = get_D_est(n_cats, 1 / sig2bs)
        A = gZ_train.T @ gZ_train / sig2e + D_inv
        b_hat = np.linalg.inv(A) @ gZ_train.T / sig2e @ (y_train.values - y_pred_tr)
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
    return b_hat


def reg_nn_ohe_or_ignore(X_train, X_test, y_train, y_test, qs, x_cols, batch_size, epochs, patience, deep=False, ignore_RE=False):
    if ignore_RE:
        X_train, X_test = X_train[x_cols], X_test[x_cols]
    else:
        X_train, X_test = process_one_hot_encoding(X_train, X_test, x_cols)

    model = Sequential()
    if deep:
        add_deep_layers_sequential(model, X_train.shape[1])
    else:
        add_shallow_layers_sequential(model, X_train.shape[1])
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    callbacks = [EarlyStopping(
        monitor='val_loss', patience=epochs if patience is None else patience)]
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_split=0.1, callbacks=callbacks, verbose=0)
    y_pred = model.predict(X_test).reshape(X_test.shape[0])
    none_sigmas = [None for _ in range(len(qs))]
    return y_pred, (None, none_sigmas), len(history.history['loss'])


def reg_nn_lmm(X_train, X_test, y_train, y_test, qs, x_cols, batch_size, epochs, patience, deep=False, Z_non_linear=False, Z_embed_dim_pct=10):
    z_cols = X_train.columns[X_train.columns.str.startswith('z')].tolist()
    X_input = Input(shape=(X_train[x_cols].shape[1],))
    y_true_input = Input(shape=(1,))
    Z_inputs = []
    for _ in qs:
        Z_input = Input(shape=(1,), dtype=tf.int64)
        Z_inputs.append(Z_input)
    if deep:
        out_hidden = add_deep_layers_functional(X_input, X_train[x_cols].shape[1])
    else:
        out_hidden = add_shallow_layers_functional(X_input)
    y_pred_output = Dense(1)(out_hidden)
    if Z_non_linear:
        Z_nll_inputs = []
        ls = []
        for k, q in enumerate(qs):
            l = int(q * Z_embed_dim_pct / 100.0)
            Z_embed = Embedding(q, l, input_length=1, name='Z_embed' + str(k))(Z_inputs[k])
            Z_embed = Reshape(target_shape=(l, ))(Z_embed)
            Z_nll_inputs.append(Z_embed)
            ls.append(l)
    else:
        Z_nll_inputs = Z_inputs
        ls = None
    sig2bs_init = np.ones_like(qs, dtype=np.float32)
    nll = NLL(1.0, sig2bs_init, Z_non_linear)(y_true_input, y_pred_output, Z_nll_inputs)
    model = Model(inputs=[X_input, y_true_input] + Z_inputs, outputs=nll)

    model.compile(optimizer='adam')

    patience = epochs if patience is None else patience
    if Z_non_linear:
        callbacks = [EarlyStoppingWithSigmasConvergence(patience=patience)]
    else:
        callbacks = [EarlyStopping(patience=patience, monitor='val_loss')]
        X_train.sort_values(by=z_cols, inplace=True)
        y_train = y_train[X_train.index]
    X_train_z_cols = [X_train[z_col] for z_col in z_cols]
    X_test_z_cols = [X_test[z_col] for z_col in z_cols]
    history = model.fit([X_train[x_cols], y_train] + X_train_z_cols, None,
                        batch_size=batch_size, epochs=epochs, validation_split=0.1,
                        callbacks=callbacks, verbose=0, shuffle=False)

    sig2e_est, sig2b_ests = model.layers[-1].get_vars()
    y_pred_tr = model.predict(
        [X_train[x_cols], y_train] + X_train_z_cols).reshape(X_train.shape[0])
    b_hat = calc_b_hat(X_train, y_train, y_pred_tr, qs, sig2e_est, sig2b_ests, Z_non_linear, model, ls)
    dummy_y_test = np.random.normal(size=y_test.shape)
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
    return y_pred, (sig2e_est, list(sig2b_ests)), len(history.history['loss'])


def reg_nn_embed(X_train, X_test, y_train, y_test, qs, x_cols, batch_size, epochs, patience, deep=False):
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
    if deep:
        out_hidden = add_deep_layers_functional(concat, X_train[x_cols].shape[1] + embed_dim * len(qs))
    else:
        out_hidden = add_shallow_layers_functional(concat)
    output = Dense(1)(out_hidden)
    model = Model(inputs=[X_input] + Z_inputs, outputs=output)

    model.compile(loss='mse', optimizer='adam')

    callbacks = [EarlyStopping(
        monitor='val_loss', patience=epochs if patience is None else patience)]
    X_train_z_cols = [X_train[z_col] for z_col in X_train.columns[X_train.columns.str.startswith('z')]]
    X_test_z_cols = [X_test[z_col] for z_col in X_train.columns[X_train.columns.str.startswith('z')]]
    history = model.fit([X_train[x_cols]] + X_train_z_cols, y_train,
                        batch_size=batch_size, epochs=epochs, validation_split=0.1,
                        callbacks=callbacks, verbose=0)
    y_pred = model.predict([X_test[x_cols]] + X_test_z_cols,
                           ).reshape(X_test.shape[0])
    none_sigmas = [None for _ in range(len(qs))]
    return y_pred, (None, none_sigmas), len(history.history['loss'])


def reg_nn_menet(X_train, X_test, y_train, y_test, q, x_cols, batch_size, epochs, patience, deep=False):
    clusters_train, clusters_test = X_train['z0'].values, X_test['z0'].values
    X_train, X_test = X_train[x_cols].values, X_test[x_cols].values
    y_train, y_test = y_train.values, y_test.values

    model = Sequential()
    if deep:
        add_deep_layers_sequential(model, X_train.shape[1])
    else:
        add_shallow_layers_sequential(model, X_train.shape[1])
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    model, b_hat, sig2e_est, n_epochs, _ = menet_fit(model, X_train, y_train, clusters_train, q, batch_size, epochs, patience, verbose=False)
    y_pred = menet_predict(model, X_test, clusters_test, q, b_hat)
    return y_pred, (sig2e_est, [None]), n_epochs


def reg_nn(X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience, reg_type, deep, Z_non_linear, Z_embed_dim_pct):
    start = time.time()
    if reg_type == 'ohe':
        y_pred, sigmas, n_epochs = reg_nn_ohe_or_ignore(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience, deep)
    elif reg_type == 'lmm':
        y_pred, sigmas, n_epochs = reg_nn_lmm(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience, deep, Z_non_linear, Z_embed_dim_pct)
    elif reg_type == 'ignore':
        y_pred, sigmas, n_epochs = reg_nn_ohe_or_ignore(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience, deep, ignore_RE=True)
    elif reg_type == 'embed':
        y_pred, sigmas, n_epochs = reg_nn_embed(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience, deep)
    elif reg_type == 'menet':
        y_pred, sigmas, n_epochs = reg_nn_menet(
            X_train, X_test, y_train, y_test, qs[0], x_cols, batch, epochs, patience, deep)
    else:
        raise ValueError(reg_type + 'is an unknown reg_type')
    end = time.time()
    mse = np.mean((y_pred - y_test)**2)
    return NNResult(mse, sigmas, n_epochs, end - start)
