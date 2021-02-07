import time
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Concatenate, Reshape, Layer, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model

from lmmnn.utils import NNResult
from lmmnn.callbacks import EarlyStoppingWithSigmasConvergence
from lmmnn.layers import NLL


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


def add_deep_layers_functional(X_input):
    hidden1 = Dense(100, activation='relu')(X_input)
    drop1 = Dropout(0.25)(hidden1)
    hidden2 = Dense(50, activation='relu')(drop1)
    drop2 = Dropout(0.25)(hidden2)
    hidden3 = Dense(25, activation='relu')(drop2)
    drop3 = Dropout(0.25)(hidden3)
    out_hidden = Dense(12, activation='relu')(drop3)
    return out_hidden


def process_one_hot_encoding(X_train, X_test, x_cols):
    X_train_ohe = pd.concat(
        [X_train[x_cols], pd.get_dummies(X_train['z'])], axis=1)
    X_test_ohe = pd.concat(
        [X_test[x_cols], pd.get_dummies(X_test['z'])], axis=1)
    X_test_cols_in_train = set(
        X_test_ohe.columns).intersection(X_train_ohe.columns)
    X_train_cols_not_in_test = set(
        X_train_ohe.columns).difference(X_test_ohe.columns)
    X_test_comp = pd.DataFrame(np.zeros((X_test.shape[0], len(X_train_cols_not_in_test))),
                               columns=X_train_cols_not_in_test, dtype=np.uint8, index=X_test.index)
    X_test_ohe_comp = pd.concat(
        [X_test_ohe[X_test_cols_in_train], X_test_comp], axis=1)
    X_test_ohe_comp = X_test_ohe_comp[X_train_ohe.columns]
    return X_train_ohe, X_test_ohe_comp


def calc_b_hat(X_train, y_train, y_pred_tr, n_cats, sig2e, sig2b):
    b_hat = []
    for i in range(n_cats):
        i_vec = X_train['z'] == i
        n_i = i_vec.sum()
        if n_i > 0:
            y_bar_i = y_train[i_vec].mean()
            y_pred_i = y_pred_tr[i_vec].mean()
            # BP(b_i) = (n_i * sig2b / (sig2a + n_i * sig2b)) * (y_bar_i - y_pred_bar_i)
            b_i = n_i * sig2b * (y_bar_i - y_pred_i) / (sig2e + n_i * sig2b)
        else:
            b_i = 0
        b_hat.append(b_i)
    return np.array(b_hat)


def reg_nn_ohe_or_ignore(X_train, X_test, y_train, y_test, q, x_cols, batch_size, epochs, patience, deep=False, ignore_RE=False):
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
    return y_pred, (None, None), len(history.history['loss'])


def reg_nn_lmm(X_train, X_test, y_train, y_test, q, x_cols, batch_size, epochs, patience, deep=False):
    X_input = Input(shape=(X_train[x_cols].shape[1],))
    y_true_input = Input(shape=(1,))
    Z_input = Input(shape=(1,), dtype=tf.int64)
    if deep:
        out_hidden = add_deep_layers_functional(X_input)
    else:
        out_hidden = add_shallow_layers_functional(X_input)
    y_pred_output = Dense(1)(out_hidden)
    nll = NLL(1.0, 1.0)(y_true_input, y_pred_output, Z_input)
    model = Model(inputs=[X_input, y_true_input, Z_input], outputs=nll)

    model.compile(optimizer='adam')

    patience = epochs if patience is None else patience
    callbacks = [EarlyStoppingWithSigmasConvergence(patience=patience)]
    history = model.fit([X_train[x_cols], y_train, X_train['z']], None,
                        batch_size=batch_size, epochs=epochs, validation_split=0.1,
                        callbacks=callbacks, verbose=0)

    sig2e_est, sig2b_est = model.layers[-1].get_vars()
    y_pred_tr = model.predict(
        [X_train[x_cols], y_train, X_train['z']]).reshape(X_train.shape[0])
    b_hat = calc_b_hat(X_train, y_train, y_pred_tr, q, sig2e_est, sig2b_est)
    dummy_y_test = np.random.normal(size=y_test.shape)
    y_pred = model.predict([X_test[x_cols], dummy_y_test, X_test['z']]).reshape(
        X_test.shape[0]) + b_hat[X_test['z']]
    return y_pred, (sig2e_est, sig2b_est), len(history.history['loss'])


def reg_nn_embed(X_train, X_test, y_train, y_test, q, x_cols, batch_size, epochs, patience, deep=False):
    embed_dim = 10

    X_input = Input(shape=(X_train[x_cols].shape[1],))
    Z_input = Input(shape=(1,))
    embed = Embedding(q, embed_dim, input_length=1)(Z_input)
    embed = Reshape(target_shape=(embed_dim,))(embed)
    concat = Concatenate()([X_input, embed])
    if deep:
        out_hidden = add_deep_layers_functional(concat)
    else:
        out_hidden = add_shallow_layers_functional(concat)
    output = Dense(1)(out_hidden)
    model = Model(inputs=[X_input, Z_input], outputs=output)

    model.compile(loss='mse', optimizer='adam')

    callbacks = [EarlyStopping(
        monitor='val_loss', patience=epochs if patience is None else patience)]
    history = model.fit([X_train[x_cols], X_train['z']], y_train,
                        batch_size=batch_size, epochs=epochs, validation_split=0.1,
                        callbacks=callbacks, verbose=0)
    y_pred = model.predict([X_test[x_cols], X_test['z']]
                           ).reshape(X_test.shape[0])
    return y_pred, (None, None), len(history.history['loss'])


def reg_nn(X_train, X_test, y_train, y_test, q, x_cols, batch, epochs, patience, reg_type, deep):
    start = time.time()
    if reg_type == 'ohe':
        y_pred, sigmas, n_epochs = reg_nn_ohe_or_ignore(
            X_train, X_test, y_train, y_test, q, x_cols, batch, epochs, patience, deep)
    elif reg_type == 'lmm':
        y_pred, sigmas, n_epochs = reg_nn_lmm(
            X_train, X_test, y_train, y_test, q, x_cols, batch, epochs, patience, deep)
    elif reg_type == 'ignore':
        y_pred, sigmas, n_epochs = reg_nn_ohe_or_ignore(
            X_train, X_test, y_train, y_test, q, x_cols, batch, epochs, patience, deep, ignore_RE=True)
    else:
        y_pred, sigmas, n_epochs = reg_nn_embed(
            X_train, X_test, y_train, y_test, q, x_cols, batch, epochs, patience, deep)
    end = time.time()
    mse = np.mean((y_pred - y_test)**2)
    return NNResult(mse, sigmas, n_epochs, end - start)
