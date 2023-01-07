import time
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
try:
    from lifelines.utils import concordance_index
except Exception:
    pass

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Concatenate, Reshape, Input, Masking, LSTM, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras import Model
import torch

from lmmnn.utils import NNResult, get_dummies
from lmmnn.callbacks import LogEstParams, EarlyStoppingWithSigmasConvergence
from lmmnn.layers import NLL
from lmmnn.menet import menet_fit, menet_predict
from lmmnn.calc_b_hat import *
try:
    import gpytorch
    from lmmnn.gpytorch_classes import *
except Exception:
    pass


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
    return X_input


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
            columns=list(X_train_cols_not_in_test), dtype=np.uint8, index=X_test.index)
        X_test_ohe_comp = pd.concat([X_test_ohe[list(X_test_cols_in_train)], X_test_comp], axis=1)
        X_test_ohe_comp = X_test_ohe_comp[X_train_ohe.columns]
        X_train_ohe.columns = list(map(lambda c: z_col + '_' + str(c), X_train_ohe.columns))
        X_test_ohe_comp.columns = list(map(lambda c: z_col + '_' + str(c), X_test_ohe_comp.columns))
        X_train_new = pd.concat([X_train_new, X_train_ohe], axis=1)
        X_test_new = pd.concat([X_test_new, X_test_ohe_comp], axis=1)
    return X_train_new, X_test_new


def process_X_to_rnn(X, y, time2measure_dict, x_cols):
    X['measure'] = X['t'].map(time2measure_dict)
    X_rnn = X.pivot(index='z0', columns=['measure'], values = x_cols).fillna(0)
    y_rnn = pd.concat([X[['z0', 'measure']], y], axis=1).pivot(index='z0', columns=['measure'], values = 'y').fillna(0)
    rnn_cols = [(x_col, i) for x_col in x_cols for i in range(len(time2measure_dict))]
    for i, col in enumerate(rnn_cols):
        if col not in X_rnn.columns:
            X_rnn.insert(loc=i, column=col, value=0)
    for i in range(len(time2measure_dict)):
        if i not in y_rnn.columns:
            y_rnn.insert(loc=i, column=i, value=0)
    X_rnn = X_rnn.values.reshape(-1,len(x_cols),len(time2measure_dict)).transpose([0,2,1])
    y_rnn = y_rnn.values.reshape(-1,1,len(time2measure_dict)).transpose([0,2,1])
    return X_rnn, y_rnn


def reg_nn_rnn(X_train, X_test, y_train, y_test, qs, x_cols, batch_size, epochs,
        patience, n_neurons, dropout, activation, mode, time2measure_dict,
        n_sig2bs, n_sig2bs_spatial, est_cors, verbose=False):
    X_train_rnn, y_train_rnn = process_X_to_rnn(X_train, y_train, time2measure_dict, x_cols)
    X_test_rnn, y_test_rnn = process_X_to_rnn(X_test, y_test, time2measure_dict, x_cols) 
    model = Sequential([
        Masking(mask_value=.0, input_shape=(len(time2measure_dict), len(x_cols))),
        LSTM(5, return_sequences=True),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    callbacks = [EarlyStopping(
        monitor='val_loss', patience=epochs if patience is None else patience)]
    history = model.fit(X_train_rnn, y_train_rnn, batch_size=batch_size,
                    epochs=epochs, validation_split=0.1, verbose=verbose,
                    callbacks=callbacks)
    y_pred = model.predict(X_test_rnn, verbose=verbose)
    mse = np.mean((y_test_rnn[y_test_rnn != 0] - y_pred[y_test_rnn != 0])**2)
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    none_rhos = [None for _ in range(len(est_cors))]
    none_weibull = [None, None] if mode == 'survival' else []
    return mse, (None, none_sigmas, none_sigmas_spatial), none_rhos, none_weibull, len(history.history['loss'])


def reg_nn_ohe_or_ignore(X_train, X_test, y_train, y_test, qs, x_cols, batch_size, epochs,
        patience, n_neurons, dropout, activation, mode,
        n_sig2bs, n_sig2bs_spatial, est_cors, verbose=False, ignore_RE=False):
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
    y_pred = model.predict(X_test, verbose=verbose).reshape(X_test.shape[0])
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    none_rhos = [None for _ in range(len(est_cors))]
    none_weibull = [None, None] if mode == 'survival' else []
    return y_pred, (None, none_sigmas, none_sigmas_spatial), none_rhos, none_weibull, len(history.history['loss'])


def process_X_to_images(X, resolution = 100, min_X = -10, max_X = 10):
    X_images = np.zeros((X.shape[0], resolution, resolution, 1), dtype=np.uint8)
    bins = np.linspace(min_X, max_X, resolution)
    X_binned = np.digitize(X, bins) - 1
    i = np.arange(X.shape[0])
    j = X_binned[:, 0]
    k = resolution - 1 - X_binned[:, 1]
    X_images[i, k, j] = 1
    return X_images


def add_layers_cnn(cnn_in):
    x = Conv2D(32, (2, 2), activation='relu')(cnn_in)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(64, (2, 2), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(32, (2, 2), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(16, (2, 2), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(10, activation='relu')(x)
    return x


def reg_nn_cnn(X_train, X_test, y_train, y_test, qs, x_cols, batch_size, epochs,
        patience, n_neurons, dropout, activation, mode,
        n_sig2bs, n_sig2bs_spatial, est_cors, resolution, verbose=False):
    x_cols_mlp = x_cols
    x_cols_cnn = X_train.columns[X_train.columns.str.startswith('D')]
    X_train_features, X_test_features = X_train[x_cols_mlp], X_test[x_cols_mlp]
    X_train_images = process_X_to_images(X_train[x_cols_cnn], resolution)
    X_test_images = process_X_to_images(X_test[x_cols_cnn], resolution)
    if mode == 'glmm':
        loss = 'binary_crossentropy'
        last_layer_activation = 'sigmoid'
    else:
        loss = 'mse'
        last_layer_activation = 'linear'
    
    cnn_in = Input((resolution, resolution, 1))
    cnn_out = add_layers_cnn(cnn_in)
    mlp_in = Input(len(x_cols_mlp))
    mlp_out = add_layers_functional(mlp_in, n_neurons, dropout, activation, len(x_cols_mlp))
    concat = Concatenate()([mlp_out, cnn_out])
    output = Dense(1, activation=last_layer_activation)(concat)
    model = Model(inputs=[mlp_in, cnn_in], outputs=output)

    model.compile(loss=loss, optimizer='adam')

    callbacks = [EarlyStopping(
        monitor='val_loss', patience=epochs if patience is None else patience)]
    history = model.fit([X_train_features, X_train_images], y_train, batch_size=batch_size, epochs=epochs,
                        validation_split=0.1, callbacks=callbacks, verbose=verbose)
    y_pred = model.predict([X_test_features, X_test_images], verbose=verbose).reshape(X_test_features.shape[0])
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    none_rhos = [None for _ in range(len(est_cors))]
    none_weibull = [None, None] if mode == 'survival' else []
    return y_pred, (None, none_sigmas, none_sigmas_spatial), none_rhos, none_weibull, len(history.history['loss'])


def reg_nn_dkl(X_train, X_test, y_train, y_test, qs, x_cols, batch_size, epochs,
        patience, n_neurons, dropout, activation, mode,
        n_sig2bs, n_sig2bs_spatial, est_cors, verbose=False):
    x_cols_mlp = [col for col in x_cols if col not in ['D1', 'D2']]
    x_cols_gp = ['D1', 'D2']

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)
    train_x_mlp = torch.Tensor(X_train[x_cols_mlp].values)
    train_x_gp = torch.Tensor(X_train[x_cols_gp].values)
    train_y = torch.Tensor(y_train.values)
    valid_x_mlp = torch.Tensor(X_valid[x_cols_mlp].values)
    valid_x_gp = torch.Tensor(X_valid[x_cols_gp].values)
    valid_y = torch.Tensor(y_valid.values)
    test_x_mlp = torch.Tensor(X_test[x_cols_mlp].values)
    test_x_gp = torch.Tensor(X_test[x_cols_gp].values)
    test_y = torch.Tensor(y_test.values)

    if torch.cuda.is_available():
        train_x_mlp, train_x_gp, train_y, valid_x_mlp, valid_x_gp, valid_y, test_x_mlp, test_x_gp, test_y = train_x_mlp.cuda(), train_x_gp.cuda(), train_y.cuda(), valid_x_mlp.cuda(), valid_x_gp.cuda(), valid_y.cuda(), test_x_mlp.cuda(), test_x_gp.cuda(), test_y.cuda()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = DKLModel((train_x_gp, train_x_mlp), train_y, likelihood, MLP(X_train[x_cols_mlp].shape[1], n_neurons, dropout, activation))

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    optimizer = torch.optim.Adam(model.parameters())

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def train():
        best_val_loss = np.Inf
        es_counter = 0
        train_losses = []
        val_losses = []
        for i in range(epochs):
            model.train()
            likelihood.train()
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            train_output = model(train_x_gp, train_x_mlp)
            # Calc loss and backprop derivatives
            train_loss = -mll(train_output, train_y)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
                valid_output = model(valid_x_gp, valid_x_mlp)
                val_loss = -mll(valid_output, valid_y).item()
            val_losses.append(val_loss)
            if verbose:
                print(f'epoch: {i}, loss: {train_loss.item():.4f}, val_loss: {val_loss:.4f}')
            if val_loss < best_val_loss:
                es_counter = 0
                best_val_loss = val_loss
            elif patience is not None and es_counter >= patience - 1:
                break
            else:
                es_counter += 1
        return train_losses, val_losses
    train_loss, valid_loss = train()
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        y_pred = likelihood(model(test_x_gp, test_x_mlp)).loc.cpu().numpy()
    sig2e_est = likelihood.noise.detach().cpu().numpy()[0]
    none_sigmas = [None for _ in range(n_sig2bs)]
    sig2b_spatial_est = model.covar_module.base_kernel.outputscale.detach().cpu().numpy().item()
    none_rhos = [None for _ in range(len(est_cors))]
    none_weibull = [None, None] if mode == 'survival' else []
    return y_pred, (sig2e_est, none_sigmas, [sig2b_spatial_est, None]), none_rhos, none_weibull, len(train_loss)


def reg_nn_svdkl(X_train, X_test, y_train, y_test, qs, x_cols, batch_size, epochs,
        patience, n_neurons, dropout, activation, mode,
        n_sig2bs, n_sig2bs_spatial, est_cors, verbose=False):
    x_cols_mlp = [col for col in x_cols if col not in ['D1', 'D2']]
    x_cols_gp = ['D1', 'D2']

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)
    train_x_mlp = torch.Tensor(X_train[x_cols_mlp].values)
    train_x_gp = torch.Tensor(X_train[x_cols_gp].values)
    train_y = torch.Tensor(y_train.values)
    valid_x_mlp = torch.Tensor(X_valid[x_cols_mlp].values)
    valid_x_gp = torch.Tensor(X_valid[x_cols_gp].values)
    valid_y = torch.Tensor(y_valid.values)
    test_x_mlp = torch.Tensor(X_test[x_cols_mlp].values)
    test_x_gp = torch.Tensor(X_test[x_cols_gp].values)
    test_y = torch.Tensor(y_test.values)

    if torch.cuda.is_available():
        train_x_mlp, train_x_gp, train_y, valid_x_mlp, valid_x_gp, valid_y, test_x_mlp, test_x_gp, test_y = train_x_mlp.cuda(), train_x_gp.cuda(), train_y.cuda(), valid_x_mlp.cuda(), valid_x_gp.cuda(), valid_y.cuda(), test_x_mlp.cuda(), test_x_gp.cuda(), test_y.cuda()

    train_dataset = torch.utils.data.TensorDataset(train_x_mlp, train_x_gp, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = torch.utils.data.TensorDataset(valid_x_mlp, valid_x_gp, valid_y)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    test_dataset = torch.utils.data.TensorDataset(test_x_mlp, test_x_gp, test_y)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_inducing_points = 500
    inducing_points = torch.Tensor(X_train[x_cols_gp].values[:n_inducing_points, :])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SVDKLModel(inducing_points, MLP(X_train[x_cols_mlp].shape[1], n_neurons, dropout, activation))

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
    
    optimizer = torch.optim.Adam(model.parameters())

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    def train_epoch(train_dataloader, valid_dataloader, model, optimizer, mll):
        model.train()
        likelihood.train()
        train_losses = []
        for X_mlp, X_gp, y in train_dataloader:
            if torch.cuda.is_available():
                X_mlp, X_gp, y = X_mlp.cuda(), X_gp.cuda(), y.cuda()
            optimizer.zero_grad()
            output = model(X_gp, x_mlp=X_mlp, prior=False, n_inducing_points=n_inducing_points)
            loss = -mll(output, y)
            loss.backward(retain_graph=True)
            train_losses.append(loss.item())
            optimizer.step()
        train_loss = np.average(train_losses)
        model.eval()
        likelihood.eval()
        val_losses = []
        for X_mlp, X_gp, y in valid_dataloader:
            if torch.cuda.is_available():
                X_mlp, X_gp, y = X_mlp.cuda(), X_gp.cuda(), y.cuda()
            with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
                output = model(X_gp, x_mlp=X_mlp, prior=False, n_inducing_points=n_inducing_points)
                loss = -mll(output, y).item()
            val_losses.append(loss)
        val_loss = np.average(val_losses)
        return train_loss, val_loss

    def train(train_dataloader, valid_dataloader, model, optimizer, mll):
        train_loss = []
        valid_loss = []
        best_val_loss = np.Inf
        es_counter = 0
        for i in range(epochs):
            train_loss_epoch, val_loss_epoch = train_epoch(train_dataloader, valid_dataloader, model, optimizer, mll)
            train_loss.append(train_loss_epoch)
            valid_loss.append(val_loss_epoch)
            if verbose:
                print(f'epoch: {i}, loss: {train_loss_epoch:.4f}, val_loss: {val_loss_epoch:.4f}')
            if val_loss_epoch < best_val_loss:
                es_counter = 0
                best_val_loss = val_loss_epoch
            elif patience is not None and es_counter >= patience - 1:
                break
            else:
                es_counter += 1
        return train_loss, valid_loss
    
    def test(test_dataloader, model):
        model.eval()
        likelihood.eval()
        y_pred_list = []
        for X_mlp, X_gp, y in test_dataloader:
            if torch.cuda.is_available():
                X_mlp, X_gp, y = X_mlp.cuda(), X_gp.cuda(), y.cuda()
            with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
                y_pred_batch = likelihood(model(X_gp, x_mlp=X_mlp, prior=False, n_inducing_points = n_inducing_points)).loc.cpu().numpy()
                y_pred_list.append(y_pred_batch)
        y_pred = np.concatenate(y_pred_list)
        return y_pred
    
    train_loss, valid_loss = train(train_dataloader, valid_dataloader, model, optimizer, mll)
    y_pred = test(test_dataloader, model)
    sig2e_est = likelihood.noise.detach().cpu().numpy()[0]
    none_sigmas = [None for _ in range(n_sig2bs)]
    sig2b_spatial_outputscale_est = model.covar_module.outputscale.detach().cpu().numpy().item()
    sig2b_spatial_lengthscale_est = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()[0][0]
    none_rhos = [None for _ in range(len(est_cors))]
    none_weibull = [None, None] if mode == 'survival' else []
    return y_pred, (sig2e_est, none_sigmas, [sig2b_spatial_outputscale_est, sig2b_spatial_lengthscale_est]), none_rhos, none_weibull, len(train_loss)


def reg_nn_lmm(X_train, X_test, y_train, y_test, qs, q_spatial, x_cols, batch_size, epochs, patience, n_neurons, dropout, activation,
        mode, n_sig2bs, n_sig2bs_spatial, est_cors, dist_matrix, spatial_embed_neurons,
        verbose=False, Z_non_linear=False, Z_embed_dim_pct=10, log_params=False, idx=0, shuffle=False, sample_n_train=10000):
    if mode in ['spatial', 'spatial_embedded', 'spatial_and_categoricals']:
        x_cols = [x_col for x_col in x_cols if x_col not in ['D1', 'D2']]
    if mode == 'survival':
        x_cols = [x_col for x_col in x_cols if x_col not in ['C0']]
    # dmatrix_tf = tf.constant(dist_matrix)
    dmatrix_tf = dist_matrix
    X_input = Input(shape=(X_train[x_cols].shape[1],))
    y_true_input = Input(shape=(1,))
    if mode in ['intercepts', 'glmm', 'spatial', 'spatial_and_categoricals']:
        z_cols = sorted(X_train.columns[X_train.columns.str.startswith('z')].tolist())
        Z_inputs = []
        if mode == 'spatial':
            n_sig2bs_init = n_sig2bs_spatial
            n_RE_inputs = 1
        elif mode == 'spatial_and_categoricals':
            n_sig2bs_init = n_sig2bs_spatial + len(qs)
            n_RE_inputs = 1 + len(qs)
        else:
            n_sig2bs_init = len(qs)
            n_RE_inputs = len(qs)
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
        # callbacks = [EarlyStoppingWithSigmasConvergence(patience=patience)]
        callbacks = [EarlyStopping(patience=patience, monitor='val_loss')]
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
                        callbacks=callbacks, verbose=verbose, shuffle=shuffle)

    sig2e_est, sig2b_ests, rho_ests, weibull_ests = model.layers[-1].get_vars()
    if mode in ['spatial', 'spatial_embedded']:
        sig2b_spatial_ests = sig2b_ests
        sig2b_ests = []
    elif mode == 'spatial_and_categoricals':
        sig2b_spatial_ests = sig2b_ests[:2]
        sig2b_ests = sig2b_ests[2:]
    else:
        sig2b_spatial_ests = []
    y_pred_tr = model.predict(
        [X_train[x_cols], y_train] + X_train_z_cols, verbose=verbose).reshape(X_train.shape[0])
    b_hat = calc_b_hat(X_train, y_train, y_pred_tr, qs, q_spatial, sig2e_est, sig2b_ests, sig2b_spatial_ests,
                Z_non_linear, model, ls, mode, rho_ests, est_cors, dist_matrix, weibull_ests, sample_n_train)
    dummy_y_test = np.random.normal(size=y_test.shape)
    if mode in ['intercepts', 'glmm', 'spatial', 'spatial_and_categoricals']:
        if Z_non_linear or len(qs) > 1 or mode == 'spatial_and_categoricals':
            delta_loc = 0
            if mode == 'spatial_and_categoricals':
                delta_loc = 1
            Z_tests = []
            for k, q in enumerate(qs):
                Z_test = get_dummies(X_test['z' + str(k + delta_loc)], q)
                if Z_non_linear:
                    W_est = model.get_layer('Z_embed' + str(k)).get_weights()[0]
                    Z_test = Z_test @ W_est
                Z_tests.append(Z_test)
            if Z_non_linear:
                Z_test = np.hstack(Z_tests)
            else:
                Z_test = sparse.hstack(Z_tests)
            if mode == 'spatial_and_categoricals':
                Z_test = sparse.hstack([Z_test, get_dummies(X_test['z0'], q_spatial)])
            y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + Z_test @ b_hat
        else:
            # if model input is that large, this 2nd call to predict may cause OOM due to GPU memory issues
            # if that is the case use tf.convert_to_tensor() explicitly with a call to model() without using predict() method
            # y_pred = model([tf.convert_to_tensor(X_test[x_cols]), tf.convert_to_tensor(dummy_y_test), tf.convert_to_tensor(X_test_z_cols[0])], training=False).numpy().reshape(
            #     X_test.shape[0]) + b_hat[X_test['z0']]
            y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + b_hat[X_test['z0']]
        if mode == 'glmm':
            y_pred = np.exp(y_pred)/(1 + np.exp(y_pred))
    elif mode == 'slopes':
        q = qs[0]
        Z0 = get_dummies(X_test['z0'], q)
        t = X_test['t'].values
        N = X_test.shape[0]
        Z_list = [Z0]
        for k in range(1, len(sig2b_ests)):
            Z_list.append(sparse.spdiags(t ** k, 0, N, N) @ Z0)
        Z_test = sparse.hstack(Z_list)
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + Z_test @ b_hat
    elif mode == 'spatial_embedded':
        last_layer = Model(inputs = model.input[2], outputs = model.layers[-2].output)
        gZ_test = last_layer.predict(X_test_z_cols, verbose=verbose)
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + gZ_test @ b_hat
        sig2b_spatial_ests = np.concatenate([sig2b_spatial_ests, [np.nan]])
    elif mode == 'survival':
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0])
        y_pred = y_pred + np.log(b_hat[X_test['z0']])
    return y_pred, (sig2e_est, list(sig2b_ests), list(sig2b_spatial_ests)), list(rho_ests), list(weibull_ests), len(history.history['loss'])


def reg_nn_embed(X_train, X_test, y_train, y_test, qs, q_spatial, x_cols, batch_size, epochs, patience,
        n_neurons, dropout, activation, mode, n_sig2bs, n_sig2bs_spatial, est_cors, verbose=False):
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
    qs_list = list(qs)
    if q_spatial is not None:
        qs_list +=  [q_spatial]
    for q in qs_list:
        Z_input = Input(shape=(1,))
        embed = Embedding(q, embed_dim, input_length=1)(Z_input)
        embed = Reshape(target_shape=(embed_dim,))(embed)
        Z_inputs.append(Z_input)
        embeds.append(embed)
    concat = Concatenate()([X_input] + embeds)
    out_hidden = add_layers_functional(concat, n_neurons, dropout, activation, X_train[x_cols].shape[1] + embed_dim * len(qs_list))
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
    y_pred = model.predict([X_test[x_cols]] + X_test_z_cols, verbose=verbose
                           ).reshape(X_test.shape[0])
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    none_rhos = [None for _ in range(len(est_cors))]
    none_weibull = [None, None] if mode == 'survival' else []
    return y_pred, (None, none_sigmas, none_sigmas_spatial), none_rhos, none_weibull, len(history.history['loss'])


def reg_nn_menet(X_train, X_test, y_train, y_test, q, x_cols, batch_size, epochs, patience,
        n_neurons, dropout, activation, mode, n_sig2bs, n_sig2bs_spatial, est_cors, verbose=False):
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
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    none_rhos = [None for _ in range(len(est_cors))]
    none_weibull = [None, None] if mode == 'survival' else []
    return y_pred, (sig2e_est, none_sigmas, none_sigmas_spatial), none_rhos, none_weibull, n_epochs


def reg_nn(X_train, X_test, y_train, y_test, qs, q_spatial, x_cols,
        batch, epochs, patience, n_neurons, dropout, activation, reg_type,
        Z_non_linear, Z_embed_dim_pct, mode, n_sig2bs, n_sig2bs_spatial, est_cors,
        dist_matrix, time2measure_dict, spatial_embed_neurons, resolution, verbose, log_params, idx, shuffle):
    start = time.time()
    if reg_type == 'ohe':
        y_pred, sigmas, rhos, weibull, n_epochs = reg_nn_ohe_or_ignore(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, n_sig2bs, n_sig2bs_spatial, est_cors, verbose)
    elif reg_type == 'lmm':
        y_pred, sigmas, rhos, weibull, n_epochs = reg_nn_lmm(
            X_train, X_test, y_train, y_test, qs, q_spatial, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode,
            n_sig2bs, n_sig2bs_spatial, est_cors, dist_matrix, spatial_embed_neurons, verbose,
            Z_non_linear, Z_embed_dim_pct, log_params, idx, shuffle)
    elif reg_type == 'ignore':
        y_pred, sigmas, rhos, weibull, n_epochs = reg_nn_ohe_or_ignore(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, n_sig2bs, n_sig2bs_spatial, est_cors, verbose, ignore_RE=True)
    elif reg_type == 'embed':
        y_pred, sigmas, rhos, weibull, n_epochs = reg_nn_embed(
            X_train, X_test, y_train, y_test, qs, q_spatial, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, n_sig2bs, n_sig2bs_spatial, est_cors, verbose)
    elif reg_type == 'menet':
        y_pred, sigmas, rhos, weibull, n_epochs = reg_nn_menet(
            X_train, X_test, y_train, y_test, qs[0], x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, n_sig2bs, n_sig2bs_spatial, est_cors, verbose)
    elif reg_type == 'rnn':
        mse_rnn, sigmas, rhos, weibull, n_epochs = reg_nn_rnn(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, time2measure_dict, n_sig2bs, n_sig2bs_spatial, est_cors, verbose)
    elif reg_type == 'dkl':
        y_pred, sigmas, rhos, weibull, n_epochs = reg_nn_dkl(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, n_sig2bs, n_sig2bs_spatial, est_cors, verbose)
    elif reg_type == 'svdkl':
        y_pred, sigmas, rhos, weibull, n_epochs = reg_nn_svdkl(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, n_sig2bs, n_sig2bs_spatial, est_cors, verbose)
    elif reg_type == 'cnn':
        y_pred, sigmas, rhos, weibull, n_epochs = reg_nn_cnn(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, n_sig2bs, n_sig2bs_spatial, est_cors, resolution, verbose)
    else:
        raise ValueError(reg_type + 'is an unknown reg_type')
    end = time.time()
    K.clear_session()
    gc.collect()
    if mode == 'glmm':
        metric = roc_auc_score(y_test, y_pred)
    elif mode == 'survival':
        if np.any(np.isnan(y_pred)):
            metric = np.nan
        else:
            metric = concordance_index(y_test, -y_pred, X_test['C0'])
    elif mode == 'slopes' and reg_type == 'rnn':
        metric = mse_rnn
    else:
        metric = np.mean((y_pred - y_test)**2)
    return NNResult(metric, sigmas, rhos, weibull, n_epochs, end - start)
