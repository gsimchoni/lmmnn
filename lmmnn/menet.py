import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model

def get_feature_map(model, X):
    last_layer = Model(inputs = model.input, outputs = model.layers[-2].output)
    return last_layer.predict(X)


def nll_i(y_i, f_hat_i, Z_i, b_hat_i, R_hat_i, D_hat):
    return np.transpose(y_i - f_hat_i - Z_i @ b_hat_i) @ np.linalg.inv(R_hat_i) @ (y_i - f_hat_i - Z_i @ b_hat_i) + \
                b_hat_i @ np.linalg.inv(D_hat) @ b_hat_i + np.log(np.linalg.det(D_hat)) + np.log(np.linalg.det(R_hat_i))


def compute_nll(model, X, y, b_hat, D_hat, sig2e_est, maps2ind, n_clusters, cnt_clusters):
    f_hat = model.predict(X, verbose=0).reshape(X.shape[0])
    Z = get_feature_map(model, X)
    nll = 0
    for cluster_id in range(n_clusters):
        indices_i = maps2ind[cluster_id]
        n_i = cnt_clusters[cluster_id]
        y_i = y[indices_i]
        Z_i = Z[indices_i, :]
        I_i = np.eye(n_i)
        f_hat_i = f_hat[indices_i]
        R_hat_i = sig2e_est * I_i
        b_hat_i = b_hat[cluster_id, :]
        nll = nll + nll_i(y_i, f_hat_i, Z_i, b_hat_i, R_hat_i, D_hat)
    return nll


def check_stop_model(nll_valid, best_loss, wait_loss, patience):
    stop_model = False
    if nll_valid < best_loss:
        best_loss = nll_valid
        wait_loss = 0
    else:
        wait_loss += 1
        if wait_loss >= patience:
            stop_model = True
    return best_loss, wait_loss, stop_model


def menet_fit(model, X, y, clusters, n_clusters, batch_size, epochs, patience, verbose=False):
    X_train, X_valid, y_train, y_valid, clusters_train, clusters_valid = train_test_split(X, y, clusters, test_size=0.1, random_state=0)
    maps2ind_train = [list(np.where(clusters_train == i)[0]) for i in range(n_clusters)]
    maps2ind_valid = [list(np.where(clusters_valid == i)[0]) for i in range(n_clusters)]
    cnt_clusters_train = Counter(clusters_train)
    cnt_clusters_valid = Counter(clusters_valid)
    Z = get_feature_map(model, X_train)
    d = Z.shape[1]
    b_hat = np.zeros((n_clusters, d))
    D_hat = np.eye(d)
    sig2e_est = 1.0
    nll_history = {'train': [], 'valid': []}
    best_loss = np.inf
    wait_loss = 0
    for epoch in range(epochs):
        y_star = np.zeros(y_train.shape)
        for cluster_id in range(n_clusters):
            indices_i = maps2ind_train[cluster_id]
            b_hat_i = b_hat[cluster_id, :]
            y_star_i = y_train[indices_i] - Z[indices_i, :] @ b_hat_i
            y_star[indices_i] = y_star_i
        model.fit(X_train, y_star, batch_size = batch_size, epochs=1, verbose=0)
        Z = get_feature_map(model, X_train)
        f_hat = model.predict(X_train, verbose=0).reshape(X_train.shape[0])
        sig2e_est_sum = 0
        D_hat_sum = 0
        for cluster_id in range(n_clusters):
            indices_i = maps2ind_train[cluster_id]
            n_i = cnt_clusters_train[cluster_id]
            f_hat_i = f_hat[indices_i]
            y_i = y_train[indices_i]
            Z_i = Z[indices_i, :]
            V_hat_i = Z_i @ D_hat @ np.transpose(Z_i) + sig2e_est * np.eye(n_i)
            V_hat_inv_i =  np.linalg.inv(V_hat_i)
            b_hat_i = D_hat @ np.transpose(Z_i) @ V_hat_inv_i @ (y_i - f_hat_i)
            eps_hat_i = y_i - f_hat_i - Z_i @ b_hat_i
            b_hat[cluster_id, :] = b_hat_i
            sig2e_est_sum = sig2e_est_sum + np.transpose(eps_hat_i) @ eps_hat_i + sig2e_est * (n_i - sig2e_est * np.trace(V_hat_inv_i))
            D_hat_sum = D_hat_sum + b_hat_i @ np.transpose(b_hat_i) + (D_hat - D_hat @ np.transpose(Z_i) @ V_hat_inv_i @ Z_i @ D_hat)
        sig2e_est = sig2e_est_sum / X_train.shape[0]
        D_hat = D_hat_sum / n_clusters
        # nll_train = compute_nll(model, X_train, y_train, b_hat, D_hat, sig2e_est, maps2ind_train, n_clusters, cnt_clusters_train)
        nll_valid = compute_nll(model, X_valid, y_valid, b_hat, D_hat, sig2e_est, maps2ind_valid, n_clusters, cnt_clusters_valid)
        # nll_history['train'].append(nll_train)
        nll_history['valid'].append(nll_valid)
        best_loss, wait_loss, stop_model = check_stop_model(nll_valid, best_loss, wait_loss, patience)
        if verbose:
            print(f'epoch: {epoch}, val_loss: {nll_valid:.2f}, sig2e_est: {sig2e_est:.2f}')
        if stop_model:
            break
    n_epochs = len(nll_history['valid'])
    return model, b_hat, sig2e_est, n_epochs, nll_history


def menet_predict(model, X, clusters, n_clusters, b_hat):
    y_hat = model.predict(X).reshape(X.shape[0])
    Z = get_feature_map(model, X)
    for cluster_id in range(n_clusters):
        indices_i = np.where(clusters == cluster_id)[0]
        if len(indices_i) == 0:
            continue
        b_i = b_hat[cluster_id, :]
        Z_i = Z[indices_i, :]
        y_hat[indices_i] = y_hat[indices_i] + Z_i @ b_i
    return y_hat


def compute_nll_generator(model, generator, b_hat, D_hat, sig2e_est, maps2ind, n_clusters, cnt_clusters):
    f_hat = model.predict(generator, verbose=0).reshape(generator.n)
    Z = get_feature_map(model, generator)
    y = generator.labels[:, 0]
    nll = 0
    for cluster_id in range(n_clusters):
        indices_i = maps2ind[cluster_id]
        n_i = cnt_clusters[cluster_id]
        y_i = y[indices_i]
        Z_i = Z[indices_i, :]
        I_i = np.eye(n_i)
        f_hat_i = f_hat[indices_i]
        R_hat_i = sig2e_est * I_i
        b_hat_i = b_hat[cluster_id, :]
        nll = nll + nll_i(y_i, f_hat_i, Z_i, b_hat_i, R_hat_i, D_hat)
    return nll


def menet_fit_generator(model, train_generator, valid_generator, clusters_train, clusters_valid, n_clusters, epochs, callbacks, patience, verbose=1):
    y_train = train_generator.labels[:, 0]
    y_valid = valid_generator.labels[:, 0]
    maps2ind_train = [list(np.where(clusters_train == i)[0]) for i in range(n_clusters)]
    maps2ind_valid = [list(np.where(clusters_valid == i)[0]) for i in range(n_clusters)]
    cnt_clusters_train = Counter(clusters_train)
    cnt_clusters_valid = Counter(clusters_valid)
    Z = get_feature_map(model, train_generator)
    d = Z.shape[1]
    b_hat = np.zeros((n_clusters, d))
    D_hat = np.eye(d)
    sig2e_est = 1.0
    nll_history = {'train': [], 'valid': []}
    best_loss = np.inf
    wait_loss = 0
    for epoch in range(epochs):
        y_star = np.zeros(y_train.shape)
        for cluster_id in range(n_clusters):
            indices_i = maps2ind_train[cluster_id]
            b_hat_i = b_hat[cluster_id, :]
            y_star_i = y_train[indices_i] - Z[indices_i, :] @ b_hat_i
            y_star[indices_i] = y_star_i
        train_generator.labels[:, 0] = y_star
        model.fit(train_generator, epochs=1, verbose=1)
        Z = get_feature_map(model, train_generator)
        f_hat = model.predict(train_generator, verbose=0).reshape(train_generator.n)
        sig2e_est_sum = 0
        D_hat_sum = 0
        for cluster_id in range(n_clusters):
            indices_i = maps2ind_train[cluster_id]
            n_i = cnt_clusters_train[cluster_id]
            f_hat_i = f_hat[indices_i]
            y_i = y_train[indices_i]
            Z_i = Z[indices_i, :]
            V_hat_i = Z_i @ D_hat @ np.transpose(Z_i) + sig2e_est * np.eye(n_i)
            V_hat_inv_i =  np.linalg.inv(V_hat_i)
            b_hat_i = D_hat @ np.transpose(Z_i) @ V_hat_inv_i @ (y_i - f_hat_i)
            eps_hat_i = y_i - f_hat_i - Z_i @ b_hat_i
            b_hat[cluster_id, :] = b_hat_i
            sig2e_est_sum = sig2e_est_sum + np.transpose(eps_hat_i) @ eps_hat_i + sig2e_est * (n_i - sig2e_est * np.trace(V_hat_inv_i))
            D_hat_sum = D_hat_sum + b_hat_i @ np.transpose(b_hat_i) + (D_hat - D_hat @ np.transpose(Z_i) @ V_hat_inv_i @ Z_i @ D_hat)
        sig2e_est = sig2e_est_sum / train_generator.n
        D_hat = D_hat_sum / n_clusters
        # nll_train = compute_nll_generator(model, train_generator, b_hat, D_hat, sig2e_est, maps2ind_train, n_clusters, cnt_clusters_train)
        nll_valid = compute_nll_generator(model, valid_generator, b_hat, D_hat, sig2e_est, maps2ind_valid, n_clusters, cnt_clusters_valid)
        # nll_history['train'].append(nll_train)
        nll_history['valid'].append(nll_valid)
        best_loss, wait_loss, stop_model = check_stop_model(nll_valid, best_loss, wait_loss, patience)
        if verbose:
            print(f'epoch: {epoch}, val_loss: {nll_valid:.2f}, sig2e_est: {sig2e_est:.2f}')
        if stop_model:
            break
    n_epochs = len(nll_history['valid'])
    return model, b_hat, sig2e_est, n_epochs, nll_history


def menet_predict_generator(model, test_generator, clusters, n_clusters, b_hat):
    y_hat = model.predict(test_generator).reshape(test_generator.n)
    Z = get_feature_map(model, test_generator)
    for cluster_id in range(n_clusters):
        indices_i = np.where(clusters == cluster_id)[0]
        if len(indices_i) == 0:
            continue
        b_i = b_hat[cluster_id, :]
        Z_i = Z[indices_i, :]
        y_hat[indices_i] = y_hat[indices_i] + Z_i @ b_i
    return y_hat