import pandas as pd
import numpy as np
from collections import namedtuple
from scipy import sparse
from scipy.spatial.kdtree import distance_matrix
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform

SimResult = namedtuple('SimResult',
                       ['N', 'sig2e', 'sig2bs', 'qs', 'deep', 'iter_id', 'exp_type', 'mse', 'sig2e_est', 'sig2b_ests', 'n_epochs', 'time'])

NNResult = namedtuple('NNResult', ['metric', 'sigmas', 'rhos', 'weibull', 'n_epochs', 'time'])

NNInput = namedtuple('NNInput', ['X_train', 'X_test', 'y_train', 'y_test', 'x_cols',
                                 'N', 'qs', 'sig2e', 'p_censor', 'sig2bs', 'rhos', 'sig2bs_spatial', 'q_spatial',
                                 'k', 'batch', 'epochs', 'patience',
                                 'Z_non_linear', 'Z_embed_dim_pct', 'mode', 'n_sig2bs', 'n_sig2bs_spatial', 'estimated_cors',
                                 'dist_matrix', 'time2measure_dict', 'verbose', 'n_neurons', 'dropout', 'activation',
                                 'spatial_embed_neurons', 'log_params',
                                 'weibull_lambda', 'weibull_nu', 'resolution', 'shuffle'])

def get_dummies(vec, vec_max):
    vec_size = vec.size
    Z = sparse.csr_matrix((np.ones(vec_size), (np.arange(vec_size), vec)), shape=(vec_size, vec_max), dtype=np.uint8)
    return Z

def get_dummies_np(vec, vec_max):
    vec_size = vec.size
    Z = np.zeros((vec_size, vec_max), dtype=np.uint8)
    Z[np.arange(vec_size), vec] = 1
    return Z

def get_cov_mat(sig2bs, rhos, est_cors):
    cov_mat = np.zeros((len(sig2bs), len(sig2bs)))
    for k in range(len(sig2bs)):
        for j in range(len(sig2bs)):
            if k == j:
                cov_mat[k, j] = sig2bs[k]
            else:
                rho_symbol = ''.join(map(str, sorted([k, j])))
                if rho_symbol in est_cors:
                    rho = rhos[est_cors.index(rho_symbol)]
                else:
                    rho = 0
                cov_mat[k, j] = rho * np.sqrt(sig2bs[k]) * np.sqrt(sig2bs[j])
    return cov_mat


def generate_data(mode, qs, sig2e, sig2bs, sig2bs_spatial, q_spatial, N, rhos, p_censor, params):
    n_fixed_effects = params['n_fixed_effects']
    X = np.random.uniform(-1, 1, N * n_fixed_effects).reshape((N, n_fixed_effects))
    betas = np.ones(n_fixed_effects)
    if mode == 'survival':
        Xbeta = X @ betas
    else:
        Xbeta = params['fixed_intercept'] + X @ betas
    dist_matrix = None
    time2measure_dict = None
    if params['X_non_linear']:
        fX = Xbeta * np.cos(Xbeta) + 2 * X[:, 0] * X[:, 1]
    else:
        fX = Xbeta
    df = pd.DataFrame(X)
    x_cols = ['X' + str(i) for i in range(n_fixed_effects)]
    df.columns = x_cols
    if mode == 'glmm':
        y = fX
    else:
        e = np.random.normal(0, np.sqrt(sig2e), N)
        y = fX + e
    if mode in ['intercepts', 'glmm', 'spatial_and_categoricals']:
        delta_loc = 0
        if mode == 'spatial_and_categoricals':
            delta_loc = 1
        for k, q in enumerate(qs):
            fs = np.random.poisson(params['n_per_cat'], q) + 1
            fs_sum = fs.sum()
            ps = fs/fs_sum
            ns = np.random.multinomial(N, ps)
            Z_idx = np.repeat(range(q), ns)
            if params['Z_non_linear']:
                Z = get_dummies(Z_idx, q)
                l = int(q * params['Z_embed_dim_pct'] / 100.0)
                b = np.random.normal(0, np.sqrt(sig2bs[k]), l)
                W = np.random.uniform(-1, 1, q * l).reshape((q, l))
                gZb = Z @ W @ b
            else:
                b = np.random.normal(0, np.sqrt(sig2bs[k]), q)
                gZb = np.repeat(b, ns)
            y = y + gZb
            df['z' + str(k + delta_loc)] = Z_idx
    if mode == 'slopes': # len(qs) should be 1
        fs = np.random.poisson(params['n_per_cat'], qs[0]) + 1
        fs_sum = fs.sum()
        ps = fs/fs_sum
        ns = np.random.multinomial(N, ps)
        Z_idx = np.repeat(range(qs[0]), ns)
        max_period = np.arange(ns.max())
        t = np.concatenate([max_period[:k] for k in ns]) / max_period[-1]
        estimated_cors = [] if params['estimated_cors'] is None else params['estimated_cors']
        cov_mat = get_cov_mat(sig2bs, rhos, estimated_cors)
        bs = np.random.multivariate_normal(np.zeros(len(sig2bs)), cov_mat, qs[0])
        b = bs.reshape((qs[0] * len(sig2bs),), order = 'F')
        Z0 = sparse.csr_matrix(get_dummies(Z_idx, qs[0]))
        Z_list = [Z0]
        for k in range(1, len(sig2bs)):
            y += t ** k # fixed part t + t^2 + t^3 + ...
            Z_list.append(sparse.spdiags(t ** k, 0, N, N) @ Z0)
        Zb = sparse.hstack(Z_list) @ b
        y = y + Zb
        df['t'] = t
        df['z0'] = Z_idx
        x_cols.append('t')
        time2measure_dict = {t: i for i, t in enumerate(np.sort(df['t'].unique()))}
    if mode in ['spatial', 'spatial_embedded', 'spatial_and_categoricals']:
        coords = np.stack([np.random.uniform(-10, 10, q_spatial), np.random.uniform(-10, 10, q_spatial)], axis=1)
        # ind = np.lexsort((coords[:, 1], coords[:, 0]))    
        # coords = coords[ind]
        dist_matrix = squareform(pdist(coords)) ** 2
        D = sig2bs_spatial[0] * np.exp(-dist_matrix / (2 * sig2bs_spatial[1]))
        b = np.random.multivariate_normal(np.zeros(q_spatial), D, 1)[0]
        fs = np.random.poisson(params['n_per_cat'], q_spatial) + 1
        fs_sum = fs.sum()
        ps = fs/fs_sum
        ns = np.random.multinomial(N, ps)
        Z_idx = np.repeat(range(q_spatial), ns)
        gZb = np.repeat(b, ns)
        df['z0'] = Z_idx
        y = y + gZb
        coords_df = pd.DataFrame(coords[Z_idx])
        co_cols = ['D1', 'D2']
        coords_df.columns = co_cols
        df = pd.concat([df, coords_df], axis=1)
        x_cols.extend(co_cols)
    if mode == 'glmm':
        p = np.exp(y)/(1 + np.exp(y))
        y = np.random.binomial(1, p, size=N)
    if mode == 'survival': # len(qs) should be 1
        fs = np.random.poisson(params['n_per_cat'], qs[0]) + 1
        fs_sum = fs.sum()
        ps = fs / fs_sum
        ns = np.random.multinomial(N, ps)
        Z_idx = np.repeat(range(qs[0]), ns)
        if params['Z_non_linear']:
            Z = get_dummies(Z_idx, qs[0])
            l = int(qs[0] * params['Z_embed_dim_pct'] / 100.0)
            b = np.random.gamma(1 / sig2bs[0], sig2bs[0], l)
            W = np.random.uniform(-1, 1, qs[0] * l).reshape((qs[0], l))
            gZb = Z @ W @ b
        else:
            b = np.random.gamma(1 / sig2bs[0], sig2bs[0], qs[0])
            gZb = np.repeat(b, ns)
        y = (-np.log(np.random.uniform(size = N)) / (params['weibull_lambda'] * np.exp(fX) * gZb)) ** (1 / params['weibull_nu'])
        y = np.clip(y, None, 1e+15)
        cens = np.random.binomial(1, p_censor / 100.0, size = N)
        df['z0'] = Z_idx
        df['C0'] = 1 - cens
        x_cols.extend(['C0'])
    df['y'] = y
    test_size = params['test_size'] if 'test_size' in params else 0.2
    pred_future = params['longitudinal_predict_future'] if 'longitudinal_predict_future' in params and mode == 'slopes' else False
    if  pred_future:
        # test set is "the future" or those obs with largest t
        df.sort_values('t', inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('y', axis=1), df['y'], test_size=test_size, shuffle=not pred_future)
    return X_train, X_test, y_train, y_test, x_cols, dist_matrix, time2measure_dict
