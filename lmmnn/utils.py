import pandas as pd
import numpy as np
from collections import namedtuple
from scipy import sparse
from sklearn.model_selection import train_test_split

SimResult = namedtuple('SimResult',
                       ['N', 'sig2e', 'sig2bs', 'qs', 'deep', 'iter_id', 'exp_type', 'mse', 'sig2e_est', 'sig2b_ests', 'n_epochs', 'time'])

NNResult = namedtuple('NNResult', ['mse', 'sigmas', 'rhos', 'n_epochs', 'time'])

NNInput = namedtuple('NNInput', ['X_train', 'X_test', 'y_train', 'y_test', 'x_cols',
                                 'N', 'qs', 'sig2e', 'sig2bs', 'rhos', 'k', 'deep', 'batch', 'epochs', 'patience',
                                 'Z_non_linear', 'Z_embed_dim_pct', 'mode', 'n_sig2bs', 'estimated_cors'])

def get_dummies(vec, vec_max):
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


def generate_data(mode, qs, sig2e, sig2bs, N, rhos, params):
    n_fixed_effects = params['n_fixed_effects']
    X = np.random.uniform(-1, 1, N * n_fixed_effects).reshape((N, n_fixed_effects))
    betas = np.ones(n_fixed_effects)
    Xbeta = params['fixed_intercept'] + X @ betas
    e = np.random.normal(0, np.sqrt(sig2e), N)
    if params['X_non_linear']:
        fX = Xbeta * np.cos(Xbeta) + 2 * X[:, 0] * X[:, 1]
    else:
        fX = Xbeta
    df = pd.DataFrame(X)
    x_cols = ['X' + str(i) for i in range(n_fixed_effects)]
    df.columns = x_cols
    y = fX + e
    if mode == 'intercepts':
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
            df['z' + str(k)] = Z_idx
    elif mode == 'slopes': # len(qs) should b1 1
        fs = np.random.poisson(params['n_per_cat'], qs[0]) + 1
        fs_sum = fs.sum()
        ps = fs/fs_sum
        ns = np.random.multinomial(N, ps)
        Z_idx = np.repeat(range(qs[0]), ns)
        max_period = np.arange(ns.max())
        t = np.concatenate([max_period[:k] for k in ns]) / max_period[-1]
        y += t + t ** 2 # fixed part
        cov_mat = get_cov_mat(sig2bs, rhos, params['estimated_cors'])
        bs = np.random.multivariate_normal(np.zeros(len(sig2bs)), cov_mat, qs[0])
        b = bs.reshape((qs[0] * len(sig2bs),), order = 'F')
        Z0 = sparse.csr_matrix(get_dummies(Z_idx, qs[0]))
        Z_list = [Z0]
        for k in range(1, len(sig2bs)):
            Z_list.append(sparse.spdiags(t ** k, 0, N, N) @ Z0)
        Zb = sparse.hstack(Z_list) @ b
        y = y + Zb
        df['t'] = t
        df['z0'] = Z_idx
        x_cols.append('t')
    df['y'] = y
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('y', axis=1), df['y'], test_size=0.2)
    return X_train, X_test, y_train, y_test, x_cols
