import pandas as pd
import numpy as np
from collections import namedtuple
from sklearn.model_selection import train_test_split

SimResult = namedtuple('SimResult',
                       ['N', 'sig2e', 'sig2b', 'q', 'deep', 'iter_id', 'exp_type', 'mse', 'sig2e_est', 'sig2b_est', 'n_epochs', 'time'])

NNResult = namedtuple('NNResult', ['mse', 'sigmas', 'n_epochs', 'time'])

NNInput = namedtuple('NNInput', ['X_train', 'X_test', 'y_train', 'y_test', 'x_cols',
                                 'N', 'q', 'sig2e', 'sig2b', 'k', 'deep', 'batch', 'epochs', 'patience'])


def generate_data(q, sig2e, sig2b, N, params):
    n_fixed_effects = params['n_fixed_effects']
    fs = np.random.poisson(params['n_per_cat'], q) + 1
    fs_sum = fs.sum()
    ps = fs/fs_sum
    ns = np.random.multinomial(N, ps)
    Z_idx = np.repeat(range(q), ns)
    b = np.random.normal(0, np.sqrt(sig2b), q)
    Zb = np.repeat(b, ns)
    X = np.random.uniform(-1, 1, N * n_fixed_effects).reshape((N, n_fixed_effects))
    betas = np.ones(n_fixed_effects)
    Xbeta = params['fixed_intercept'] + X @ betas
    e = np.random.normal(0, np.sqrt(sig2e), N)
    if params['non_linear']:
        fX = Xbeta * np.cos(Xbeta) + 2 * X[:, 0] * X[:, 1]
    else:
        fX = Xbeta
    y = fX + Zb + e
    X_df = pd.DataFrame(X)
    x_cols = ['X' + str(i) for i in range(n_fixed_effects)]
    X_df.columns = x_cols
    df = pd.concat([pd.DataFrame({'y': y, 'z': Z_idx}), X_df], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('y', axis=1), df['y'], test_size=0.2)
    return X_train, X_test, y_train, y_test, x_cols
