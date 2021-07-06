from itertools import product
import os
import logging

import pandas as pd

from lmmnn.nn import reg_nn
from lmmnn.utils import generate_data, NNInput, SimResult

logger = logging.getLogger('LMMNN.logger')
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

class Count:
    curr = 0

    def __init__(self, startWith=None):
        if startWith is not None:
            Count.curr = startWith - 1

    def gen(self):
        while True:
            Count.curr += 1
            yield Count.curr


def iterate_reg_types(counter, res_df, out_file, nn_in):
    if all(map(lambda q: q <= 10000, nn_in.qs)):
        res = run_reg_nn(nn_in, 'ohe')
        ohe_res = summarize_sim(nn_in, res, 'ohe')
    else:
        ohe_res = None
        logger.warning(
            'OHE is unreasonable for a categorical variable of over 10K levels.')
    logger.debug('  Finished OHE.')
    res = run_reg_nn(nn_in, 'ignore')
    ig_res = summarize_sim(nn_in, res, 'ignore')
    logger.debug('  Finished Ignore.')
    res = run_reg_nn(nn_in, 'embed')
    embed_res = summarize_sim(nn_in, res, 'embed')
    logger.debug('  Finished Embedding.')
    res = run_reg_nn(nn_in, 'lmm')
    lmm_res = summarize_sim(nn_in, res, 'lmm')
    logger.debug('  Finished LMM.')
    if len(nn_in.qs) == 1:
        res = run_reg_nn(nn_in, 'menet')
        me_res = summarize_sim(nn_in, res, 'menet')
        logger.debug('  Finished MeNet.')
    res_df.loc[next(counter)] = ohe_res
    res_df.loc[next(counter)] = lmm_res
    res_df.loc[next(counter)] = ig_res
    res_df.loc[next(counter)] = embed_res
    if len(nn_in.qs) == 1:
        res_df.loc[next(counter)] = me_res
    res_df.to_csv(out_file)


def run_reg_nn(nn_in, reg_type):
    return reg_nn(nn_in.X_train, nn_in.X_test, nn_in.y_train, nn_in.y_test, nn_in.qs,
        nn_in.x_cols, nn_in.batch, nn_in.epochs, nn_in.patience, reg_type=reg_type,
        deep=nn_in.deep, Z_non_linear=nn_in.Z_non_linear, Z_embed_dim_pct = nn_in.Z_embed_dim_pct)


def summarize_sim(nn_in, res, reg_type):
    res = [nn_in.N, nn_in.sig2e] + list(nn_in.sig2bs) + list(nn_in.qs) + \
        [nn_in.deep, nn_in.k, reg_type, res.mse, res.sigmas[0]] + res.sigmas[1] + \
        [res.n_epochs, res.time]
    return res


def simulation(out_file, params):
    counter = Count().gen()
    n_categoricals = len(params['q_list'])
    assert n_categoricals == len(params['sig2b_list'])
    qs_names =  list(map(lambda x: 'q' + str(x), range(n_categoricals)))
    sig2bs_names =  list(map(lambda x: 'sig2b' + str(x), range(n_categoricals)))
    sig2bs_est_names =  list(map(lambda x: 'sig2b_est' + str(x), range(n_categoricals)))
    res_df = pd.DataFrame(columns=['N', 'sig2e'] + sig2bs_names + qs_names +
        ['deep', 'experiment', 'exp_type', 'mse', 'sig2e_est'] +
        sig2bs_est_names + ['n_epochs', 'time'])
    deep = params['deep']
    for N in params['N_list']:
        for sig2e in params['sig2e_list']:
            for qs in product(*params['q_list']):
                for sig2bs in product(*params['sig2b_list']):
                    logger.info('N: %d, sig2e: %.2f; sig2bs: [%s]; qs: [%s]' %
                                (N, sig2e, ', '.join(map(str, sig2bs)), ', '.join(map(str, qs))))
                    for k in range(params['n_iter']):
                        X_train, X_test, y_train, y_test, x_cols = generate_data(
                            qs, sig2e, sig2bs, N, params)
                        if deep in ['no', 'both']:
                            logger.info(' iteration: %d, deep: %s' %
                                        (k, False))
                            nn_in = NNInput(X_train, X_test, y_train, y_test, x_cols, N, qs, sig2e,
                                            sig2bs, k, False, params['batch'], params['epochs'], params['patience'],
                                            params['Z_non_linear'], params['Z_embed_dim_pct'])
                            iterate_reg_types(counter, res_df, out_file, nn_in)
                        if deep in ['yes', 'both']:
                            logger.info(' iteration: %d, deep: %s' % (k, True))
                            nn_in = NNInput(X_train, X_test, y_train, y_test, x_cols, N, qs, sig2e,
                                            sig2bs, k, True, params['batch'], params['epochs'], params['patience'],
                                            params['Z_non_linear'], params['Z_embed_dim_pct'])
                            iterate_reg_types(counter, res_df, out_file, nn_in)
