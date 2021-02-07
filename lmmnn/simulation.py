import logging

import pandas as pd

from lmmnn.nn import reg_nn
from lmmnn.utils import generate_data, NNInput, SimResult

logger = logging.getLogger('LMMNN.logger')


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
    if nn_in.q <= 10000:
        res = run_reg_nn(nn_in, 'ohe')
        ohe_res = summarize_sim(nn_in, res, 'ohe')
    else:
        ohe_res = None
        logger.warning(
            'OHE is unreasonable for a categorical variable of over 10K levels.')
    logger.debug('  Finished OHE.')
    res = run_reg_nn(nn_in, 'lmm')
    lmm_res = summarize_sim(nn_in, res, 'lmm')
    logger.debug('  Finished LMM.')
    res = run_reg_nn(nn_in, 'ignore')
    ig_res = summarize_sim(nn_in, res, 'ignore')
    logger.debug('  Finished Ignore.')
    res = run_reg_nn(nn_in, 'embed')
    embed_res = summarize_sim(nn_in, res, 'embed')
    logger.debug('  Finished Embedding.')
    res_df.loc[next(counter)] = ohe_res
    res_df.loc[next(counter)] = lmm_res
    res_df.loc[next(counter)] = ig_res
    res_df.loc[next(counter)] = embed_res
    res_df.to_csv(out_file)


def run_reg_nn(nn_in, reg_type):
    return reg_nn(nn_in.X_train, nn_in.X_test, nn_in.y_train, nn_in.y_test, nn_in.q, nn_in.x_cols, nn_in.batch, nn_in.epochs, nn_in.patience, reg_type=reg_type, deep=nn_in.deep)


def summarize_sim(nn_in, res, reg_type):
    return SimResult(nn_in.N, nn_in.sig2e, nn_in.sig2b, nn_in.q, nn_in.deep, nn_in.k, reg_type, res.mse, res.sigmas[0], res.sigmas[1], res.n_epochs, res.time)


def simulation(out_file, params):
    counter = Count().gen()
    res_df = pd.DataFrame(columns=['N', 'sig2e', 'sig2b', 'q', 'deep', 'experiment', 'exp_type',
                                   'mse', 'sig2e_est', 'sig2b_est', 'n_epochs', 'time'])
    deep = params['deep']
    for N in params['N_list']:
        for sig2e in params['sig2e_list']:
            for sig2b in params['sig2b_list']:
                for q in params['q_list']:
                    logger.info('N: %d, sig2e: %.2f; sig2b: %.2f; q: %d' %
                                (N, sig2e, sig2b, q))
                    for k in range(params['n_iter']):
                        X_train, X_test, y_train, y_test, x_cols = generate_data(
                            q, sig2e, sig2b, N, params)
                        if deep in ['no', 'both']:
                            logger.info(' iteration: %d, deep: %s' %
                                        (k, False))
                            nn_in = NNInput(X_train, X_test, y_train, y_test, x_cols, N, q, sig2e,
                                            sig2b, k, False, params['batch'], params['epochs'], params['patience'])
                            iterate_reg_types(counter, res_df, out_file, nn_in)
                        if deep in ['yes', 'both']:
                            logger.info(' iteration: %d, deep: %s' % (k, True))
                            nn_in = NNInput(X_train, X_test, y_train, y_test, x_cols, N, q, sig2e,
                                            sig2b, k, True, params['batch'], params['epochs'], params['patience'])
                            iterate_reg_types(counter, res_df, out_file, nn_in)
