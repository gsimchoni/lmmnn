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


def iterate_reg_types(counter, res_df, out_file, nn_in, exp_types):
    if 'ohe' in exp_types:
        if all(map(lambda q: q <= 10000, nn_in.qs)):
            res = run_reg_nn(nn_in, 'ohe')
            ohe_res = summarize_sim(nn_in, res, 'ohe')
        else:
            ohe_res = None
            logger.warning(
                'OHE is unreasonable for a categorical variable of over 10K levels.')
        res_df.loc[next(counter)] = ohe_res
        logger.debug('  Finished OHE.')
    if 'ignore' in exp_types:
        res = run_reg_nn(nn_in, 'ignore')
        ig_res = summarize_sim(nn_in, res, 'ignore')
        res_df.loc[next(counter)] = ig_res
        logger.debug('  Finished Ignore.')
    if 'embeddings' in exp_types:
        res = run_reg_nn(nn_in, 'embed')
        embed_res = summarize_sim(nn_in, res, 'embed')
        res_df.loc[next(counter)] = embed_res
        logger.debug('  Finished Embedding.')
    if 'lmmnn' in exp_types:
        res = run_reg_nn(nn_in, 'lmm')
        lmm_res = summarize_sim(nn_in, res, 'lmm')
        res_df.loc[next(counter)] = lmm_res
        logger.debug('  Finished LMM.')
    if 'menet' in exp_types:
        if len(nn_in.qs) == 1 and nn_in.mode == 'intercepts':
            res = run_reg_nn(nn_in, 'menet')
            me_res = summarize_sim(nn_in, res, 'menet')
            res_df.loc[next(counter)] = me_res
            logger.debug('  Finished MeNet.')
    res_df.to_csv(out_file)


def run_reg_nn(nn_in, reg_type):
    return reg_nn(nn_in.X_train, nn_in.X_test, nn_in.y_train, nn_in.y_test, nn_in.qs,
        nn_in.x_cols, nn_in.batch, nn_in.epochs, nn_in.patience,
        nn_in.n_neurons, nn_in.dropout, nn_in.activation, reg_type=reg_type,
        Z_non_linear=nn_in.Z_non_linear, Z_embed_dim_pct = nn_in.Z_embed_dim_pct,
        mode = nn_in.mode, n_sig2bs = nn_in.n_sig2bs, est_cors = nn_in.estimated_cors,
        dist_matrix = nn_in.dist_matrix, spatial_embed_neurons = nn_in.spatial_embed_neurons,
        verbose = nn_in.verbose, log_params = nn_in.log_params, idx = nn_in.k)


def summarize_sim(nn_in, res, reg_type):
    if nn_in.spatial_embed_neurons is None:
        spatial_embed_out_dim = []
    else:
        spatial_embed_out_dim = [nn_in.spatial_embed_neurons[-1]]
    if nn_in.weibull_lambda is None:
        weibull_params = []
    else:
        weibull_params = [nn_in.p_censor, nn_in.weibull_lambda, nn_in.weibull_nu]
    res = [nn_in.mode, nn_in.N, nn_in.sig2e] + list(nn_in.sig2bs) + list(nn_in.qs) + \
        weibull_params + list(nn_in.rhos) + spatial_embed_out_dim +\
        [nn_in.k, reg_type, res.metric, res.sigmas[0]] + res.sigmas[1] + res.rhos + res.weibull +\
        [res.n_epochs, res.time]
    return res


def simulation(out_file, params):
    counter = Count().gen()
    n_sig2bs = len(params['sig2b_list'])
    n_categoricals = len(params['q_list'])
    n_rhos = len(params['rho_list'])
    estimated_cors = params['estimated_cors']
    mode = params['mode']
    spatial_embed_out_dim_name = []
    p_censor_name = []
    p_censor_list = [0.0]
    weibull_nu_name = []
    weibull_lambda_name = []
    weibull_nu_est_name = []
    weibull_lambda_est_name = []
    rhos_names =  []
    rhos_est_names =  []
    metric = 'mse'
    if mode == 'intercepts':
        assert n_sig2bs == n_categoricals
    elif mode == 'slopes':
        assert n_categoricals == 1
        assert n_rhos == len(estimated_cors)
        rhos_names =  list(map(lambda x: 'rho' + str(x), range(n_rhos)))
        rhos_est_names =  list(map(lambda x: 'rho_est' + str(x), range(n_rhos)))
    elif mode == 'glmm':
        assert n_categoricals == 1
        assert n_sig2bs == n_categoricals
        metric = 'auc'
    elif mode == 'spatial':
        assert n_categoricals == 1
        assert n_sig2bs == 2
    elif mode == 'spatial_embedded':
        assert n_categoricals == 1
        assert n_sig2bs == 2
        spatial_embed_out_dim_name = ['spatial_embed_out_dim']
    elif mode == 'survival':
        assert n_categoricals == 1
        assert n_sig2bs == n_categoricals
        metric = 'concordance'
        p_censor_name = ['p_censor']
        weibull_nu_name = ['weibull_nu']
        weibull_lambda_name = ['weibull_lambda']
        weibull_nu_est_name = ['weibull_nu_est']
        weibull_lambda_est_name = ['weibull_lambda_est']
        p_censor_list = params['p_censor_list']
    else:
        raise ValueError('Unknown mode')
    qs_names =  list(map(lambda x: 'q' + str(x), range(n_categoricals)))
    sig2bs_names =  list(map(lambda x: 'sig2b' + str(x), range(n_sig2bs)))
    sig2bs_est_names =  list(map(lambda x: 'sig2b_est' + str(x), range(n_sig2bs)))
    
    res_df = pd.DataFrame(columns=['mode', 'N', 'sig2e'] + sig2bs_names + qs_names + rhos_names +
        spatial_embed_out_dim_name + p_censor_name + weibull_nu_name + weibull_lambda_name +
        ['experiment', 'exp_type', metric, 'sig2e_est'] +
        sig2bs_est_names + rhos_est_names + weibull_nu_est_name + weibull_lambda_est_name + ['n_epochs', 'time'])
    for N in params['N_list']:
        for sig2e in params['sig2e_list']:
            for qs in product(*params['q_list']):
                for sig2bs in product(*params['sig2b_list']):
                    for rhos in product(*params['rho_list']):
                        for p_censor in p_censor_list:
                            logger.info('mode: %s, N: %d, sig2e: %.2f; sig2bs: [%s]; qs: [%s]; rhos: [%s], p_censor: %.2f' %
                                        (mode, N, sig2e, ', '.join(map(str, sig2bs)), ', '.join(map(str, qs)), ', '.join(map(str, rhos)), p_censor))
                            for k in range(params['n_iter']):
                                X_train, X_test, y_train, y_test, x_cols, dist_matrix = generate_data(
                                    mode, qs, sig2e, sig2bs, N, rhos, p_censor, params)
                                logger.info(' iteration: %d' % k)
                                nn_in = NNInput(X_train, X_test, y_train, y_test, x_cols, N, qs, sig2e, p_censor,
                                                sig2bs, rhos, k, params['batch'], params['epochs'], params['patience'],
                                                params['Z_non_linear'], params['Z_embed_dim_pct'], mode, n_sig2bs,
                                                params['estimated_cors'], dist_matrix, params['verbose'],
                                                params['n_neurons'], params['dropout'], params['activation'],
                                                params['spatial_embed_neurons'], params['log_params'],
                                                params['weibull_lambda'], params['weibull_nu'])
                                iterate_reg_types(counter, res_df, out_file, nn_in, params['exp_types'])