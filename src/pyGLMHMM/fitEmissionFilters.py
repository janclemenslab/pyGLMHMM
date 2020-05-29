import numpy as np

from .emitLearningFun import _emit_learning_fun
from .emitMultistepLearningFun import _emit_multistep_learning_fun
from .emitLikelihood import _emit_likelihood
from .emitLearningStats import _emit_learning_stats
from .minimizeLBFGS import _minimize_LBFGS

def _fit_emission_filters(stim, symb, gamma, xi, emit_w, options, train_data):
    
    new_stim = []
        
    for trial in range(0, len(train_data)):
        # Please don't ask me why I decided it was a good idea to call the number of emissions 'num_states' here. Just roll with it!
        new_stim.append({'emit' : symb[train_data[trial]], 'gamma' : gamma[train_data[trial]], 'xi' : xi[train_data[trial]], 'num_states' : options['num_emissions']})
        
        if options['GLM_emissions'] == True:
            new_stim[trial]['data'] = stim[train_data[trial]]
            new_stim[trial]['num_total_bins'] = options['num_total_bins']                            
        else:
            new_stim[trial]['data']  = stim[train_data[trial]][-1, :]
            new_stim[trial]['num_total_bins'] = 1
    
    tmp_pgd1 = np.zeros((options['num_states'], 1))
    tmp_pgd2 = np.zeros((options['num_states'], 1))
    tmp_pgd3 = np.zeros((options['num_states'], options['num_emissions'] + 1, options['num_emissions'] + 1))
    tmp_pgd4 = np.zeros((options['num_states'], options['num_emissions'] + 1, options['num_emissions'] + 1))
    tmp_pgd_lik = np.zeros((options['num_states'], 1))
    
    if options['evaluate'] == True:
        for i in range(0, options['num_states']):
            [tmp_pgd_lik[i], tmp_pgd1[i], tmp_pgd2[i], tmp_pgd3_temp, tmp_pgd4_temp] = _emit_likelihood(np.reshape(emit_w[i, :, :].T, (emit_w.shape[1] * emit_w.shape[2], 1), order = 'F'), new_stim, i)
            tmp_pgd3[i, :, :] = tmp_pgd3_temp
            tmp_pgd4[i, :, :] = tmp_pgd4_temp
        
        pgd_lik = np.sum(tmp_pgd_lik, axis = 0)
        pgd_prob1 = np.sum(tmp_pgd1, axis = 0)
        pgd_prob2 = np.sum(tmp_pgd2, axis = 0)
        pgd_prob3 = np.sum(tmp_pgd3, axis = 0)
        pgd_prob4 = np.sum(tmp_pgd4, axis = 0)

        if options['generate'] == True:
            for trial in range(0, len(stim)):
                new_stim[trial]['analog_symb'] = np.nan
                new_stim[trial]['analog_emit_w'] = 0
    
    else:
        hess_diag_emit = np.zeros((options['num_states'], options['num_emissions'], options['num_total_bins']))
    
        for i in range(0, options['num_states']):
            if options['num_steps'] == 1:
                outweights = _minimize_LBFGS(lambda x: _emit_learning_fun(x, new_stim, i, options), np.reshape(emit_w[i, :, :].T, (emit_w.shape[1] * emit_w.shape[2]), order = 'F'), lr = 1, max_iter = 500, tol = 1e-5, line_search = 'Wolfe', interpolate = True, max_ls = 25, history_size = 100, out = True)
            else:
                outweights = _minimize_LBFGS(lambda x: _emit_multistep_learning_fun(x, new_stim, i, options), np.reshape(emit_w[i, :, :].T, (emit_w.shape[1] * emit_w.shape[2]), order = 'F'), lr = 1, max_iter = 500, tol = 1e-5, line_search = 'Wolfe', interpolate = True, max_ls = 25, history_size = 100, out = True)
            
            emit_w[i, :, :] = np.reshape(outweights, (emit_w.shape[2], emit_w.shape[1]), order = 'F').T   # Make sure this is reformatted properly!!!
            
            [tmp_pgd1[i], hess_d] = _emit_learning_stats(np.reshape(emit_w[i, :, :].T, (emit_w.shape[1] * emit_w.shape[2], 1), order = 'F'), new_stim, i, options)
            hess_diag_emit[i, :, :] = np.reshape(hess_d, (hess_diag_emit.shape[2], hess_diag_emit.shape[1]), order = 'F').T
                
        pgd_lik = np.sum(tmp_pgd1, axis = 0)
        
        for i in range(0, options['num_states']):            
            [tmp_pgd_lik[i], tmp_pgd1[i], tmp_pgd2[i], tmp_pgd3_temp, tmp_pgd4_temp] = _emit_likelihood(np.reshape(emit_w[i, :, :].T, (emit_w.shape[1] * emit_w.shape[2], 1), order = 'F'), new_stim, i)
            tmp_pgd3[i, :, :] = tmp_pgd3_temp
            tmp_pgd4[i, :, :] = tmp_pgd4_temp
    
        pgd_prob1 = np.sum(tmp_pgd1, axis = 0)
        pgd_prob2 = np.sum(tmp_pgd2, axis = 0)
        pgd_prob3 = np.sum(tmp_pgd3, axis = 0)
        pgd_prob4 = np.sum(tmp_pgd4, axis = 0) 
    
    return emit_w, pgd_lik, pgd_prob1, pgd_prob2, pgd_prob3, pgd_prob4