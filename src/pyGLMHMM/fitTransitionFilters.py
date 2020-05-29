import numpy as np

from .transLearningFun import _trans_learning_fun
from .transLearningStats import _trans_learning_stats
from .minimizeLBFGS import _minimize_LBFGS

def _fit_transition_filters(stim, symb, gamma, xi, trans_w, options, train_data):
    new_stim = []
            
    for trial in range(0, len(train_data)):
        new_stim.append({'gamma' : gamma[train_data[trial]], 'xi' : xi[train_data[trial]], 'num_states' : options['num_states']})
        
        if options['GLM_transitions']  == True:
                new_stim[trial]['data'] = stim[train_data[trial]]
                new_stim[trial]['num_total_bins'] = options['num_total_bins']
        else:
            new_stim[trial]['data'] = stim[train_data[trial]][-1, :]
            new_stim[trial]['num_total_bins'] = 1
    
    tmp_tgd = np.zeros(options['num_states'])

    if options['evaluate'] == True:
        for i in range(0, options['num_states']):
            tmp_tgd[i] = _trans_learning_fun(np.reshape(trans_w[i, :, :].T, (trans_w.shape[1] * trans_w.shape[2]), order = 'F'), new_stim, i, options)[0]
        
        tgd_lik = np.sum(tmp_tgd, axis = 0)
    
    else:
        hess_diag_trans = np.zeros((options['num_states'], options['num_states'], options['num_total_bins']))

        for i in range(0, options['num_states']):
            outweights = _minimize_LBFGS(lambda x: _trans_learning_fun(x, new_stim, i, options), np.reshape(trans_w[i, :, :].T, (trans_w.shape[1] * trans_w.shape[2]), order = 'F'), lr = 1, max_iter = 500, tol = 1e-5, line_search = 'Wolfe', interpolate = True, max_ls = 25, history_size = 100, out = True)
            trans_w[i, :, :] = np.reshape(outweights, (trans_w.shape[2], trans_w.shape[1]), order = 'F').T
            
            tmp_tgd[i] = _trans_learning_fun(np.reshape(trans_w[i, :, :].T, (trans_w.shape[1] * trans_w.shape[2]), order = 'F'), new_stim, i, options)[0]

            if options['num_states'] > 1:
                [tmp_tgd[i], hess_d] = _trans_learning_stats(np.reshape(trans_w[i, :, :].T, (trans_w.shape[1] * trans_w.shape[2], 1), order = 'F'), new_stim, i, options)
                hess_diag_trans[i, :, :] = np.reshape(hess_d, (hess_diag_trans.shape[2], hess_diag_trans.shape[1])).T
            else:
                hess_diag_trans = 0

        tgd_lik = np.sum(tmp_tgd, axis = 0)

    return trans_w, tgd_lik