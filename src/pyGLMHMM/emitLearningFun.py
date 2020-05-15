import numpy as np

from numba import jit
from scipy.sparse import spdiags
from scipy.linalg import block_diag

@jit
def _emit_learning_fun(emit_w, stim, state_num, options):   
    # emit_w are the weights that we are learning: in format states x weights
    # stim is a list/dictionary with each stimulus (stim[]['data']) and the probability emission functions (stim[]['gamma'] and stim[]['xi'])

    num_states = stim[0]['num_states']
    num_total_bins = stim[0]['num_total_bins']
    # states x bins
    emit_w = np.reshape(emit_w, (num_total_bins, num_states), order = 'F').T

    all_grad = np.zeros((num_states, num_total_bins))
    all_value = 0
    total_T = 0

    this_lambda = options['emit_lambda']

    # Find out how many data points we are dealing with so that we can normalize
    # (I don't think we actually need to do this but it helps keep regularization values consistent from fit to fit)
    for trial in range(0, len(stim)):
        total_T = total_T + stim[trial]['data'].shape[1]

    for trial in range(0, len(stim)):
        T = stim[trial]['data'].shape[1]
        
        # Convert into states x bins x time and sum across bins
        filtpower = np.reshape(np.sum(np.reshape(np.tile(np.expand_dims(emit_w, axis = 2), (1, 1, T)), (num_states, num_total_bins, T), order = 'F') * np.tile(np.reshape(stim[trial]['data'], (1, num_total_bins, T), order = 'F'), (num_states, 1, 1)), axis = 1), (num_states, T), order = 'F')
        # Now filtpower is states x time
        # filtpower is the filter times the stimulus

        # Build up the value function:
        # gamma * log(exp(filtpower) / (1 + sum(exp(filtpower)))) = gamma * filtpower - gamma * log(1 + sum(exp(filtpower)))
        # Gradient is then:
        # gamma * (1|emission - exp(filtpower) / (1+sum(exp(filtpower)))) * stim
        value = stim[trial]['gamma'][state_num, :] * -np.log(1 + np.sum(np.exp(filtpower), axis = 0))
        tgrad = -np.exp(filtpower) / np.tile(1 + np.sum(np.exp(filtpower), axis = 0), (num_states))

        for i in range(0, filtpower.shape[0]):
            tgrad[i, stim[trial]['emit'].astype(int) == (i + 1)] = 1 + tgrad[i, stim[trial]['emit'].astype(int) == (i + 1)]
            value[stim[trial]['emit'].astype(int) == (i + 1)] = value[stim[trial]['emit'].astype(int) == (i + 1)] + stim[trial]['gamma'][state_num, stim[trial]['emit'].astype(int) == (i + 1)] * filtpower[i, stim[trial]['emit'].astype(int) == (i + 1)]

        value = np.sum(value, axis = 0)
        if np.any(np.isnan(value)):
            print('Ugh!')

        tgrad = tgrad * np.tile(stim[trial]['gamma'][state_num, :], (num_states))
        tgrad = np.sum(np.tile(np.reshape(tgrad, (num_states, 1, T), order = 'F'), (1, num_total_bins, 1)) * np.tile(np.reshape(stim[trial]['data'], (1, num_total_bins, T), order = 'F'), (num_states, 1, 1)), axis = 2)

        all_grad = all_grad + tgrad
        all_value = all_value + value
    
    grad_regularization = 0
    value_regularization = 0

    if options['L2_smooth'] == True:

        Dx1 = spdiags((np.ones((emit_w.shape[1] - 1, 1)) * np.array([-1, 1])).T, np.array([0, 1]), emit_w.shape[1] - 1 - 1, emit_w.shape[1] - 1).toarray()
        Dx = np.matmul(Dx1.T, Dx1)

        for fstart in range(options['num_filter_bins'], emit_w.shape[1] - 1, options['num_filter_bins']):
            Dx[fstart, fstart] = 1
            Dx[fstart - 1, fstart - 1] = 1
            Dx[fstart - 1, fstart] = 0
            Dx[fstart, fstart - 1] = 0
        
        D = block_diag(Dx, 0)
        
        if options['AR_lambda'] != -1:
            if len(options['smooth_lambda']) == 1:
                options['smooth_lambda'] = np.tile(options['smooth_lambda'][0], [emit_w.shape[0], emit_w.shape[1]])
                options['smooth_lambda'][:, options['AR_vec']] = options['AR_lambda']
            
            grad_regularization = grad_regularization + options['smooth_lambda'] * (np.matmul(D, emit_w.T)).T
            value_regularization = value_regularization + np.sum(np.sum(np.power((options['smooth_lambda'] / 2) * (np.matmul(D, emit_w.T)).T, 2), axis = 0), axis = 0)
        else:
            grad_regularization = grad_regularization + options['smooth_lambda'] * (np.matmul(D, emit_w.T)).T
            value_regularization = value_regularization + np.sum(np.sum(np.power((options['smooth_lambda'] / 2) * (np.matmul(D, emit_w.T)).T, 2), axis = 0), axis = 0)
    
    if this_lambda != 0:
        if options['AR_lambda'] != -1:
            grad_regularization = grad_regularization + [this_lambda * emit_w[:, options['stim_vec']], options['AR_lambda'] * emit_w[:, options['AR_vec']]]
            value_regularization = value_regularization + (this_lambda / 2) * np.sum(np.sum(np.power(emit_w[:, options['stim_vec']], 2), axis = 0), axis = 0) + (options['AR_lambda'] / 2) * np.sum(np.sum(np.power(emit_w[:, options['AR_vec']], 2), axis = 0), axis = 0)
        else:
            grad_regularization = grad_regularization + this_lambda * emit_w
            value_regularization = value_regularization + (this_lambda/2) * np.sum(np.sum(np.power(emit_w, 2), axis = 0), axis = 0)

    all_grad = -all_grad / total_T + grad_regularization
    all_value = -all_value / total_T + value_regularization

    if np.any(np.isnan(all_grad)) or np.any(np.isnan(all_value)):
        print('WTF! SOMETHING BAD HAPPENED! ISNAN! OH!')
 
    all_grad = np.reshape(all_grad.T, (all_grad.shape[0] * all_grad.shape[1]), order = 'F')
    
    return all_value, all_grad