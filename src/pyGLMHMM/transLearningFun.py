import copy
import numpy as np

from scipy.sparse import spdiags
from scipy.linalg import block_diag

def _trans_learning_fun(trans_w, stim, state_num, options):    
    # trans_w are the weights that we are learning: in format states x weights
    # stim is a list/dictionary with each stimulus (stim[]['data']) and the probability transition functions (stim[]['gamma'] and stim[]['xi'])
    # NOTE: This transition function is dependent on where we are transitioning from, and relies on each of the other possible states we could be transitioning to,
    # so we cannot minimize these independently. Thus we are really going to find the gradient of all transition filters originating from some state.

    num_states = stim[0]['num_states']
    num_total_bins = stim[0]['num_total_bins']
    
    trans_w = np.reshape(trans_w, (num_total_bins, num_states), order = 'F').T

    all_grad = np.zeros((num_states, num_total_bins))
    all_value = 0
    total_T = 0

    this_lambda = options['trans_lambda']

    for trial in range(0, len(stim)):
        total_T = total_T + stim[trial]['data'].shape[1]

    for trial in range(0, len(stim)):
        T = stim[trial]['data'].shape[1] - 1
        # Use data from 1:end-1 or 2:end?
        filtpower = np.sum(np.tile(np.expand_dims(trans_w, axis = 2), (1, 1, T)) * np.tile(np.reshape(stim[trial]['data'][:, 1:], (1, num_total_bins, T), order = 'F'), (num_states, 1, 1)), axis = 1)
        # Now filtpower is states x time

        value = -stim[trial]['gamma'][state_num, 0:-1] * np.log(1 + np.sum(np.exp(filtpower[np.setdiff1d(np.arange(num_states), state_num), :]), axis = 0))
        
        if stim[trial]['xi'].shape[2] == 1:
            tgrad = copy.copy(stim[trial]['xi'][state_num, :, :].T)
        else:
            tgrad = copy.copy(stim[trial]['xi'][state_num, :, :])

        i = state_num
        
        # Should it be 1:end-1 or 2:end?
        offset = stim[trial]['gamma'][i, 0:-1] / (1 + np.sum(np.exp(filtpower[np.setdiff1d(np.arange(num_states), i), :]), axis = 0))
        
        for j in range(0, num_states):
            if i != j:
                value = value + stim[trial]['xi'][state_num, j, :].T * filtpower[j, :]
                tgrad[j, :] = tgrad[j, :] - np.exp(filtpower[j, :]) * offset
            else:
                tgrad[j, :] = 0

        tgrad = np.sum(np.tile(np.reshape(tgrad, (num_states, 1, T), order = 'F'), (1, num_total_bins, 1)) * np.tile(np.reshape(stim[trial]['data'][:, 1:], (1, num_total_bins, T), order = 'F'), (num_states, 1, 1)), axis = 2)

        # I probably don't need to rescale here because that happens naturally but... oh well!
        all_grad = all_grad + tgrad
        all_value = all_value + np.sum(value, axis = 0)

    grad_regularization = np.zeros(all_grad.shape)
    value_regularization = 0

    if options['L2_smooth'] == True:

        Dx1 = spdiags((np.ones((trans_w.shape[1] - 1, 1)) * np.array([-1, 1])).T, np.array([0, 1]), trans_w.shape[1] - 1 - 1, trans_w.shape[1] - 1).toarray()
        Dx = np.matmul(Dx1.T, Dx1)

        for fstart in range(options['num_filter_bins'], trans_w.shape[1] - 1, options['num_filter_bins']):
            Dx[fstart, fstart] = 1
            Dx[fstart - 1, fstart - 1] = 1
            Dx[fstart - 1, fstart] = 0
            Dx[fstart, fstart - 1] = 0
        
        D = block_diag(Dx, 0)
        
        if options['AR_lambda'] != -1:
            if len(options['smooth_lambda']) == 1:
                options['smooth_lambda'] = np.tile(options['smooth_lambda'][0], [trans_w.shape[0] - 1, trans_w.shape[1]])
                options['smooth_lambda'][:, options['AR_vec']] = options['AR_lambda']
            
            grad_regularization[np.setdiff1d(np.arange(num_states), state_num), :] = grad_regularization[np.setdiff1d(np.arange(num_states), state_num), :] + options['smooth_lambda'] * (np.matmul(D, trans_w[np.setdiff1d(np.arange(num_states), state_num), :].T)).T
            value_regularization = value_regularization + np.sum(np.sum(np.power((options['smooth_lambda'] / 2) * (np.matmul(D, trans_w[np.setdiff1d(np.arange(num_states), state_num), :].T)).T, 2), axis = 0), axis = 0)
        else:
            grad_regularization[np.setdiff1d(np.arange(num_states), state_num), :] = grad_regularization[np.setdiff1d(np.arange(num_states), state_num), :] + options['smooth_lambda'] * (np.matmul(D, trans_w[np.setdiff1d(np.arange(num_states), state_num), :].T)).T
            value_regularization = value_regularization + np.sum(np.sum(np.power((options['smooth_lambda'] / 2) * (np.matmul(D, trans_w.T)).T, 2), axis = 0), axis = 0)
    
    if this_lambda != 0:
        if options['AR_lambda'] != -1:
            grad_regularization = grad_regularization + [this_lambda * trans_w[:, options['stim_vec']], options['AR_lambda'] * trans_w[:, options['AR_vec']]]
            value_regularization = value_regularization + (this_lambda / 2) * np.sum(np.sum(np.power(trans_w[:, options['stim_vec']], 2), axis = 0), axis = 0) + (options['AR_lambda'] / 2) * np.sum(np.sum(np.power(trans_w[:, options['AR_vec']], 2), axis = 0), axis = 0)
        else:
            grad_regularization = grad_regularization + this_lambda * trans_w
            value_regularization = value_regularization + (this_lambda/2) * np.sum(np.sum(np.power(trans_w, 2), axis = 0), axis = 0)

    all_grad = -all_grad / total_T + grad_regularization
    all_value = -all_value / total_T + value_regularization

    if all_value < 0:
        print('Why oh why oh why negative values!')
        
    all_grad = np.reshape(all_grad.T, (all_grad.shape[0] * all_grad.shape[1]), order = 'F')
        
    return all_value, all_grad