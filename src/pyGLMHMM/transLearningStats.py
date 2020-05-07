import numpy as np

def _trans_learning_stats(trans_w, stim, state_num, options):
    # trans_w are the weights that we are learning: in format states x weights
    # stim is a list/dictionary with each stimulus (stim[]['data']) and the probability transition functions (stim[]['gamma'] and stim[]['xi'])

    num_states = stim[0]['num_states']
    num_bins = stim[0]['data'].shape[0]

    trans_w = np.reshape(trans_w, (num_bins, num_states), order = 'F').T

    all_hess = np.zeros((num_states - 1, num_states - 1, num_bins))
    all_value = 0
    total_T = 0

    this_lambda = options['trans_lambda']

    for trial in range(0, len(stim)):
        total_T = total_T + stim[trial]['data'].shape[1]

    set_order = np.setdiff1d(np.arange(0, num_states), state_num)
    
    for trial in range(0, len(stim)):
        T = stim[trial]['data'].shape[1] - 1
        
        # Use data from 1:end-1 or 2:end?
        filtpower = np.sum(np.tile(np.expand_dims(trans_w, axis = 2), (1, 1, T)) * np.tile(np.reshape(stim[trial]['data'][:, 1:], (1, num_bins, T), order = 'F'), (num_states, 1, 1)), axis = 1)
        # Now filtpower is states x time

        value = -stim[trial]['gamma'][state_num, 0:-1] * np.log(1 + np.sum(np.exp(filtpower[set_order, :]), axis = 0))
        norm = 1 + np.sum(np.exp(filtpower[set_order, :]), axis = 0)

        hess = np.zeros((num_states - 1, num_states - 1, num_bins))
        
        # We have to 'reindex' because we now have a states-1 x states-1 matrix, where the diagonal is whichever state filter we are currently interested in...
        data_vec = np.power(np.reshape(stim[trial]['data'][:, 1:], (1, num_bins, T), order = 'F'), 2)
        
        for i in range(0, len(set_order)):
            for j in range(0, len(set_order)):
                if i != j:
                    hess[i, j, :] = np.sum(np.tile(np.reshape(stim[trial]['gamma'][state_num, 0:-1] * np.exp(filtpower[set_order[i], :]) * np.exp(filtpower[set_order[j], :]) / np.power(norm, 2), (1, 1, T), order = 'F'), (1, num_bins, 1)) * data_vec, axis = 2)
                else:
                    hess[i, i, :] = np.sum(np.tile(np.reshape(stim[trial]['gamma'][state_num, 0:-1] * (norm * np.exp(filtpower[set_order[i], :]) - np.exp(2 * filtpower[set_order[i], :])) / np.power(norm, 2), (1, 1, T), order = 'F'), (1, num_bins, 1)) * data_vec, axis = 2)

        # I probably don't need to rescale here because that happens naturally but... oh well!
        all_hess = all_hess + hess
        all_value = all_value + np.sum(value, axis = 0)

    all_hess = -all_hess + this_lambda
    all_value = -all_value / total_T + (this_lambda / 2) * np.sum(np.sum(np.power(trans_w, 2), axis = 0), axis = 0)

    for i in range(0, all_hess.shape[2]):
        all_hess[:, :, i] = np.power(all_hess[:, :, i], -1)

    if all_value < 0:
        print('Why oh why oh why!')
        
    out_hess = np.zeros((num_states, all_hess.shape[2]))
    for i in range(0, len(set_order)):
        out_hess[set_order[i], :] = all_hess[i, i, :]
    
    out_hess[state_num, :] = 0
    all_hess = np.reshape(out_hess.T, (out_hess.shape[0] * out_hess.shape[1], 1), order = 'F')

    return all_value, all_hess