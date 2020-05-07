import numpy as np

def _emit_learning_stats(emit_w, stim, state_num, options):
    # emit_w are the weights that we are learning: in format states x weights
    # stim is a list/dictionary with each stimulus (stim[]['data']) and the probability transition functions (stim[]['gamma'] and stim[]['xi'])
    
    # http://www.ism.ac.jp/editsec/aism/pdf/044_1_0197.pdf

    num_states = stim[0]['num_states']
    num_bins = stim[0]['data'].shape[0]
    
    emit_w = np.reshape(emit_w, (num_bins, num_states), order = 'F').T

    all_hess = np.zeros((num_states, num_states, num_bins))
    all_value = 0
    total_T = 0

    this_lambda = options['emit_lambda']

    for trial in range(0, len(stim)):
        total_T = total_T + stim[trial]['data'].shape[1]

    for trial in range(0, len(stim)):

        T = stim[trial]['data'].shape[1]
        # Convert into states x bins x time and sum across bins
        filtpower = np.reshape(np.sum(np.reshape(np.tile(np.expand_dims(emit_w, axis = 2), (1, 1, T)), (num_states, num_bins, T), order = 'F') * np.tile(np.reshape(stim[trial]['data'], (1, num_bins, T), order = 'F'), (num_states, 1, 1)), axis = 1), (num_states, T), order = 'F')
        # Hack: change this...
        filtpower[filtpower > 600] = 600
        # Now filtpower is states x time

        value = stim[trial]['gamma'][state_num, :] * -np.log(1 + np.sum(np.exp(filtpower), axis = 0))

        norm = np.tile(1 + np.sum(np.exp(filtpower), axis = 0), (num_states, 1))
        hess_t = (np.exp(2 * filtpower) - np.exp(filtpower) * norm) / np.power(norm, 2)
        hess_t[np.isnan(hess_t)] = 0
        hess = np.zeros((num_states, num_states, num_bins))

        for i in range(0, filtpower.shape[0]):
            hess[i, i, :] = np.sum(np.tile(stim[trial]['gamma'][state_num, :], (num_bins, 1)) * np.tile(hess_t[i, :], (stim[trial]['data'].shape[0], 1)) * np.power(stim[trial]['data'], 2), axis = 1)
            
            for j in range(0, filtpower.shape[0]):
                if j == i:
                    continue
                
                hess[i, j, :] = np.sum(np.tile(stim[trial]['gamma'][state_num, :], (num_bins, 1)) * np.tile(np.exp(filtpower[i, :]) * np.exp(filtpower[j, :]) / np.power(norm[i, :], 2), (stim[trial]['data'].shape[0], 1)) * np.power(stim[trial]['data'], 2), axis = 1)

            value[stim[trial]['emit'] == i] = value[stim[trial]['emit'] == i] + stim[trial]['gamma'][state_num, stim[trial]['emit'] == i] * filtpower[i, stim[trial]['emit'] == i]
        
        value = np.sum(value, axis = 0)

        all_hess = all_hess + hess
        all_value = all_value + value

    if options['auto_anneal'] == True:
        all_hess = -all_hess / np.power(total_T, 2) + this_lambda
        all_value = -all_value / total_T + (this_lambda / 2) * np.sum(np.sum(np.power(emit_w, 2), axis = 0), axis = 0)
    else:
        all_hess = -all_hess + this_lambda
        all_value = -all_value / total_T + (this_lambda / 2) * np.sum(np.sum(np.power(emit_w, 2), axis = 0), axis = 0)

    for i in range(0, all_hess.shape[2]):
        all_hess[:, :, i] = np.power(all_hess[:, :, i], -1)

    if np.any(np.isnan(all_hess)) or np.any(np.isnan(all_value)):
        print('WTF! SOMETHING BAD HAPPENED! ISNAN! EEEE!')

    out_hess = np.zeros((all_hess.shape[0], all_hess.shape[2]))
    for i in range(0, all_hess.shape[0]):
        out_hess[i, :] = all_hess[i, i, :]
    
    all_hess = np.reshape(out_hess.T, (out_hess.shape[0] * out_hess.shape[1], 1), order = 'F')

    return all_value, all_hess