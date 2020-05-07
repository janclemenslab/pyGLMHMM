import numpy as np

def _emit_likelihood(emit_w, stim, state_num):   
    # trans_w are the weights that we are learning: in format states x weights
    # stim is a list/dictionary with each stimulus (stim[]['data']) and the probability transition functions (stim[]['gamma'] and stim[]['xi'])

    # all_value = <log likelihood>
    # all_lik = <pure likelihood>
    # all_lik2 = <MAP decision f'n of likelihood>
    # all_lik3 = <likelihood | song type>
    # all_lik4 = <MAP decision f'n | song type>

    num_states = stim[0]['num_states']
    num_bins = stim[0]['data'].shape[0]
    
    emit_w = np.reshape(emit_w, (num_bins, num_states), order = 'F').T

    all_value = 0
    
    all_lik1 = 0
    all_lik2 = 0
    all_lik3 = np.zeros((num_states + 1, num_states + 1))
    all_lik4 = np.zeros((num_states + 1, num_states + 1))
    
    total_T12 = 0
    total_T34 = np.zeros((num_states + 1, 1))

    for trial in range(0, len(stim)):
        total_T12 = total_T12 + stim[trial]['data'].shape[1]
        
        total_T34[-1] = total_T34[-1] + np.sum(stim[trial]['emit'] == 0, axis = 0)
        
        for i in range(0, num_states):
            total_T34[i] = total_T34[i] + np.sum(stim[trial]['emit'] == i, axis = 0)

    for trial in range(0, len(stim)):

        T = stim[trial]['data'].shape[1]
        # Convert into states x bins x time and sum across bins
        filtpower = np.reshape(np.sum(np.reshape(np.tile(np.expand_dims(emit_w, axis = 2), (1, 1, T)), (num_states, num_bins, T), order = 'F') * np.tile(np.reshape(stim[trial]['data'], (1, num_bins, T), order = 'F'), (num_states, 1, 1)), axis = 1), (num_states, T))
        # Now filtpower is states x time

        value = stim[trial]['gamma'][state_num, :] * -np.log(1 + np.sum(np.exp(filtpower), axis = 0))
        
        lik1 = 1 / (1 + np.sum(np.exp(filtpower), axis = 0))
        lik2 = np.tile(np.reshape(1 / (1 + np.sum(np.exp(filtpower), axis = 0)), (1, T), order = 'F'), (num_states + 1, 1))
        
        lik3 = np.zeros((filtpower.shape[0] + 1, filtpower.shape[0] + 1))
        lik3[-1, -1] = np.sum(lik1[stim[trial]['emit'] == 0] * stim[trial]['gamma'][state_num, stim[trial]['emit'] == 0], axis = 0)
        
        for j in range(0, filtpower.shape[0]):
            lik3[-1, j] = np.sum(lik1[stim[trial]['emit'] == 0] * np.exp(filtpower[j, stim[trial]['emit'] == 0]) * stim[trial]['gamma'][state_num, stim[trial]['emit'] == 0], axis = 0)
            lik3[j, -1] = np.sum(lik1[stim[trial]['emit'] == j] * stim[trial]['gamma'][state_num, stim[trial]['emit'] == j], axis = 0)

        for i in range(0, filtpower.shape[0]):
            value[stim[trial]['emit'] == i] = value[stim[trial]['emit'] == i] + stim[trial]['gamma'][state_num, stim[trial]['emit'] == i] * np.log(np.exp(filtpower[i, stim[trial]['emit'] == i]))
            
            for j in range(0, filtpower.shape[0]):
                lik3[i, j] = np.sum(lik1[stim[trial]['emit'] == i] * np.exp(filtpower[j, stim[trial]['emit'] == i]) * stim[trial]['gamma'][state_num, stim[trial]['emit'] == i], axis = 0)

            lik1[stim[trial]['emit'] == i] = lik1[stim[trial]['emit'] == i] * np.exp(filtpower[i, stim[trial]['emit'] == i])
            lik2[i, :] = lik2[i, :] * np.exp(filtpower[i, :])
        
        lik4 = np.zeros((filtpower.shape[0] + 1, filtpower.shape[0] + 1))
        
        for i in range(0, filtpower.shape[0]):
            lik4[i, -1] = np.sum((lik2[-1, stim[trial]['emit'] == i] == np.amax(lik2[:, stim[trial]['emit'] == i], axis = 0)) * stim[trial]['gamma'][state_num, stim[trial]['emit'] == i], axis = 0)
            lik4[-1, i] = np.sum((lik2[i, stim[trial]['emit'] == 0] == np.amax(lik2[:, stim[trial]['emit'] == 0], axis = 0)) * stim[trial]['gamma'][state_num, stim[trial]['emit'] == 0], axis = 0)
            
            for j in range(0, filtpower.shape[0]):
                lik4[i, j] = np.sum((lik2[j, stim[trial]['emit'] == i] == np.amax(lik2[:, stim[trial]['emit'] == i], axis = 0)) * stim[trial]['gamma'][state_num, stim[trial]['emit'] == i], axis = 0)
                
        lik4[-1, -1] = np.sum((lik2[-1, stim[trial]['emit'] == 0] == np.amax(lik2[:, stim[trial]['emit'] == 0], axis = 0)) * stim[trial]['gamma'][state_num, stim[trial]['emit'] == 0], axis = 0)
        
        lik2 = (lik1 == np.amax(lik2, axis = 0)) * stim[trial]['gamma'][state_num, :]
        lik1 = lik1 * stim[trial]['gamma'][state_num, :]

        value = np.sum(value, axis = 0)
        lik1 = np.sum(lik1, axis = 0)
        lik2 = np.sum(lik2, axis = 0)

        # I probably don't need to rescale here because that happens naturally but... oh well!
        all_value = all_value + value
        
        all_lik1 = all_lik1 + lik1
        all_lik2 = all_lik2 + lik2
        all_lik3 = all_lik3 + lik3
        all_lik4 = all_lik4 + lik4

    all_value = -all_value / total_T12
    
    all_lik1 = all_lik1 / total_T12
    all_lik2 = all_lik2 / total_T12
    all_lik3 = all_lik3 / np.tile(total_T34, (1, all_lik3.shape[1]))
    all_lik4 = all_lik4 / np.tile(total_T34, (1, all_lik4.shape[1]))

    if np.any(np.isnan(all_lik1)) or np.any(np.isnan(all_value)):
        print('WTF! SOMETHING BAD HAPPENED! ISNAN! AH!')
    
    return all_value, all_lik1, all_lik2, all_lik3, all_lik4