import numpy as np

from generatePosteriorNStep import _generate_posterior_nstep
from scipy.sparse import spdiags
from scipy.linalg import block_diag

def _emit_multistep_learning_fun(emit_w, stim, state_num, options):    
    # emit_w are the weights that we are learning: in format states x weights
    # stim is a list/dictionary with each stimulus (stim[]['data']) and the probability emission functions (stim[]['gamma'] and stim[]['xi'])
    
    # I will have to do something to make this more generic to work with other formats   
    num_steps = options['num_steps']
    num_samples = options['num_samples']
    
    num_states = stim[0]['num_states']
    num_total_bins = stim[0]['num_total_bins']
    # states x bins
    emit_w = np.reshape(emit_w, (num_total_bins, num_states), order = 'F').T

    all_grad = np.zeros((num_states, num_total_bins))
    all_value = 0
    total_T = 0

    this_lambda = options['emit_lambda']
    
    for trial in range(0, len(stim)):
        total_T = total_T + stim[trial]['data'].shape[1]
        
    for trial in range(0, len(stim)):
        # Basically, for each step we are looking ahead, we are going to generate a sample and then use that to calculate the lookahead likelihood
        # Since we are using large amounts of data here, we can get away with only using one sample (I think!)
        
        # I might have to use ADAM for SGD?
        # https://www.mathworks.com/matlabcentral/fileexchange/61616-adam-stochastic-gradient-descent-optimization
        for sample in range(0, num_samples):
            
            new_stim = stim[trial]['data']
                
            # Two steps:
            # First, find the likelihood of the actual data at STEPs away
            # Second, find the likelihood of all generated data...

            T = new_stim.shape[1]
            # Convert into states x bins x time and sum across bins
            filtpower = np.reshape(np.sum(np.reshape(np.tile(np.expand_dims(emit_w, axis = 2), (1, 1, T)), (num_states, num_total_bins, T), order = 'F') * np.tile(np.reshape(new_stim, (1, num_total_bins, T), order = 'F'), (num_states, 1, 1)), axis = 1), (num_states, T), order = 'F')
            # Now filtpower is states x time

            value = stim[trial]['gamma'][state_num, :] * -np.log(1 + np.sum(np.exp(filtpower), axis = 0))
            tgrad = -np.exp(filtpower) / np.tile(1 + np.sum(np.exp(filtpower), axis = 0), (num_states))

            for i in range(0, filtpower.shape[0]):
                tgrad[i, stim[trial]['emit'] == i] = 1 + tgrad[i, stim[trial]['emit'] == i]
                value[stim[trial]['emit'] == i] = value[stim[trial]['emit'] == i] + stim[trial]['gamma'][state_num, stim[trial]['emit'] == i] * filtpower[i, stim[trial]['emit'] == i]
            
            value = np.sum(value, axis = 0)
            tgrad = tgrad * np.tile(stim[trial]['gamma'][state_num, :], (num_states))

            tgrad = np.sum(np.tile(np.reshape(tgrad, (num_states, 1, T), order = 'F'), (1, num_total_bins, 1)) * np.tile(np.reshape(new_stim, (1, num_total_bins , T), order = 'F'), (num_states, 1, 1)), axis = 2)

            all_grad = all_grad + tgrad
            all_value = all_value + value
            
            [new_value, new_grad] = _generate_posterior_nstep(stim[trial]['data'], stim[trial]['emit'], num_steps - 1, emit_w, stim[trial]['gamma'][state_num, :])
            
            for c_num in range(0, num_steps - 1):
                all_grad = all_grad + new_grad[c_num]
                all_value = all_value + np.sum(new_value[c_num], axis = 0)

    # Implement smoothing: block matrix that is lambda_2 * [[1,-1,...],[-1,2,-1,...],[0,-1,2,-1,0,...]]
    # I need to make sure that this matrix takes into account the boundary size...   
    if options['auto_anneal'] == True:
        all_grad = all_grad / total_T + this_lambda * emit_w
        all_value = -all_value / total_T + (this_lambda / 2) * np.sum(np.sum(np.power(emit_w, 2), axis = 0), axis = 0)
    
    elif options['L2_smooth'] == True:
        all_grad = -all_grad / total_T + this_lambda * emit_w
        all_value = -all_value / total_T + (this_lambda / 2) * np.sum(np.sum(np.power(emit_w, 2), axis = 0), axis = 0)

        Dx1 = spdiags((np.ones(emit_w.shape[1] - 1, 1) * np.array([-1, 1])).T, np.array([0, 1]), emit_w.shape[1] - 1 - 1, emit_w.shape[1] - 1).toarray()
        Dx = np.matmul(Dx1.T, Dx1)
        
        for fstart in range(options['num_filter_bins'], emit_w.shape[1] - 1, options['num_filter_bins']):
            Dx[fstart, fstart] = 1
            Dx[fstart - 1, fstart - 1] = 1
            Dx[fstart - 1, fstart] = 0
            Dx[fstart, fstart - 1] = 0
            
        D = block_diag(Dx, 0)

        all_grad = all_grad + options['smooth_lambda'] * (np.matmul(D, emit_w.T)).T
        all_value = all_value + (options['smooth_lambda'] / 2) * np.sum(np.sum(np.power(np.matmul(D, emit_w.T), 2), axis = 0), axis = 0)
    
    else:
        all_grad = -all_grad / total_T + this_lambda * emit_w
        all_value = -all_value / total_T + (this_lambda / 2) * np.sum(np.sum(np.power(emit_w, 2), axis = 0), axis = 0)

    if np.any(np.isnan(all_grad)) or np.any(np.isnan(all_value)):
        print('WTF! SOMETHING BAD HAPPENED! ISNAN! OH!')
 
    all_grad = np.reshape(all_grad.T, (all_grad.shape[0] * all_grad.shape[1]), order = 'F')
    
    return all_value, all_grad