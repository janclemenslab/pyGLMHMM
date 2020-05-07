import numpy as np

def _emit_generate(emit_w, trans_w, data, symb, options):
    # emit_w are the weights that we are learning: in format states x weights
    # trans_w are the weights that we are learning: in format states x weights

    feature_bins = 120

    options['use_AR']
    
    # We are going to have to hand-code some features here for a bit...
    if options['use_AR'] == True:
        p2_mean = np.mean(symb == 3, axis = 0)

        if p2_mean == 0:
            s_start = emit_w.shape[2] - 1 - feature_bins * 3
            p1_start = emit_w.shape[2] - 1 - feature_bins * 2
            a_start = emit_w.shape[2] - 1 - feature_bins

            s_score = np.unique(data[s_start - 1 + np.arange(0, feature_bins), :])
            p1_score = np.unique(data[p1_start - 1 + np.arange(0, feature_bins), :])
            a_score = np.unique(data[a_start - 1 + np.arange(0, feature_bins), :])
        
        else:
            s_start = emit_w.shape[2] - 1 - feature_bins * 4
            p1_start = emit_w.shape[2] - 1 - feature_bins * 3
            p2_start = emit_w.shape[2] - 1 - feature_bins * 2
            a_start = emit_w.shape[2] - 1 - feature_bins

            s_score = np.unique(data[s_start - 1 + np.arange(0, feature_bins), :])
            p1_score = np.unique(data[p1_start - 1 + np.arange(0, feature_bins), :])
            p2_score = np.unique(data[p2_start - 1 + np.arange(0, feature_bins), :])
            a_score = np.unique(data[a_start - 1 + np.arange(0, feature_bins), :])

    num_states = trans_w.shape[0]
    num_emissions = emit_w.shape[1]
    num_bins = trans_w.shape[2]
    T = data.shape[1]

    output = np.zeros((T))
    state = np.zeros((T))

    p_sample = np.random.rand(T)
    p_sample_state = np.random.rand(T)
    s_guess = np.zeros((num_states))
    
    for s1 in range(0, num_states):
        s_guess[s1] = 1 / (1 + np.sum(np.exp(np.sum(np.reshape(trans_w[s1, np.setdiff1d(np.arrange(0, num_states), s1), :], (num_states - 1, num_bins), order = 'F') * np.tile(data[:, 1].T, (num_states - 1, 1)), axis = 1)), axis = 0))

    tmp = np.argwhere(s_guess == np.max(s_guess))
    # Whatever, I guess this is just random at this point...
    state[0] = tmp[0]
    
    for t in range(0, T - feature_bins):
        filtpower = np.exp(np.sum(emit_w[state[t], :, :] * np.tile(np.reshape(data[:, t].T, (1, 1, data.shape[0]), order = 'F'), (1, num_emissions, 1)), axis = 2)).T
        likelihood2 = [[0], filtpower / (1 + np.sum(filtpower, axis = 0))]
        
        out_symb = np.argwhere(np.cumsum(likelihood2, axis = 0) < p_sample[t])
        out_symb = out_symb[-1]
        if out_symb == len(likelihood2):
            out_symb = 0
        
        output[t] = out_symb
        
        if options['use_AR'] == True:
            for b in range(0, feature_bins):
                
                if output[t] == 1:
                    data[p1_start + b - 1, t + b] = p1_score[1]
                else:
                    data[p1_start + b - 1, t + b] = p1_score[0]
                
                if output[t] == 2:
                    data[s_start + b - 1, t + b] = s_score[1]
                else:
                    data[s_start + b - 1, t + b] = s_score[0]
                
                if p2_mean != 0:
                    if output[t] == 3:
                        data[p2_start + b - 1, t + b] = p2_score[1]
                    else:
                        data[p2_start + b - 1, t + b] = p2_score[0]

                if output[t] == 0:
                    data[a_start + b - 1, t + b] = a_score[0]
                else:
                    data[a_start + b - 1, t + b] = a_score[1]

        filtpower = np.exp(np.sum(np.reshape(trans_w[state[t], np.setdiff1d(np.arange(0, num_states), state[t]), :], (num_states - 1, num_bins), order = 'F') * np.tile(data[:, t].T, (num_states - 1, 1)), axis = 2))
        
        ind = 0
        for s1 in np.setdiff1d(np.arrane(0, num_states), state[t]):
            ind = ind + 1
            s_guess[s1] = filtpower[ind] / (1 + np.sum(filtpower, axis = 0))
            
        s_guess[state[t]] = 1 / (1 + np.sum(filtpower, axis = 0))
        
        tmp = np.anywhere(np.cumsum([0, s_guess], axis = 0) < p_sample_state[t])
        
        state[t + 1] = tmp[-1]

    return output, state