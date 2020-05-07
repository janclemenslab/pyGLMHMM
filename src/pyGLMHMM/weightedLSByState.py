import numpy as np

def _weighted_LS_by_state(these_stim, these_symb, these_gamma, t_size):
    num_fits = 10
    num_sub_data = 0.3
    num_bins = these_stim.shape[1]
    num_states = these_gamma.shape[0]

    out_weights = np.zeros((num_states, num_bins))
    out_std = np.zeros((num_states,num_bins))

    for states in range(0, num_states):
        tik = np.power(np.diag(np.ones((num_bins, 1))) * t_size, 2)

        n_weights = np.zeros((num_fits, num_bins))
        
        for n_repeats in range(0, num_fits):
            use_stim = np.random.permutation(these_stim.shape[0])
            use_stim = use_stim[0:np.round(len(use_stim) * num_sub_data)]
            W = np.diag(these_gamma[states, use_stim])

            Z = np.power(np.matmul(these_stim[use_stim, :].T, np.matmul(W, these_stim[use_stim, :])) + tik, -1)
            STA = np.matmul(these_stim[use_stim, :].T, np.matmul(W, these_symb[use_stim]))
            n_weights[n_repeats, :] = np.matmul(Z, STA)

        out_weights[states, :] = np.mean(n_weights)
        out_std[states, :] = np.std(n_weights)
    
    return out_weights, out_std