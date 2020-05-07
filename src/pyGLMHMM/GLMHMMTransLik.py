import numpy as np
import scipy.misc

def _GLMHMM_trans_lik(trans_w, X_trial):
    T = X_trial.shape[1]
    num_states = trans_w.shape[0]
    num_total_bins = trans_w.shape[2]
    trans_lik = np.zeros((num_states, num_states, T))
    
    for i in range(0, num_states):
        filtpower = np.sum(np.tile(np.reshape(trans_w[i, :, :], (num_states, num_total_bins, 1), order = 'F'), (1, 1, T)) * np.tile(np.reshape(X_trial, (1, X_trial.shape[0], T), order = 'F'), (num_states, 1, 1)), axis = 1)

        if num_states == 1:
            filtpower = filtpower.T
        
        # There is no filter for going from state i to state i
        filtpower[i, :] = 0
        for j in range(0, num_states):
            trans_lik[i, j, :] = np.exp(filtpower[j, :] - scipy.misc.logsumexp(filtpower, axis = 0))

    return trans_lik