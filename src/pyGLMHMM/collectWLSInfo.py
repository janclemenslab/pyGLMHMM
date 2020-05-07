import numpy as np

def _collect_WLS_info(stim):
    # I should compute the sigma of the fit for later use as well

    num_states = stim[0]['num_states']
    num_total_bins = stim[0]['num_total_bins']

    total_emit = 0
    
    for trial in range(0, len(stim)):
        total_emit = total_emit + np.sum(stim[trial]['good_emit'], axis = 0)

    these_stim= np.zeros((total_emit, num_total_bins))
    these_symb = np.zeros((total_emit, 1))
    these_gamma = np.zeros((num_states, total_emit, 1))
     
    ind = 0
    
    for trial in range(0, len(stim)):
        good_emit = stim[trial]['good_emit']
        T = np.sum(good_emit, axis = 0)

        these_stim[ind + range(0, T), :] = stim[trial]['data'][:, good_emit].T
        these_symb[ind + range(0, T)] = stim[trial]['symb'][good_emit]
        these_gamma[:, ind + range(0, T)] = stim[trial]['gamma'][:, good_emit]

        ind = ind + T
    
    return these_stim, these_symb, these_gamma