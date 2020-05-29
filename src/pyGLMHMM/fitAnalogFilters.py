import numpy as np

from .weightedLSByState import _weighted_LS_by_state
from .collectWLSInfo import _collect_WLS_info
from .waveletTransform import _w_corr
from .fastASD import _fast_ASD_weighted_group

def _fit_analog_filters(stim, analog_symb, gamma, xi, analog_emit_w, options, train_data):
    
    new_stim = []
    analog_emit_std = np.array([])
    num_analog_params = analog_symb[0].shape[0]
                    
    ar_corr1 = np.zeros((options['num_states'], num_analog_params))
    ar_corr2 = np.zeros(num_analog_params)

    if options['evaluate'] == True:                            
        for analog_num in range(0, num_analog_params):
            for trial in range(0, len(train_data)):
                new_stim.append({'gamma' : gamma[train_data[trial]], 'xi' : xi[train_data[trial]], 'num_states' : options['num_emissions']})

                new_stim[trial]['symb'] = analog_symb[train_data[trial]][analog_num, :]
                new_stim[trial]['good_emit'] = ~np.isnan(analog_symb[train_data[trial]][analog_num, :])
       
            [these_stim, these_symb, these_gamma] = _collect_WLS_info(new_stim)
            
            for states in range(0, options['num_states']):
                ar_corr1[states, analog_num] = _w_corr(these_stim * analog_emit_w[states, analog_num, :], these_symb, these_gamma[states, :].T)
            
            ar_corr2[analog_num] = np.sum(np.mean(these_gamma, axis = 1) * ar_corr1[:, analog_num], axis = 0)
        
    else:
        for analog_num in range(0, num_analog_params):
            for trial in range(0, len(train_data)):
                new_stim.append({'num_states' : options['num_emissions']})

                new_stim[trial]['symb'] = analog_symb[train_data[trial]][analog_num, :]
                new_stim[trial]['good_emit'] = ~np.isnan(analog_symb[train_data[trial]][analog_num, :])
                
            [these_stim, these_symb, these_gamma] = _collect_WLS_info(new_stim)

            # If more than this, loop until we have gone through all of them. How to deal with e.g. ~1k over this max? Overlapping subsets? Could just do e.g. 4 sub-samples, or however many depending on amount >15k
            max_good_pts = 15000
            num_analog_iter = np.ceil(these_stim.shape[0] / max_good_pts)
            
            if num_analog_iter > 1:
                analog_offset = (these_stim.shape[0] - max_good_pts) / (num_analog_iter - 1)
                iter_stim = np.zeros((num_analog_iter, 2))
                
                for nai in range(0, num_analog_iter):
                    iter_stim[nai, :] = np.floor(analog_offset * (nai - 1)) + [1, max_good_pts]
            else:
                iter_stim = [1, these_stim.shape[0]]

            randomized_stim = np.random.permutation(these_stim.shape[0])
            ae_w = np.zeros((num_analog_iter, options['num_states'], analog_emit_w.shape[2]))
            ae_std = np.zeros((num_analog_iter, options['num_states'], analog_emit_w.shape[2]))
            iter_weight = np.zeros((num_analog_iter, options['num_states']))
            
            for nai in range(0, num_analog_iter):
                use_stim = randomized_stim[iter_stim[nai, 0]:iter_stim[nai, 1]]
                
                for states in range(0, options['num_states']):
                    if options['use_ASD'] == True:
                        [out_weights, ASD_stats] = _fast_ASD_weighted_group(these_stim[use_stim, :], these_symb[use_stim], these_gamma[states, use_stim], [np.ones((np.round(these_stim.shape[1] / options['num_filter_bins']), 1)) * options['num_filter_bins'], [1]], 2)
                        ae_w[nai, states, :] = out_weights
                        ae_std[nai, states, :] = ASD_stats['L_post_diag']
                    else:
                        [out_weights, out_std] = _weighted_LS_by_state(these_stim[use_stim, :], these_symb[use_stim], these_gamma[states, use_stim], 10)
                        ae_w[nai, states, :] = out_weights
                        ae_std[nai, states, :] = out_std

                    iter_weight[nai, states] = np.sum(these_gamma[states, use_stim], axis = 0)
                    ar_corr1[states, analog_num] = 0
 
            for states in range(0, options['num_states']):
                analog_emit_w[states, analog_num, :] = np.sum(ae_w[:, states, :] * np.tile(np.expand_dims(np.expand_dims(iter_weight[:, states] / np.sum(iter_weight[:, states], axis = 0), axis = 1), axis = 2), (1, 1, analog_emit_w.shape[2])), axis = 0)
                analog_emit_std[states, analog_num, :] = np.sum(ae_std[:, states, :] * np.tile(np.expand_dims(np.expand_dims(iter_weight[:, states] / np.sum(iter_weight[:, states], axis = 0), axis = 1), axis = 2), (1, 1, analog_emit_std.shape[2])), axis = 0)

            ar_corr1[states, analog_num] = 0
            ar_corr2[analog_num] = 0
            
    return analog_emit_w, analog_emit_std, ar_corr1, ar_corr2