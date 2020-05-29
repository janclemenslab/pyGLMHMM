import numpy as np

from .GLMHMMSymbLik import _GLMHMM_symb_lik
from .GLMHMMAnalogLik import _GLMHMM_analog_lik
from .GLMHMMTransLik import _GLMHMM_trans_lik
from .computeTrialExpectation import _compute_trial_expectation 

def _HMMGLM_likelihoods(symb, emit_w, trans_w, stim, analog_emit_w, analog_symb, options):
    
    total_trials = max(len(symb), len(analog_symb))  # How many different trials are we fitting?
    num_states = max(emit_w.shape[0], analog_emit_w.shape[0], trans_w.shape[0])

    if options['analog_flag'] == True:
        num_analog_params = analog_symb[0].shape[0]
        num_analog_emit = np.zeros(num_analog_params)
    else:
        num_analog_params = 0
    
    prior = []
    gamma = []
    xi = []
    
    for trial in range(0, total_trials):
        prior.append(np.ones(num_states) / num_states)     # Is this good?!?!
        
        if options['analog_flag'] == True:
            for analog_num in range(0, num_analog_params):
                num_analog_emit[analog_num] = num_analog_emit[analog_num] + np.nansum(analog_symb[trial][analog_num, :], axis = 0)

            gamma.append(np.ones((num_states, analog_symb[trial].shape[1])))
        else:
            if len(symb[trial].shape) > 1:
                gamma.append(np.ones((num_states, symb[trial].shape[1])))
            else:
                gamma.append(np.ones((num_states, 1)))
                
        gamma[trial] = gamma[trial] / np.tile(np.sum(gamma[trial], axis = 0), (num_states, 1))
        
        xi.append([])
        
    symb_lik = []
    analog_lik = []
    trans_lik = []
    
    for trial in range(0, total_trials):
        if options['symb_exists'] == True:
            if options['GLM_emissions'] == True:
                symb_lik.append(_GLMHMM_symb_lik(emit_w, stim[trial], symb[trial]))
            else:
                symb_lik.append(_GLMHMM_symb_lik(emit_w[:, :, -1], stim[trial][-1, :], symb[trial]))
                        
        trans_lik.append(_GLMHMM_trans_lik(trans_w, stim[trial]))
    
    if options['analog_flag'] == True:
        analog_lik_ = _GLMHMM_analog_lik(analog_emit_w, stim, analog_symb, num_analog_emit)
    else:
        analog_lik_ = []
        
    for trial in range(0, total_trials):
         # Maybe first compute likelihoods for the symbols?
         if options['analog_flag'] == True and options['symb_exists'] == True:
             emit_likelihood = symb_lik[trial] * np.prod(analog_lik[trial], axis = 0)
         elif options['symb_exists'] == True:
             emit_likelihood = symb_lik[trial]
         elif options['analog_flag'] == True:
             emit_likelihood = np.prod(analog_lik_[trial], axis = 0)

         # Things get funky if the likelihood is exactly 0
         emit_likelihood[emit_likelihood < np.finfo(float).eps * 1e3] = np.finfo(float).eps * 1e3    # Do we need this?
         
         prior[trial], gamma[trial], xi[trial] = _compute_trial_expectation(prior[trial], emit_likelihood, trans_lik[trial])
    
    return symb_lik, trans_lik, analog_lik, gamma, xi