import math
import numpy as np

def _GLMHMM_analog_lik(analog_emit_w, X, y_analog, num_analog_emit):
    # Create lists to store data in
    num_analog_params = y_analog[0].shape[0]
    num_total_bins = analog_emit_w.shape[2]
    num_states = analog_emit_w.shape[0]
    
    residual_bag = list(range(0, num_analog_params))               
    for analog_num in range(num_analog_params - 1, -1, -1):
        
        residual_bag[analog_num] = np.zeros((num_states, num_analog_emit[analog_num]))
        analog_residuals = []
        for trial in range(0, len(y_analog)):
            analog_residuals[trial] = np.zeros((y_analog[trial].shape[0], num_states, y_analog[trial].shape[1]))

    # Compute the residuals for the predictions in each state and each analog variable
    analog_ind = np.zeros((num_analog_params, 1))
    for trial in range(0, len(y_analog)):
        for analog_num in range(0, num_analog_params):
            good_emit = not np.isnan(y_analog[trial][analog_num, :])
            T = np.sum(good_emit, axis = 0)

            this_stim = X[trial][:, good_emit]
            this_stim = np.tile(np.reshape(this_stim, (1, num_total_bins, T), order = 'F'), (num_states, 1, 1))

            prediction = np.sum(this_stim * np.tile(np.reshape(analog_emit_w[:, analog_num, :], (num_states, num_total_bins, 1)), (1, 1, T)), axis = 1)
            analog_residuals[trial][analog_num, :, good_emit] = np.tile(np.expand_dims(y_analog[trial][analog_num, good_emit], axis = 1), (num_states, 1)) - np.reshape(prediction, (num_states, T))

            residual_bag[analog_num][:, analog_ind(analog_num) + 1:analog_ind(analog_num) + T] = analog_residuals[trial][analog_num, :, good_emit]
            if np.sum(np.isnan(analog_residuals[trial]), axis = 0) > 0:
                print('Shit!')

            analog_ind[analog_num] = analog_ind[analog_num] + T

    # Compute the variance in the predictions
    analog_var = np.zeros((num_states, num_analog_params))
    Z_analog = np.zeros((num_states, num_analog_params))
    for analog_num in range(0, num_analog_params):
        for ss in range(0, num_states):
            analog_var[ss, analog_num] = np.var(residual_bag[analog_num][ss, :])
        
        Z_analog[:, analog_num] = np.real(np.sqrt(2 * math.pi * analog_var[:, analog_num]))

    # Hack; because a normalization constant below 1 can seriously screw things up in the likelihood
    if np.any(Z_analog < 1):
        Z_analog = Z_analog / np.max(Z_analog[Z_analog < 1])

    # Now that we know the distribution that our predictions are coming from, compute the likelihoods
    analog_lik = []
    for trial in range(0, len(y_analog)):
        analog_lik[trial] = np.zeros((num_analog_params, num_states, y_analog[trial].shape[1]))
        for analog_num in range(0, num_analog_params):
            for ss in range(0, num_states):
                analog_lik[trial][analog_num, ss, :] = 1 / Z_analog[ss, analog_num] * np.exp((-1) * np.power(analog_residuals[trial][analog_num, ss, :], 2) / (2 * analog_var[ss, analog_num]))
    
    return analog_lik