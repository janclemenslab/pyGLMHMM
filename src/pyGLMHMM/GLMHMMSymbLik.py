import numpy as np

def _GLMHMM_symb_lik(emit_w, X_trial, y_trial):
    num_states = emit_w.shape[0]
    num_emissions = emit_w.shape[1]
    
    # Put the stimulus (X_trial) in a different format for easier multiplication
    X_trial_mod = np.tile(np.reshape(X_trial, (1, 1, X_trial.shape[0], X_trial.shape[1]), order = 'F'), (num_states, num_emissions, 1, 1))
    symb_lik = np.zeros((emit_w.shape[0], len(y_trial)))

    # Likelihood is exp(k*w) / (1 + sum(exp(k*w)))
    for t in range(0, len(y_trial)):
        symb_lik[:, t] = 1 / (1 + np.sum(np.exp(np.sum(emit_w * X_trial_mod[:, :, :, t], axis = 2)), axis = 1))

        # If the emission symbol is 0, we have 1 on the numerator otherwise exp(k*w)
        if y_trial[t] != 0:
            if emit_w.shape[1] == 1:
                symb_lik[:, t] = symb_lik[:, t] * np.squeeze(np.exp(np.sum(np.expand_dims(emit_w[:, int(y_trial[t]) - 1, :] * X_trial_mod[:, int(y_trial[t]) - 1, :, t], axis = 1), axis = 2)))
            else:
                symb_lik[:, t] = symb_lik[:, t] * np.exp(np.sum(emit_w[:, int(y_trial[t]) - 1, :] * X_trial_mod[:, int(y_trial[t]) - 1, :, t], axis = 2))

        if np.any(np.isnan(symb_lik[:, t])):
            print('Oh dear!')

    return symb_lik