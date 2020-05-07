import numpy as np

def _generate_next_step(stim, emit_w, num_states, num_bins):
    T = stim.shape[1]
    new_stim = stim
    
    filtpower = np.reshape(np.sum(np.reshape(np.tile(np.expand_dims(emit_w, axis = 2), (1, 1, T)), (num_states, num_bins, T), order = 'F') * np.tile(np.reshape(stim, (1, num_bins, T), order = 'F'), (num_states, 1, 1)), axis = 1), (num_states, T), order = 'F')

    p = np.concatenate((np.ones((1, T)), np.exp(filtpower)), axis = 0) / np.tile(1 + np.sum(np.exp(filtpower), axis = 0), (emit_w.shape[0] + 1, 1))

    new_emit = np.sum(np.tile(np.random.rand((1, T)), (emit_w.shape[0] + 1, 1)) < np.cumsum(p, axis = 0), axis = 0)

    new_stim[600, new_emit == 1] = np.max(np.max(new_stim[600:629, :]))
    new_stim[600, new_emit != 1] = np.min(np.min(new_stim[600:629, :]))

    new_stim[570, new_emit == 2] = np.max(np.max(new_stim[570:599, :]))
    new_stim[570, new_emit != 2] = np.min(np.min(new_stim[570:599, :]))

    new_stim[510, new_emit == 3] = np.max(np.max(new_stim[510:539, :]))
    new_stim[510, new_emit != 3] = np.min(np.min(new_stim[510:539, :]))
    
    return new_stim, new_emit