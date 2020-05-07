import numpy as np

def _generate_posterior_nstep(stim, emit, steps, emit_w, gamma):
    # Given that we want to generate STEPS worth of latent data and we know what is happening at STEPS+1, how do we sample from the posterior?
    # p(y_l | Y_t+m, Y_t) ~ p(Y_t+m | y_l, Y_t) * p(y_l | Y_t)
    # So for each possible y_l find the probability of generating that sample and the probability of seeing Y_t+m given that sample, then draw from (the normalized version of) this probability distribution
    # What if instead of going through all possible latents we just sample from them? And hope that on average we converge on the right distribution...
    # Then instead of needing 3^N possibilities for N steps away, we can just get away with some smaller number, e.g. N+1 or even just two options...
    # (If this works, I could use the same technique for the analog signal!)

    num_paths = 3
    symb = np.unique(emit)
    new_grad = []
    new_value = []

    # Start with step 1 being of length T then generate a stimulus and make the next set of values T-1
    # We will use this to predict T+1, meaning we only need to use emit(2:end)
    
    for ll in range(0, steps):
        if ll == 0:
            # At the first time step, we want to take the generate likelihoods, gradients, and values and remove one time bin
            # We only need to generate 1:end-1, because generating a synthetic value for the last bin will NEVER be used
            # (because there is no emission in the following bin that will use that latent prediction)
            old_stim = stim

            curr_vec = np.zeros((num_paths, ll, old_stim.shape[1] - 1))
            p_each = np.zeros((curr_vec.shape[0], old_stim.shape[1] - 1))
            auto_stim = np.zeros((curr_vec.shape[0], old_stim.shape[0], old_stim.shape[1] - 1))
            grad = np.zeros((curr_vec.shape[0], emit_w.shape[0], emit_w.shape[1], old_stim.shape[1] - 1))
            value = np.zeros((num_paths, old_stim.shape[1] - 1))

            for vv in range(0, num_paths):
                choice = np.floor(np.random.rand(1, old_stim.shape[1] - 1) * len(symb))

                p_new = _stim_likelihood(old_stim[:, 0:-1], emit_w)

                # Not sure either this or the value are quite right...
                grad[vv, :, :, :] = [[np.reshape(np.tile(((choice == 1) - p_new[1, :] * gamma[0:-ll]), (emit_w.shape[1], 1)) * old_stim[:, 0:-1], (1, emit_w.shape[1], old_stim.shape[1] - 1), order = 'F')],...
                    [np.reshape(np.tile(((choice == 2) - p_new[2, :] * gamma[0:-ll]), (emit_w.shape[1], 1)) * old_stim[:, 0:-1], (1, emit_w.shape[1], old_stim.shape[1] - 1), order = 'F')]]

                # I have to do that stupid indexing thing here...
                p_lik = p_new
                value[vv, choice[0:len(p_lik)] == 0] = gamma[choice[0:len(p_lik)] == 0] * np.log(p_lik[0, choice[0:len(p_lik)] == 0])
                value[vv, choice[0:len(p_lik)] == 1] = gamma[choice[0:len(p_lik)] == 1] * np.log(p_lik[1, choice[0:len(p_lik)] == 1])
                value[vv, choice[0:len(p_lik)] == 2] = gamma[choice[0:len(p_lik)] == 2] * np.log(p_lik[2, choice[0:len(p_lik)] == 2])

                p_each[vv, :] = p_lik[sub2ind(p_lik.shape, choice + 1, np.arange(0, len(p_lik)))]
                
                curr_vec[vv, 0, :] = choice
                
                # Now generate what the new stimulus would look like given this sequence of choices
                auto_stim[vv, :, :] = _make_new_stim_from_vec(old_stim[:, 0:-1], choice)
            
            # At the end of this, we want to save time by generating a sequence using the posterior
            [new_emit, real_lik, hold_lik] = _choose_new_stim(auto_stim, emit_w, p_each, emit)
            [new_grad_temp, new_value_temp] = _update_grad(hold_lik, real_lik, new_emit, auto_stim, grad, value, gamma, emit)
            
            new_grad.append(new_grad_temp)
            new_value.append(new_value_temp)
        else:
            # At every time step (after the first) we want to take the previous likelihoods, gradients, and values and remove one time bin            
            # At this time step, we no longer care about whatever is in the first bin: if we are generating ll+1 stimuli at this point in the algorithm, we will have to rely on ll bins!
            # So the first ll bins WON'T have latent stimuli but rather REAL stimuli

            old_stim = auto_stim[:, :, 1:]

            curr_vec = curr_vec[:, :, 0:-1]
            p_each_old = p_each[:, 0:-1]
            old_grad = grad[:, :, :, 0:-1]
            old_value = value[:, 1:-1]
 
            p_each = np.zeros((curr_vec.shape[0], old_stim.shape[2]))
            auto_stim = np.zeros((curr_vec.shape[0], old_stim.shape[1], old_stim.shape[2]))
            grad = np.zeros((curr_vec.shape[0], emit_w.shape[0], emit_w.shape[1], old_stim.shape[2]))
            value = np.zeros((num_paths, old_stim.shape[1]))
            
            for vv in range(0, num_paths):
                # Now we have taken the stimuli from the previous time step and shifted them into place as if they were already generated and are in the autocorr stimuli parameters:
                choice = np.floor(np.random.rand(1, old_stim.shape[2]) * len(symb))
                p_lik = _stim_likelihood(old_stim[vv, :, :], emit_w)

                value[vv, choice[0:len(p_lik)] == 0] = gamma[choice[0:len(p_lik)] == 0] * np.log(p_lik[0, choice[0:len(p_lik)] == 0]) + old_value[vv, choice[0:len(p_lik)] == 0]
                value[vv, choice[0:len(p_lik)] == 1] = gamma[choice[0:len(p_lik)] == 1] * np.log(p_lik[1, choice[0:len(p_lik)] == 1]) + old_value[vv, choice[0:len(p_lik)] == 1]
                value[vv, choice[0:len(p_lik)] == 2] = gamma[choice[0:len(p_lik)] == 2] * np.log(p_lik[2, choice[0:len(p_lik)] == 2]) + old_value[vv, choice[0:len(p_lik)] == 2]

                p_each[vv, :] = p_lik[sub2ind(p_lik.shape, choice + 1, np.arange(0, len(p_lik)))] * p_each_old[vv, :]

                grad[vv, :, :, :] = old_grad[vv, :, :, :] + [[np.reshape(np.tile(((choice == 1) - p_lik[1, :] * gamma[ll:-1]), (emit_w.shape[1], 1)) * old_stim[vv, :, :], (1, emit_w.shape[1], stim.shape[1]- ll), order = 'F')],...
                    [np.reshape(np.tile(((choice == 2) - p_lik[2, :] * gamma[ll:-1]), (emit_w.shape[1], 1)) * old_stim[vv, :, :], (1, emit_w.shape[1], stim.shape[1]- ll), order = 'F')]]

                curr_vec[vv, ll, :] = choice
                auto_stim[vv, :, :] = _make_new_stim_from_vec(old_stim[vv, :, :], choice)
            
            # At the end of this, we want to save time by generating a sequence using the posterior
            [new_emit, real_lik, hold_lik] = _choose_new_stim(auto_stim, emit_w, p_each, emit)
            [new_grad_temp, new_value_temp] = _update_grad(hold_lik, real_lik, new_emit, auto_stim, grad, value, gamma, emit)
            
            new_grad.append(new_grad_temp)
            new_value.append(new_value_temp)

    return  new_value, new_grad 

def _make_new_stim_from_vec(old_stim, symb):
    if old_stim.shape[0] > 602:
        symb0 = np.array(range(600, 630))
        symb1 = np.array(range(540, 600))
        symb2 = np.array(range(510, 540))
    else:
        symb0 = np.array(range(570, 600))
        symb1 = np.array(range(540, 570))
        symb2 = np.array(range(510, 540))
    
    new_stim = old_stim
    new_stim[[[symb0(0)], [symb1(0)], [symb2(0)]] + 1, 1:] = old_stim[[[symb0(0)], [symb1(0)], [symb2(0)]], 0:-1]

    new_stim[symb0(1), symb != 0] = np.amax(new_stim[symb0, :])
    new_stim[symb0(1), symb == 0] = np.amin(new_stim[symb0, :])

    new_stim[symb1(1), symb == 1] = np.amax(new_stim[symb1, :])
    new_stim[symb1(1), symb != 1] = np.amin(new_stim[symb1, :])

    new_stim[symb2(1), symb == 2] = np.amax(new_stim[symb2, :])
    new_stim[symb2(1), symb != 2] = np.amin(new_stim[symb2, :])

    return new_stim

def _stim_likelihood(stim, emit_w):
    T = stim.shape[1]
    num_bins = stim.shape[0]
    num_states = emit_w.shape[0]

    filtpower = np.reshape(np.sum(np.reshape(np.tile(np.expand_dims(emit_w, axis = 2), (1, 1, T)), (num_states, num_bins, T), order = 'F') * np.tile(np.reshape(stim, (1, num_bins, T), order = 'F'), (num_states, 1, 1)), axis = 1), (num_states, T))
    lik = [[np.ones((1, T))], np.exp(filtpower)] / np.tile(np.expand_dims(1 + np.sum(np.exp(filtpower), axis = 0), axis = 1), (emit_w.shape[0] + 1, 1))
    
    return lik, filtpower

def _choose_new_stim(auto_stim, emit_w, p_each, emit):
    steps = len(emit) - len(auto_stim)
    
    real_lik = np.zeros((auto_stim.shape[0], auto_stim.shape[2]))
    hold_lik = np.zeros((auto_stim.shape[0], emit_w.shape[0], auto_stim.shape[2]))
    
    for vv in range(0, auto_stim.shape[0]):
        # Now compute the likelihood of observing the ACTUAL emission
        lik_path = _stim_likelihood(auto_stim[vv, :, :], emit_w)
        real_lik[vv, :] = lik_path[sub2ind(lik_path.shape, emit[steps:].T + 1, np.arange(0, (len(emit) - steps)))]
        hold_lik[vv, :, :] = lik_path[1:2, :]
    
    new_lik = real_lik * p_each
    new_lik = new_lik / np.tile(np.expand_dims(np.sum(new_lik, axis = 0), axis = 1), (new_lik.shape[0], 1))
    new_emit = np.sum(np.tile(np.random.rand(1, auto_stim.shape[2]), (new_lik.shape[0], 1)) > np.cumsum(new_lik, axis = 0), axis = 0) + 1
    
    return new_emit, real_lik, hold_lik 

def _update_grad(hold_lik, real_lik, new_emit, auto_stim, grad, value, gamma, emit):
    steps = len(emit) - len(auto_stim)
    
    all_grad = [[-hold_lik[sub2ind(hold_lik.shape, new_emit, np.ones((1, len(new_emit)))), 0:len(new_emit)]], [-hold_lik[sub2ind(hold_lik.shape, new_emit, np.ones((1, len(new_emit))) + 1), 0:len(new_emit)]]]
    all_grad[0, emit[steps:] == 1] = 1 + all_grad[0, emit[steps:] == 1]
    all_grad[1, emit[steps:] == 2] = 1 + all_grad[1, emit[steps:] == 2]

    good_stim = np.zeros((auto_stim.shape[1], auto_stim.shape[2]))
    good_grad = np.zeros((grad.shape[1], auto_stim.shape[1], auto_stim.shape[2]))

    for vv in range(0, auto_stim.shape[0]):
        good_stim[:, new_emit == vv] = auto_stim[vv, :, new_emit == vv]
        good_grad[:, :, new_emit == vv] = grad[vv, :, :, new_emit == vv]

    new_grad = [[np.matmul(all_grad[0, :] * gamma[steps:], good_stim.T)], [np.matmul(all_grad[1, :] * gamma[steps:], good_stim.T)]]
    new_grad = new_grad + np.sum(good_grad, axis = 2)

    new_value = np.sum(value[sub2ind(value.shape, new_emit, np.arange(0, len(value)))], axis = 0) + np.sum(gamma[steps:] * np.log(real_lik[sub2ind(real_lik.shape, new_emit, np.arange(0, len(real_lik)))]), axis = 0)
    
    return new_grad, new_value

def sub2ind(array_shape, rows, cols):
    ind = rows * array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    return ind