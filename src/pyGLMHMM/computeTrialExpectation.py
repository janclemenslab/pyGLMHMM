import copy
import numpy as np

def _compute_trial_expectation(prior, likelihood, transition):
    # Forward-backward algorithm, see Rabiner for implementation details
	# http://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf

    t = 0
    T = likelihood.shape[1]
    num_states = likelihood.shape[0]

    # E-step
    # alpha1 is the forward probability of seeing the sequence
    alpha1 = np.zeros((len(prior), T))
    alpha2 = np.zeros((len(prior), T))
    scale = np.zeros((len(prior), T))
    scale_a = np.ones((T, 1))
    score = np.zeros((T, 1))
    
    alpha1[:, 0] = prior * likelihood[:, 0]
    alpha1[:, 0] = alpha1[:, 0] / np.sum(alpha1[:, 0], axis = 0)
    scale[:, 0] = alpha1[:, 0]

    alpha2[:, 0] = prior

    for t in range(1, T):
        alpha1[:, t] = np.matmul(transition[:, :, t].T, alpha1[:, t - 1])
        scale[:, t] = alpha1[:, t] / np.sum(alpha1[:, t], axis = 0)
        alpha1[:, t] = alpha1[:, t] * likelihood[:, t]

        # Use this scaling component to try to prevent underflow errors
        scale_a[t] = np.sum(alpha1[:, t], axis = 0)
        alpha1[:, t] = alpha1[:, t] / scale_a[t]
        
        alpha2[:, t] = np.matmul(transition[:, :, t].T, alpha2[:, t - 1])
        alpha2[:, t] = alpha2[:, t] / np.sum(alpha2[:, t], axis = 0)
        score[t] = np.sum(alpha2[:, t] * likelihood[:, t], axis = 0)

    # beta is the backward probability of seeing the sequence
    beta = np.zeros((len(prior), T))	# beta(i, t) = Pr(O(t + 1:T) | X(t) = i)
    beta[:, -1] = np.ones(len(prior)) / len(prior)
    
    scale_b = np.ones((T, 1))
       
    for t in range(T - 2, -1, -1):
        beta[:, t] = np.matmul(transition[:, :, t + 1], (beta[:, t + 1] * likelihood[:, t + 1]))       
        scale_b[t] = np.sum(beta[:, t], axis = 0)
        beta[:, t] = beta[:, t] / scale_b[t]

    # If any of the values are 0, it's defacto an underflow error so set it to eps
    alpha1[alpha1 == 0] = np.finfo(float).eps
    beta[beta == 0] = np.finfo(float).eps

    # gamma is the probability of seeing the sequence, found by combining alpha and beta
    gamma = np.exp(np.log(alpha1) + np.log(beta) - np.tile(np.log(np.cumsum(scale_a, axis = 0)).T, (num_states, 1)) - np.tile(np.log(np.flip(np.cumsum(np.flip(scale_b, axis = 0), axis = 0), axis = 0)).T, (num_states, 1)))
    gamma[gamma == 0] = np.finfo(float).eps
    gamma = gamma / np.tile(np.sum(gamma, axis = 0), (num_states, 1))

    # xi is the probability of seeing each transition in the sequence
    xi = np.zeros((len(prior), len(prior), T - 1))
    transition2 = copy.copy(transition[:, :, 1:])

    for s1 in range(0, num_states):
        for s2 in range(0, num_states):
            xi[s1, s2, :] = np.log(likelihood[s2, 1:]) + np.log(alpha1[s1, 0:-1]) + np.log(transition2[s1, s2, :].T) + np.log(beta[s2, 1:]) - np.log(np.cumsum(scale_a[0:-1], axis = 0)).T - np.log(np.flip(np.cumsum(np.flip(scale_b[1:], axis = 0), axis = 0), axis = 0)).T
            xi[s1, s2, :] = np.exp(xi[s1, s2, :])

    xi[xi == 0] = np.finfo(float).eps
    
    # Renormalize to make sure everything adds up properly
    xi = xi / np.tile(np.expand_dims(np.expand_dims(np.sum(np.sum(xi, axis = 0), axis = 0), axis = 0), axis = 0), (num_states, num_states, 1))
    
    if xi.shape[2] == 1:
        xi = np.reshape(xi, (xi.shape[0], xi.shape[1], 1))

    # Save the prior initialization state for next time
    prior = gamma[:, 0]

    return prior, gamma, xi, alpha1, alpha2, scale, scale_a, score