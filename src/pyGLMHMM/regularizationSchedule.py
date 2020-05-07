import numpy as np

def _regularization_schedule(iter):
    tau = 1.5

    # Schedule 1:by iteration
    if iter < 5:
        this_lambda = 1
    else:
        this_lambda = np.exp((-1) * (iter - 4) / tau) + 0.005
    
    return this_lambda