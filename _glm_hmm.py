import copy
import warnings
import numpy as np

import scipy.special
import scipy.stats
import scipy.ndimage.filters
from scipy.sparse import spdiags
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.io import savemat

from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_random_state

class GLMHMMEstimator(BaseEstimator):
    """
    GLMHMMEstimator - Fits a combination of Multinomial Generalized Linear Model (GLM) and Hidden Markov Model (HMM) to behavioral data.

    Internal states shape stimulus responses and decision-making, but we lack methods to identify them. To address this gap, we
    developed an unsupervised method to identify internal states from behavioral data and applied it to a dynamic social interaction.
    During courtship, Drosophila melanogaster males pattern their songs using feedback cues from their partner. Our model
    uncovers three latent states underlying this behavior and is able to predict moment-to-moment variation in song-patterning
    decisions. These states correspond to different sensorimotor strategies, each of which is characterized by different mappings
    from feedback cues to song modes. We show that a pair of neurons previously thought to be command neurons for song production
    are sufficient to drive switching between states. Our results reveal how animals compose behavior from previously
    unidentified internal states, which is a necessary step for quantitative descriptions of animal behavior that link environmental
    cues, internal needs, neuronal activity and motor outputs.

    .. versionadded:: 1.0.0

    Inputs
    ----------
    stim (X) : The stimulus to be used for fitting. These should be in the form of (regressors, time) per trial in a list.
    symb (y) : The emitted discrete symbols to be fitted. These should be numbers from 0...N-1 (N: the number of possible outputs, i.e. song types) per trial in a list.
    analog_symb (y_analog) : The emitted continuous symbols to be fitted (for future extension).

    Parameters
    ----------
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    tol : float, defaults to 1e-4.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain on the likelihood (of the training data with
        respect to the model) is below this threshold.
    max_iter : int, defaults to 1000.
        The number of EM iterations to perform.
    num_samples : int, defaults to 1.
        The number of samples.
    num_states : int, defaults to 2.
        The number of hidden states.
    num_emissions : int, defaults to 2.
        The number of possible outputs, i.e. song types.
    num_feedbacks : int, defaults to 3.
        The number of feedback cues.
    num_filter_bins : int, defaults to 30.
        The sampling frequency of feedback cues.
    num_steps : int, defaults to 1.
        The number of steps in m-step of EM algorithm.
    init_loglik : float, defaults to -1e7.
        The initial log likelihood.
    smooth_lambda : float, defaults to 1.
        The regularization scheme.
    emit_lambda : float, defaults to 1.
        The regularization scheme.
    trans_lambda : float, defaults to 0.01.
        The regularization scheme.
    AR_lambda : float, defaults to -1.
        ...
    AR_vec : array-like, defaults to [value for value in range(511, 631)].
        ...
    stim_vec : array-like, defaults to [[value for value in range(1, 511)], 631].
        ...
    symb_exists : bool, defaults to True.
        True if symb exists, False otherwise.
    GLM_emissions : bool, defaults to True.
        True if GLM must be performed on emission outputs, False otherwise.
    GLM_transitions : bool, defaults to True.
        True if GLM must be performed on state transitions, False otherwise.
    evaluate : bool, defaults to False.
        True if the model must be evaluated, False otherwise.
    generate : bool, defaults to False.
        True if the model must be generated, False otherwise.
    L2_smooth : bool, defaults to True.
        True if regularization must be performed, False otherwise.
    analog_flag : bool, defaults to False.
        True if the analog version of the algorithm must be run, False otherwise.
    auto_anneal : bool, defaults to False.
        ...
    outfilename : String, defaults to 'GLMHMM_out.mat'.
        Output file name.

    Attributes
    ----------
    emit_w_ : array-like, shape (states, N - 1, regressors)
        The emission filter matrix.
    analog_emit_w_ : array-like, ...
        The continuous emission filter (for future extension).
    trans_w_ : array-like, shape (states, states, regressors)
        The transition filter matrix.
    symb_lik_ : array-like (list)
        The likelihood of output symbols.
    trans_lik_ : array-like (list)
        The likelihood of output states.
    analog_lik_ : array-like, ...
        The likelihood of continuous output states.
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    is_fitted_ : bool
        True if the fitting has been already performed, False otherwise.
    n_iter_ : int
        Number of step used by the best fit of inference to reach the
        convergence.

    References
    ----------
    .. [1] Calhoun, A.J., Pillow, J.W. & Murthy, M. Unsupervised identification of the internal states that shape natural behavior. Nat Neurosci 22, 2040–2049 (2019).
    .. [2] Escola, S., Fontanini, A., Katz, D. & Paninski, L. Hidden Markov models for the stimulus-response relationships of multistate neural systems. Neural Comput 23, 1071–1132 (2011).
    """

    def __init__(self,
                 random_state = None,
                 tol = 1e-4,
                 max_iter = 1000,
                 num_samples = 1,
                 num_states = 2,
                 num_emissions = 2,
                 num_feedbacks = 3,
                 num_filter_bins = 30,
                 num_steps = 1,
                 init_loglik = -1e7,
                 smooth_lambda = 1,
                 emit_lambda = 1,
                 trans_lambda = 0.01,
                 AR_lambda = -1,
                 AR_vec = [value for value in range(511, 631)],
                 stim_vec = [[value for value in range(1, 511)], 631],
                 symb_exists = True,
                 GLM_emissions = True,
                 GLM_transitions = True,
                 evaluate = False,
                 generate = False,
                 L2_smooth = True,
                 analog_flag = False,
                 auto_anneal = False,
                 outfilename = 'GLMHMM_out.mat'):

        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter
        self.num_samples = num_samples
        self.num_states = num_states
        self.num_emissions = num_emissions
        self.num_feedbacks = num_feedbacks
        self.num_filter_bins = num_filter_bins
        self.num_steps = num_steps
        self.init_loglik = init_loglik
        self.smooth_lambda = smooth_lambda
        self.emit_lambda = emit_lambda
        self.trans_lambda = trans_lambda
        self.AR_lambda = AR_lambda
        self.AR_vec = AR_vec
        self.stim_vec = stim_vec
        self.symb_exists = symb_exists
        self.GLM_emissions = GLM_emissions
        self.GLM_transitions = GLM_transitions
        self.evaluate = evaluate
        self.generate = generate
        self.L2_smooth = L2_smooth
        self.analog_flag = analog_flag
        self.auto_anneal = auto_anneal
        self.outfilename = outfilename

    def fit(self, X, y, y_analog):
        """Estimate model parameters with the EM algorithm.
        The method fits the model and sets the parameters with which the model
        has the largest likelihood. Within each trial, the method iterates
        between E-step and M-step for `max_iter` times until the change of
        likelihood is less than `tol`, otherwise, a `ConvergenceWarning` is
        raised.

        Parameters
        ----------
        X : array-like, shape (These should be in the form of (regressors, time) per trial in a list).
            The training input samples.
        y : array-like, shape (These should be numbers from 0...N-1 (N: the number of possible outputs, i.e. song types) per trial in a list).
            The target values (class labels in classification).
        y_analog : array-like, ...
            ...

        Returns
        -------
        self : object
            Returns self.
        """

        self.random_state = check_random_state(self.random_state)
        self._check_initial_parameters(X)

        do_init = not(hasattr(self, 'converged_'))
        self.converged_ = False

        if do_init:
            self._initialize_parameters(X, self.random_state)

        ###################################################
        # First set everything up
        ###################################################

        if len(y) > 0:
            self.symb_exists = True
        else:
            self.symb_exists = False

        total_trials = np.max((len(y), len(y_analog)))  # How many different trials are we fitting?

        loglik = np.zeros(self.max_iter + 1)
        loglik[0] = self.init_loglik

        self.num_states = np.max((self.emit_w_.shape[0], self.analog_emit_w_.shape[0]))
        self.num_emissions = self.emit_w_.shape[1]
        num_total_bins = np.max((self.emit_w_.shape[2], self.trans_w_.shape[2]))

        if self.analog_flag == True:
            num_analog_params = np.array(y_analog[0]).shape[0]
            num_analog_emit = np.zeros(num_analog_params)

            analog_residuals = np.zeros(num_analog_params, dtype = list)
            analog_prediction = np.zeros((len(y_analog), num_analog_params), dtype = list)
        else:
            num_analog_params = 0

        prior = []
        gamma = []
        xi = []

        if self.analog_flag == True:
            for trial in range(0, total_trials):
                prior.append(np.ones(self.num_states) / self.num_states)     # Is this good?!?!

                for analog_num in range(0, num_analog_params):
                    num_analog_emit[analog_num] = num_analog_emit[analog_num] + np.nansum(np.array(y_analog[trial][analog_num]))

                gamma.append(np.ones((self.num_states, np.array(y_analog[trial]).shape[0])))
                gamma[trial] = gamma[trial] / np.tile(np.sum(gamma[trial], axis = 0), (self.num_states, 1))

                xi.append(np.zeros(0))
        else:
            for trial in range(0, total_trials):
                prior.append(np.ones(self.num_states) / self.num_states)

                gamma.append(np.ones((self.num_states, np.array(y[trial]).shape[0])))
                gamma[trial] = gamma[trial] / np.tile(np.sum(gamma[trial], axis = 0), (self.num_states, 1))

                xi.append(np.zeros(0))

        ###################################################
        # Then the E-step
        ###################################################

        # First we need to know the likelihood of seeing each symbol given the filters as well as the likelihood of seeing a transition from state to state
        for trial in range(0, total_trials):
            if self.symb_exists == True:
                self.symb_lik_.append(_GLMHMM_symb_lik(self.emit_w_, X[trial], y[trial]))

            self.trans_lik_.append(_GLMHMM_trans_lik(self.trans_w_, X[trial]))

        if self.analog_flag == True:
            self.analog_lik_ = _GLMHMM_analog_lik(self.analog_emit_w_, X, y_analog, num_analog_emit)

        output = []

        for ind in range(0, self.max_iter):

            print('Fitting iteration:   ' + str(ind + 1))

            prior, gamma, xi = self._e_step(X, prior, gamma, xi, total_trials)

            ###################################################
            # Now the M-step
            ###################################################

            # First do gradient descent for the emission filter
            # NOTE: 'symb_exists' is holdover from the initial parameters to fit only analog features
            if self.symb_exists == True:
                print('Fitting categorical emission filters')

                # Format the data appropriately to pass into the M-step function
                # NOTE: I can probably do this more cleanly
                X_new = []

                for trial in range(0, total_trials):
                    # Please don't ask me why I decided it was a good idea to call the number of emissions 'num_states' here. Just roll with it!
                    X_new.append({'gamma' : gamma[trial], 'xi' : xi[trial], 'emit' : y[trial], 'num_states' : self.num_emissions})
                    if self.GLM_emissions == True:
                        X_new[trial]['data'] = X[trial]
                        X_new[trial]['num_total_bins'] = num_total_bins
                    else:
                        X_new[trial]['data'] = X[trial][-1, :]
                        X_new[trial]['num_total_bins'] = 1

                if self.evaluate == True:
                    if self.generate == True:
                        if self.analog_flag == True:
                            for trial in range(0, total_trials):
                                X_new[trial]['y_analog'] = y_analog[trial]
                                X_new[trial]['analog_emit_w'] = self.analog_emit_w_
                        else:
                            for trial in range(0, total_trials):
                                X_new[trial]['y_analog'] = np.nan
                                X_new[trial]['analog_emit_w'] = 0
                else:
                    self._m_step_emission(X_new)

            # Gradient descent for the transition filter
            print('Fitting state transition filters')

            # This is essentially the same as fitting the emission filter
            X_new = []

            for trial in range(0, total_trials):

                X_new.append({'gamma' : gamma[trial], 'xi' : xi[trial], 'num_states' : self.num_states})
                if self.GLM_emissions == True:
                    X_new[trial]['data'] = X[trial]
                    X_new[trial]['num_total_bins'] = num_total_bins
                else:
                    X_new[trial]['data'] = X[trial][-1, :]
                    X_new[trial]['num_total_bins'] = 1

            tgd_likelihood = self._m_step_transition(X_new)

            # Now we have done the E and the M steps! Just save the likelihoods!
            symb_likelihood = 0
            trans_likelihood = 0
            analog_likelihood = 0

            if self.analog_flag == True:
                self.analog_lik_ = _GLMHMM_analog_lik(self.analog_emit_w_, X, y_analog, num_analog_emit)

            for trial in range(0, total_trials):
                if self.symb_exists == True:
                    self.symb_lik_[trial] = _GLMHMM_symb_lik(self.emit_w_, X[trial], y[trial])

                self.trans_lik_[trial] = _GLMHMM_trans_lik(self.trans_w_, X[trial])

                symb_likelihood = symb_likelihood + -np.sum(gamma[trial] * np.log(self.symb_lik_[trial]))
                trans_likelihood = trans_likelihood + -np.sum(xi[trial] * np.log(self.trans_lik_[trial][:, :, 1:]))

                if self.analog_flag == True:
                    analog_prod = np.prod(np.array(self.analog_lik_[trial]), 0)
                    analog_prod[analog_prod < np.finfo(float).eps*1e3] = np.finfo(float).eps*1e3
                    analog_likelihood = analog_likelihood + -sum(gamma[trial] * np.log(analog_prod))

            # Basic log likelihood: sum(gamma(n) * log(gamma(n))) + tgd_lik + pgd_lik
            basic_likelihood = np.zeros(total_trials)

            for i in range(0, total_trials):
                gamma[i][gamma[i][:, 0] == 0, 0] = np.finfo(float).eps
                basic_likelihood[i] = -np.sum(gamma[i][:, 0] * np.log(gamma[i][:, 0]))

            loglik[ind + 1] = np.sum(basic_likelihood) + symb_likelihood + trans_likelihood + analog_likelihood

            # Saving variables
            output.append({'emit_w' : self.emit_w_, 'trans_w': self.trans_w_})
            if self.symb_exists == True:
                output[ind]['symb_likelihood'] = symb_likelihood
            if self.analog_flag == True:
                output[ind]['analog_emit_w'] = self.analog_emit_w_
                output[ind]['analog_likelihood'] = analog_likelihood
            output[ind]['trans_likelihood'] = trans_likelihood
            output[ind]['tgd_likelihood'] = tgd_likelihood
            output[ind]['loglik'] = loglik[1:ind + 1]
            output[ind]['gamma'] = gamma
            output[ind]['smooth_lambda'] = self.smooth_lambda
            output[ind]['emit_lambda'] = self.emit_lambda
            output[ind]['trans_lambda'] = self.trans_lambda

            parameters_dict = self.get_params()
            parameters_dict.pop('random_state')
            savemat(self.outfilename, {'output' : output, 'parameters' : parameters_dict})

            print('Log likelihood: ' + str(loglik[ind + 1]))

            # I have been stopping if the % change in log likelihood is below some threshold
            if (abs(loglik[ind + 1] - loglik[ind]) / abs(loglik[ind])) < self.tol:

                print('Change in log likelihood is below threshold!')
                self.converged_ = True
                self.n_iter_ = ind + 1

                break

        print('FINISHED!')

        if not self.converged_:
            warnings.warn('Initialization did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          , ConvergenceWarning)

        self.is_fitted_ = True

        # `fit` should always return `self`
        return self

    def predict(self, X):
        """Estimate model parameters using X and predict the labels for X.
        The method fits the model and sets the parameters with which the model
        has the largest likelihood. Within each trial, the method iterates
        between E-step and M-step for `max_iter` times until the change of
        likelihood is less than `tol`, otherwise, a `ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        Parameters
        ----------
        X : array-like, shape (These should be in the form of (regressors, time) per trial in a list).
            The training input samples.

        Returns
        -------
        y : array-like, shape (These should be numbers from 0...N-1 (N: the number of possible outputs, i.e. song types) per trial in a list).
            The target values (class labels in classification).
        """

        pass

    def _check_initial_parameters(self, X):
        """Check values of the basic parameters.

        Parameters
        ----------
        X : array-like, shape (These should be in the form of (regressors, time) per trial in a list).
            The training input samples.
        """

        if self.tol < 0.:
            raise ValueError("Invalid value for 'tol': %.5f "
                             "Tolerance used by the EM must be non-negative"
                             % self.tol)

        if type(self.max_iter) != int or self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d "
                             "It must be an integer value greater than one"
                             % self.max_iter)

        if type(self.num_samples) != int or self.num_samples < 1:
            raise ValueError("Invalid value for 'num_samples': %d "
                             "It must be an integer value greater than one"
                             % self.num_samples)

        if type(self.num_states) != int or self.num_states < 1:
            raise ValueError("Invalid value for 'num_states': %d "
                             "It must be an integer value greater than one"
                             % self.num_states)

        if type(self.num_emissions) != int or self.num_emissions < 1:
            raise ValueError("Invalid value for 'num_emissions': %d "
                             "It must be an integer value greater than one"
                             % self.num_emissions)

        if type(self.num_feedbacks) != int or self.num_feedbacks < 1:
            raise ValueError("Invalid value for 'num_feedbacks': %d "
                             "It must be an integer value greater than one"
                             % self.num_feedbacks)

        if type(self.num_filter_bins) != int or self.num_filter_bins < 1:
            raise ValueError("Invalid value for 'num_filter_bins': %d "
                             "It must be an integer value greater than one"
                             % self.num_filter_bins)

        if type(self.num_steps) != int or self.num_steps < 1:
            raise ValueError("Invalid value for 'num_steps': %d "
                             "It must be an integer value greater than one"
                             % self.num_steps)

    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like, shape (These should be in the form of (regressors, time) per trial in a list).
            The training input samples.
        random_state : RandomState
            A random number generator instance.
        """

        self.emit_w_ = np.zeros((self.num_states, self.num_emissions - 1, self.num_filter_bins * self.num_feedbacks + 1))     # states x emissions-1 x filter bins
        self.analog_emit_w_ = np.array([])
        self.trans_w_ = np.zeros((self.num_states, self.num_states, self.num_filter_bins * self.num_feedbacks + 1))     # states x states x filter bins (diagonals are ignored!)

        for ss1 in range(0, self.num_states):
            for ff in range(0, self.num_feedbacks):
                for ee in range(0, self.num_emissions - 1):
                    self.emit_w_[ss1, ee, (ff - 1) * self.num_filter_bins + np.arange(self.num_filter_bins)] = np.exp(-np.arange(self.num_filter_bins) / self.num_filter_bins) * np.round(self.random_state.rand(1) * 2 - 1)

                for ss2 in range(self.num_states):
                    self.trans_w_[ss1, ss2, (ff - 1) * self.num_filter_bins + np.arange(self.num_filter_bins)] = np.exp(-np.arange(self.num_filter_bins) / self.num_filter_bins) * np.round(self.random_state.rand(1) * 2 - 1)

        self.symb_lik_ = []
        self.trans_lik_ = []
        self.analog_lik_ = []

    def _e_step(self, X, prior, gamma, xi, total_trials):
        """E step.
        """

        for trial in range(0, total_trials):
             # Maybe first compute likelihoods for the symbols?
             if self.analog_flag == True and self.symb_exists == True:
                 emit_likelihood = np.array(self.symb_lik_[trial]) * np.prod(np.array(self.analog_lik_[trial]), 0)
             elif self.symb_exists == True:
                 emit_likelihood = np.array(self.symb_lik_[trial])
             elif self.analog_flag == True:
                 emit_likelihood = np.prod(np.array(self.analog_lik_[trial]), 0)

             # Things get funky if the likelihood is exactly 0
             emit_likelihood[emit_likelihood < np.finfo(float).eps*1e3] = np.finfo(float).eps*1e3    # Do we need this?

             # Use the forward-backward algorithm to estimate the probability of being in a given state (gamma),
             # probability of transitioning between states (xi), and hold the prior for initialization of next round of fitting
             prior[trial], gamma[trial], xi[trial] = _compute_trial_expectation(prior[trial], emit_likelihood, self.trans_lik_[trial])

        return prior, gamma, xi

    def _m_step_emission(self, X_new):
        """M step for emitted symbols.
        """

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        # I am using the scipy minimization function and passing the analytic value and the gradient to it
        # NOTE: I also could compute the Hessian but in my experience, it ends up not speeding the fitting up much because it is very slow and memory intensive to compute on each iteration
        # NOTE: This also returns the inverse Hessian so we can get the error bars from that if we want to
        for i in range(0, self.num_states):
            if self.num_steps == 1:
                outweights = minimize(lambda x: _emit_learning_fun(x, X_new, i, self.get_params()), np.reshape(self.emit_w_[i, :, :].T, (self.emit_w_.shape[1] * self.emit_w_.shape[2], 1), order = 'F'), jac = True, method = 'BFGS')
            else:
                outweights = minimize(lambda x: _emit_multistep_learning_fun(x, X_new, i, self.get_params()), np.reshape(self.emit_w_[i, :, :].T, (self.emit_w_.shape[1] * self.emit_w_.shape[2], 1), order = 'F'), jac = True, method = 'BFGS')

            self.emit_w_[i, :, :] = np.reshape(outweights['x'], (self.emit_w_.shape[2], self.emit_w_.shape[1]), order = 'F').T   # Make sure this is reformatted properly!!!

        return

    def _m_step_transition(self, X_new):
        """M step for state transition.
        """

        tmp_tgd = np.zeros(self.num_states)

        if self.evaluate == True:
            for i in range(0, self.num_states):
                tmp_tgd[i] = _trans_learning_fun(np.reshape(self.trans_w_[i, :, :].T, (self.trans_w_.shape[1] * self.trans_w_.shape[2], 1), order = 'F'), X_new, i, self.get_params())

            tgd_lik = np.sum(tmp_tgd)

        else:
            for i in range(0, self.num_states):
                print('In state ' + str(i))
                outweights = minimize(lambda x: _trans_learning_fun(x, X_new, i, self.get_params()), np.reshape(self.trans_w_[i,:,:].T, (self.trans_w_.shape[1] * self.trans_w_.shape[2], 1), order = 'F'), jac = True, method = 'BFGS')
                self.trans_w_[i,:,:] = np.reshape(outweights['x'], (self.trans_w_.shape[2], self.trans_w_.shape[1]), order = 'F').T
                tmp_tgd[i] = _trans_learning_fun(np.reshape(self.trans_w_[i, :, :].T, (self.trans_w_.shape[1] * self.trans_w_.shape[2], 1), order = 'F'), X_new, i, self.get_params())[0]

            tgd_lik = np.sum(tmp_tgd)

        return tgd_lik

    def get_params(self, deep = True):

        return {"random_state" : self.random_state,
                "tol" : self.tol,
                "max_iter" : self.max_iter,
                "num_samples" : self.num_samples,
                "num_states" : self.num_states,
                "num_emissions" : self.num_emissions,
                "num_feedbacks" : self.num_feedbacks,
                "num_filter_bins" : self.num_filter_bins,
                "num_steps" : self.num_steps,
                "init_loglik" : self.init_loglik,
                "smooth_lambda" : self.smooth_lambda,
                "emit_lambda" : self.emit_lambda,
                "trans_lambda" : self.trans_lambda,
                "AR_lambda" : self.AR_lambda,
                "AR_vec" : self.AR_vec,
                "stim_vec" : self.stim_vec,
                "symb_exists" : self.symb_exists,
                "GLM_emissions" : self.GLM_emissions,
                "GLM_transitions" : self.GLM_transitions,
                "evaluate" : self.evaluate,
                "generate" : self.generate,
                "L2_smooth" : self.L2_smooth,
                "analog_flag" : self.analog_flag,
                "auto_anneal" : self.auto_anneal,
                "outfilename" : self.outfilename}

    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

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
            symb_lik[:, t] = symb_lik[:, t] * np.exp(np.sum(emit_w[:, int(y_trial[t]) - 1, :] * X_trial_mod[:, int(y_trial[t]) - 1, :, t], axis = 1))

        if np.any(np.isnan(symb_lik[:, t])):
            print('WTF!')

    return symb_lik

def _GLMHMM_trans_lik(trans_w, X_trial):
    T = X_trial.shape[1]
    num_states = trans_w.shape[0]
    num_total_bins = trans_w.shape[2]
    trans_lik = np.zeros((num_states, num_states, T))

    for i in range(0, num_states):
        filtpower = np.sum(np.tile(np.reshape(trans_w[i, :, :], (num_states, num_total_bins, 1), order = 'F'), (1 , 1, T)) * np.tile(np.reshape(X_trial, (1, X_trial.shape[0], T), order = 'F'), (num_states, 1, 1)), axis = 1)

        # There is no filter for going from state i to state i
        filtpower[i, :] = 0
        for j in range(0, num_states):
            trans_lik[i, j, :] = np.exp(filtpower[j, :] - scipy.special.logsumexp(filtpower, axis = 0))

    return trans_lik

def _GLMHMM_analog_lik(analog_emit_w, X, y_analog, num_analog_emit):

    analog_lik = 0

    return analog_lik

def _compute_trial_expectation(prior, likelihood, transition):
    # Forward-backward algorithm, see Rabiner for implementation details
	# http://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf

    total_time = likelihood.shape[1]
    num_states = likelihood.shape[0]

    # E-step
    # alpha is the forward probability of seeing the sequence
    alpha = np.zeros((prior.shape[0], total_time))
    scale_a = np.ones(total_time)

    alpha[:, 0] = prior * likelihood[:, 0]
    alpha[:, 0] = alpha[:, 0] / np.sum(alpha[:, 0])

    for t in range(1, total_time):
        alpha[:, t] = np.matmul(transition[:, :, t], alpha[:, t - 1]) * likelihood[:, t]

        # Use this scaling component to try to prevent underflow errors
        scale_a[t] = np.sum(alpha[:,t])
        alpha[:, t] = alpha[:, t] / scale_a[t]

    # beta is the backward probability of seeing the sequence
    beta = np.zeros((len(prior), total_time))	# beta(i, t) = Pr(O(t + 1 : total_time) | X(t) = i)
    scale_b = np.ones(total_time)

    beta[:, -1] = np.ones(prior.shape[0]) / len(prior)

    for t in range(total_time - 2, 0, -1):
        beta[:, t] = np.matmul(transition[:, :, t + 1], (beta[:, t + 1] * likelihood[:, t + 1]))

        scale_b[t] = np.sum(beta[:,t])
        beta[:, t] = beta[:, t] / scale_b[t]

    # If any of the values are 0, it's defacto an underflow error so set it to eps
    alpha[alpha == 0] = np.finfo(float).eps
    beta[beta == 0] = np.finfo(float).eps

    # gamma is the probability of seeing the sequence, found by combining alpha and beta
    gamma = np.exp(np.log(alpha) + np.log(beta) - np.tile(np.log(np.cumsum(scale_a)).T, (num_states, 1)) - np.tile(np.log(np.flip(np.cumsum(np.flip(scale_b, 0)), 0)).T, (num_states, 1)))
    gamma[gamma == 0] = np.finfo(float).eps
    gamma = gamma / np.tile(np.sum(gamma, axis = 0), (num_states, 1))

    # xi is the probability of seeing each transition in the sequence
    xi = np.zeros((len(prior), len(prior), total_time - 1))
    transition2 = copy.copy(transition[:, :, 1:])

    for s1 in range(0, num_states):
        for s2 in range(0, num_states):
            xi[s1, s2, :] = np.log(likelihood[s2, 1:]) + np.log(alpha[s1, 0:-1]) + np.log(np.squeeze(transition2[s1, s2, :]).T) + np.log(beta[s2, 1:]) - np.log(np.cumsum(scale_a[:-1])).T - np.log(np.flip(np.cumsum(np.flip(scale_b[1:], 0)), 0)).T
            xi[s1, s2, :] = np.exp(xi[s1, s2, :])

    xi[xi == 0] = np.finfo(float).eps

    # Renormalize to make sure everything adds up properly
    xi = xi / np.tile(np.sum(np.sum(xi, axis = 0), axis = 0), (num_states, num_states, 1))

    # Save the prior initialization state for next time
    prior = gamma[:, 0]

    return prior, gamma, xi

def _emit_learning_fun(emit_w, stim, state_num, options):

    # emit_w are the weights that we are learning: in format states x weights
    # stim is a list/dictionary with each stimulus (stim[]['data']) and the probability emission functions (stim[]['gamma'] and stim[]['xi'])

    num_states = stim[0]['num_states']
    num_total_bins = stim[0]['num_total_bins']
    # states x bins
    emit_w = np.reshape(emit_w, (num_total_bins, num_states), order = 'F').T

    all_grad = np.zeros((num_states, num_total_bins))
    all_value = 0
    total_T = 0

    this_lambda = options['emit_lambda']

    # Find out how many data points we are dealing with so that we can normalize
    # (I don't think we actually need to do this but it helps keep regularization values consistent from fit to fit)
    for trial in range(0, len(stim)):
        total_T = total_T + stim[trial]['data'].shape[1]

    for trial in range(0, len(stim)):
        T = stim[trial]['data'].shape[1]

        # Convert into states x bins x time and sum across bins
        filtpower = np.reshape(np.sum(np.reshape(np.tile(emit_w, (1, 1, T)), (num_states, num_total_bins, T), order = 'F') * np.tile(np.reshape(stim[trial]['data'], (1, num_total_bins, T), order = 'F'), (num_states, 1, 1)), axis = 1), (num_states, T), order = 'F')
        # Now filtpower is states x time
        # filtpower is the filter times the stimulus

        # Build up the value function:
        # gamma * log(exp(filtpower) / (1 + sum(exp(filtpower)))) = gamma * filtpower - gamma * log(1 + sum(exp(filtpower)))
        # Gradient is then:
        # gamma * (1|emission - exp(filtpower) / (1+sum(exp(filtpower)))) * stim
        value = stim[trial]['gamma'][state_num, :] * -np.log(1 + np.sum(np.exp(filtpower), axis = 0))
        tgrad = -np.exp(filtpower) / np.tile(1 + np.sum(np.exp(filtpower), axis = 0), (num_states, 1))

        for i in range(0, filtpower.shape[0]):
            tgrad[i, stim[trial]['emit'] == i] = 1 + tgrad[i, stim[trial]['emit'] == i]
            value[stim[trial]['emit'] == i] = value[stim[trial]['emit'] == i] + stim[trial]['gamma'][state_num, stim[trial]['emit'] == i] * filtpower[i, stim[trial]['emit'] == i]

        value = np.sum(value)
        if np.any(np.isnan(value)):
            print('ugh')

        tgrad = tgrad * np.tile(stim[trial]['gamma'][state_num, :], (num_states, 1))
        tgrad = np.sum(np.tile(np.reshape(tgrad, (num_states, 1, T), order = 'F'), (1, num_total_bins, 1)) * np.tile(np.reshape(stim[trial]['data'], (1, num_total_bins, T), order = 'F'), (num_states, 1, 1)), axis = 2)

        all_grad = all_grad + tgrad
        all_value = all_value + value

    grad_regularization = 0
    value_regularization = 0

    if options['L2_smooth'] == True:

        Dx1 = spdiags((np.ones((emit_w.shape[1] - 1, 1)) * np.array([-1, 1])).T, np.array([0, 1]), emit_w.shape[1] - 1 - 1, emit_w.shape[1] - 1).toarray()
        Dx = np.matmul(Dx1.T, Dx1)

        for fstart in range(options['num_filter_bins'] + 1, emit_w.shape[1] - 1, options['num_filter_bins']):
            Dx[fstart, fstart] = 1
            Dx[fstart - 1, fstart - 1] = 1
            Dx[fstart - 1, fstart] = 0
            Dx[fstart, fstart - 1] = 0

        D = block_diag(Dx, 0)

        if options['AR_lambda'] != -1:
            if len(options['smooth_lambda']) == 1:
                options['smooth_lambda'] = np.tile(options['smooth_lambda'][0], [emit_w.shape[0], emit_w.shape[1]])
                options['smooth_lambda'][:, options['AR_vec']] = options['AR_lambda']

            grad_regularization = grad_regularization + options['smooth_lambda'] * (np.matmul(D, emit_w.T)).T
            value_regularization = value_regularization + np.sum(np.sum(np.power((options['smooth_lambda'] / 2) * (np.matmul(D, emit_w.T)).T, 2)))
        else:
            grad_regularization = grad_regularization + options['smooth_lambda'] * (np.matmul(D, emit_w.T)).T
            value_regularization = value_regularization + np.sum(np.sum(np.power((options['smooth_lambda'] / 2) * (np.matmul(D, emit_w.T)).T, 2)))

    if this_lambda != 0:
        if options['AR_lambda'] != -1:
            grad_regularization = grad_regularization + [this_lambda * emit_w[:, options['stim_vec']], options['AR_lambda'] * emit_w[:, options['AR_vec']]]
            value_regularization = value_regularization + (this_lambda / 2) * np.sum(np.sum(np.power(emit_w[:, options['stim_vec']], 2))) + (options['AR_lambda'] / 2) * np.sum(np.sum(np.power(emit_w[:, options['AR_vec']], 2)))
        else:
            grad_regularization = grad_regularization + this_lambda * emit_w
            value_regularization = value_regularization + (this_lambda/2) * np.sum(np.sum(np.power(emit_w, 2)))

    all_grad = -all_grad / total_T + grad_regularization
    all_value = -all_value / total_T + value_regularization

    if np.any(np.isnan(all_grad)) or np.any(np.isnan(all_value)):
        print('WTF! SOMETHING BAD HAPPENED! ISNAN! OH!')

    all_grad = np.reshape(all_grad.T, (all_grad.shape[0] * all_grad.shape[1], 1), order = 'F')

    return all_value, np.squeeze(all_grad)

def _emit_multistep_learning_fun(emit_w, stim, state_num, options):

    # emit_w are the weights that we are learning: in format states x weights
    # stim is a list/dictionary with each stimulus (stim[]['data']) and the probability emission functions (stim[]['gamma'] and stim[]['xi'])

    # I will have to do something to make this more generic to work with other formats
    num_steps = options['num_steps']
    num_samples = options['num_samples']

    num_states = stim[0]['num_states']
    num_total_bins = stim[0]['num_total_bins']
    # states x bins
    emit_w = np.reshape(emit_w, (num_total_bins, num_states), order = 'F').T

    all_grad = np.zeros((num_states, num_total_bins))
    all_value = 0
    total_T = 0

    this_lambda = options['emit_lambda']

    for trial in range(0, len(stim)):
        total_T = total_T + stim[trial]['data'].shape[1]

    for trial in range(0, len(stim)):
        # Basically, for each step we are looking ahead, we are going to generate a sample and then use that to calculate the lookahead likelihood
        # Since we are using large amounts of data here, we can get away with only using one sample (I think!)

        # I might have to use ADAM for SGD?
        # https://www.mathworks.com/matlabcentral/fileexchange/61616-adam-stochastic-gradient-descent-optimization
        for sample in range(0, num_samples):

            new_stim = stim[trial]['data']
            new_emit = np.array([])

            for step in range(0, num_steps):

                # Two steps:
                # First, find the likelihood of the actual data at STEPs away
                # Second, find the likelihood of all generated data...

                # FIRST:
                if step == 0:
                    T = new_stim.shape[1]
                    # Convert into states x bins x time and sum across bins
                    filtpower = np.reshape(np.sum(np.reshape(np.tile(emit_w, (1, 1, T)), (num_states, num_total_bins, T), order = 'F') * np.tile(np.reshape(new_stim, (1, num_total_bins, T), order = 'F'), (num_states, 1, 1)), axis = 1), (num_states, T), order = 'F')
                    # Now filtpower is states x time

                    value = stim[trial]['gamma'][state_num, :] * -np.log(1 + np.sum(np.exp(filtpower), axis = 0))
                    tgrad = -np.exp(filtpower) / np.tile(1 + np.sum(np.exp(filtpower), axis = 0), (num_states, 1))

                    for i in range(0, filtpower.shape[0]):
                        tgrad[i, stim[trial]['emit'] == i] = 1 + tgrad[i, stim[trial]['emit'] == i]
                        value[stim[trial]['emit'] == i] = value[stim[trial]['emit'] == i] + stim[trial]['gamma'][state_num, stim[trial]['emit'] == i] * filtpower[i, stim[trial]['emit'] == i]

                    value = np.sum(value)
                    tgrad = tgrad * np.tile(stim[trial]['gamma'][state_num, :], [num_states, 1])

                    tgrad = np.sum(np.tile(np.reshape(tgrad, (num_states, 1, T), order = 'F'), (1, num_total_bins, 1)) * np.tile(np.reshape(new_stim, (1, num_total_bins , T), order = 'F'), (num_states, 1, 1)), axis = 2)

                    all_grad = all_grad + tgrad
                    all_value = all_value + value

                    old_stim = new_stim;
                    old_gamma = stim[trial]['gamma']

                # SECOND
                if step > 0:
                    T = old_stim.shape[1]
                    # Convert into states x bins x time and sum across bins
                    filtpower = np.reshape(np.sum(np.reshape(np.tile(emit_w, (1, 1, T)), (num_states, num_total_bins, T), order = 'F') * np.tile(np.reshape(old_stim, (1, num_total_bins, T), order = 'F'), (num_states, 1, 1)), axis = 1), (num_states, T), order = 'F')
                    # Now filtpower is states x time

                    value = old_gamma[state_num, :] * -np.log(1 + np.sum(np.exp(filtpower), axis = 0))
                    tgrad = -np.exp(filtpower) / np.tile(1 + np.sum(np.exp(filtpower), axis = 0), (num_states, 1))

                    for i in range(0, filtpower.shape[0]):
                        tgrad[i, new_emit == i] = 1 + tgrad[i, new_emit == i]
                        value[new_emit == i] = value[new_emit == i] + old_gamma[state_num, new_emit == i] * filtpower[i, new_emit == i]

                    value = np.sum(value)
                    tgrad = tgrad * np.tile(old_gamma[state_num, :], (num_states, 1))

                    tgrad = np.sum(np.tile(np.reshape(tgrad, (num_states, 1, T), order = 'F'), (1, num_total_bins, 1)) * np.tile(np.reshape(old_stim, (1, num_total_bins, T), order = 'F'), (num_states, 1, 1)), axis = 2)

                    all_grad = all_grad + tgrad * (num_steps - step + 1)
                    all_value = all_value + value * (num_steps - step + 1)

    # Implement smoothing: block matrix that is lambda_2 * [[1,-1,...],[-1,2,-1,...],[0,-1,2,-1,0,...]]
    # I need to make sure that this matrix takes into account the boundary size...
    if options['auto_anneal'] == True:
        all_grad = -all_grad / total_T + this_lambda * emit_w
        all_value = -all_value / total_T + (this_lambda/2) * np.sum(np.sum(np.power(emit_w, 2)))

    elif options['L2_smooth'] == False:
        all_grad = -all_grad / total_T + this_lambda * emit_w
        all_value = -all_value / total_T + (this_lambda / 2) * np.sum(np.sum(np.power(emit_w, 2)))

        Dx1 = spdiags((np.ones(emit_w.shape[1] - 1, 1) * np.array([-1, 1])).T, np.array([0, 1]), emit_w.shape[1] - 1 - 1, emit_w.shape[1] - 1).toarray()
        Dx = Dx1.T * Dx1

        for fstart in range(options['num_filter_bins'] + 1, emit_w.shape[1] - 1, options['num_filter_bins']):
            Dx[fstart, fstart] = 1
            Dx[fstart - 1, fstart - 1] = 1
            Dx[fstart - 1, fstart] = 0
            Dx[fstart, fstart - 1] = 0

        D = block_diag(Dx, 0)

        all_grad = all_grad + options['smooth_lambda'] * (np.matmul(D, emit_w.T)).T
        all_value = all_value + (options['smooth_lambda'] / 2) * np.sum(np.sum(np.power(np.matmul(D, emit_w.T), 2)))

    else:
        all_grad = -all_grad / total_T + this_lambda * emit_w
        all_value = -all_value / total_T + (this_lambda / 2) * np.sum(np.sum(np.power(emit_w, 2)))

    if np.any(np.isnan(all_grad)) or np.any(np.isnan(all_value)):
        print('WTF! SOMETHING BAD HAPPENED! ISNAN! OH!')

    all_grad = np.reshape(all_grad.T, (all_grad.shape[0] * all_grad.shape[1], 1), order = 'F')

    return all_value, np.squeeze(all_grad)

def _trans_learning_fun(trans_w, stim, state_num, options):

    # trans_w are the weights that we are learning: in format states x weights
    # stim is a list/dictionary with each stimulus (stim[]['data']) and the probability transition functions (stim[]['gamma'] and stim[]['xi'])
    # NOTE: This transition function is dependent on where we are transitioning from, and relies on each of the other possible states we could be transitioning to,
    # so we cannot minimize these independently. Thus we are really going to find the gradient of all transition filters originating from some state.

    num_states = stim[0]['num_states']
    num_total_bins = stim[0]['num_total_bins']

    trans_w = np.reshape(trans_w, (num_total_bins, num_states), order = 'F').T

    all_grad = np.zeros((num_states, num_total_bins))
    all_value = 0
    total_T = 0

    this_lambda = options['trans_lambda']

    for trial in range(0, len(stim)):
        total_T = total_T + stim[trial]['data'].shape[1]

    for trial in range(0, len(stim)):
        tgrad = np.zeros(trans_w.shape[0])
        T = stim[trial]['data'].shape[1] - 1
        # Use data from 1:end-1 or 2:end?
        filtpower = np.sum(np.tile(np.expand_dims(trans_w, axis = 2), (1, 1, T)) * np.tile(np.reshape(stim[trial]['data'][:, 1:], (1, num_total_bins, T), order = 'F'), (num_states, 1, 1)), axis = 1)
        # Now filtpower is states x time

        value = -stim[trial]['gamma'][state_num, 0:-1] * np.log(1 + np.sum(np.exp(filtpower[np.setdiff1d(np.arange(num_states), state_num), :]), axis = 0))

        if stim[trial]['xi'].shape[2] == 1:
            tgrad = stim[trial]['xi'][state_num, :, :].T;
        else:
            tgrad = stim[trial]['xi'][state_num, :, :]

        offset = stim[trial]['gamma'][state_num, 0:-1] / (1 + np.sum(np.exp(filtpower[np.setdiff1d(np.arange(num_states), state_num), :]), axis = 0))
        for i in range(0, num_states):
            if state_num != i:
                value = value + stim[trial]['xi'][state_num, i, :].T * filtpower[i, :]
                tgrad[i, :] = tgrad[i, :] - np.exp(filtpower[i, :]) * offset
            else:
                tgrad[i, :] = 0

        tgrad = np.sum(np.tile(np.reshape(tgrad, (num_states, 1, T), order = 'F'), (1, num_total_bins, 1)) * np.tile(np.reshape(stim[trial]['data'][:, 1:], (1, num_total_bins, T), order = 'F'), (num_states, 1, 1)), axis = 2)

        # I probably don't need to rescale here because that happens naturally but... oh well!
        all_grad = all_grad + tgrad
        all_value = all_value + np.sum(value)

    grad_regularization = np.zeros(all_grad.shape)
    value_regularization = 0

    if options['L2_smooth'] == True:

        Dx1 = spdiags((np.ones((trans_w.shape[1] - 1, 1)) * np.array([-1, 1])).T, np.array([0, 1]), trans_w.shape[1] - 1 - 1, trans_w.shape[1] - 1).toarray()
        Dx = np.matmul(Dx1.T, Dx1)

        for fstart in range(options['num_filter_bins'] + 1, trans_w.shape[1] - 1, options['num_filter_bins']):
            Dx[fstart, fstart] = 1
            Dx[fstart - 1, fstart - 1] = 1
            Dx[fstart - 1, fstart] = 0
            Dx[fstart, fstart - 1] = 0

        D = block_diag(Dx, 0)

        if options['AR_lambda'] != -1:
            if len(options['smooth_lambda']) == 1:
                options['smooth_lambda'] = np.tile(options['smooth_lambda'][0], [trans_w.shape[0] - 1, trans_w.shape[1]])
                options['smooth_lambda'][:, options['AR_vec']] = options['AR_lambda']

            grad_regularization[np.setdiff1d(np.arange(num_states), state_num), :] = grad_regularization[np.setdiff1d(np.arange(num_states), state_num), :] + options['smooth_lambda'] * (np.matmul(D, trans_w[np.setdiff1d(np.arange(num_states), state_num), :].T)).T
            value_regularization = value_regularization + np.sum(np.sum(np.power((options['smooth_lambda'] / 2) * (np.matmul(D, trans_w[np.setdiff1d(np.arange(num_states), state_num), :].T)).T, 2)))
        else:
            grad_regularization[np.setdiff1d(np.arange(num_states), state_num), :] = grad_regularization[np.setdiff1d(np.arange(num_states), state_num), :] + options['smooth_lambda'] * (np.matmul(D, trans_w[np.setdiff1d(np.arange(num_states), state_num), :].T)).T
            value_regularization = value_regularization + np.sum(np.sum(np.power((options['smooth_lambda'] / 2) * (np.matmul(D, trans_w.T)).T, 2)))

    if this_lambda != 0:
        if options['AR_lambda'] != -1:
            grad_regularization = grad_regularization + [this_lambda * trans_w[:, options['stim_vec']], options['AR_lambda'] * trans_w[:, options['AR_vec']]]
            value_regularization = value_regularization + (this_lambda / 2) * np.sum(np.sum(np.power(trans_w[:, options['stim_vec']], 2))) + (options['AR_lambda'] / 2) * np.sum(np.sum(np.power(trans_w[:, options['AR_vec']], 2)))
        else:
            grad_regularization = grad_regularization + this_lambda * trans_w
            value_regularization = value_regularization + (this_lambda/2) * np.sum(np.sum(np.power(trans_w, 2)))

    all_grad = -all_grad / total_T + grad_regularization
    all_value = -all_value / total_T + value_regularization

    if all_value < 0:
        print('Why oh why oh why!')

    all_grad = np.reshape(all_grad.T, (all_grad.shape[0] * all_grad.shape[1], 1), order = 'F')

    return all_value, np.squeeze(all_grad)