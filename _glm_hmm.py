import copy
import warnings
import numpy as np

from scipy.optimize import minimize
from scipy.io import savemat

from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_random_state

from _GLMHMM_symb_lik import _GLMHMM_symb_lik
from _GLMHMM_analog_lik import _GLMHMM_analog_lik
from _GLMHMM_trans_lik import _GLMHMM_trans_lik
from _compute_trial_expectation import _compute_trial_expectation
from _emit_learning_fun import _emit_learning_fun
from _emit_multistep_learning_fun import _emit_multistep_learning_fun
from _trans_learning_fun import _trans_learning_fun
from _emit_likelihood import _emit_likelihood
from _emit_learning_stats import _emit_learning_stats
from _trans_learning_stats import _trans_learning_stats
from _weighted_LS_by_state import _weighted_LS_by_state
from _collect_WLS_info import _collect_WLS_info
from _regularization_schedule import _regularization_schedule

#import _emit_generate, _generate_next_step, _generate_posterior_nstep
#import _fit_emission_filters, _fit_transition_filters, _fit_analog_filters, _HMMGLM_likelihoods, _w_corr, _fast_ASD_weighted_group

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
        The number of steps in M-step of EM algorithm.
    filter_offset : int, defaults to 1.
        ...
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
    AR_vec : array-like, defaults to np.arange(510, 630).
        ...
    stim_vec : array-like, defaults to np.setdiff1d(np.arange(0, 631), np.arange(510, 630)).
        ...
    auto_anneal_vec : array-like, defaults to np.array([0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]). 
        ...
    auto_anneal_schedule : array-like, defaults to np.array([1, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).
        ...
    train_bins : array-like, defaults to np.array([]).
        ...  
    symb_exists : bool, defaults to True.
        True if symb exists, False otherwise.
    use_ASD : bool, defaults to True.
        ...    
    add_filters : bool, defaults to False.
        True if filters must be added, False otherwise.
    fit_emissions : bool, defaults to True.
        True if emissions must be fitted, False otherwise.
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
    anneal_lambda : bool, defaults to False.
        ...
    get_error_bars : bool, defaults to False.
        True if error-bars must be calculated, False otherwise.
    CV_regularize : bool, defaults to False.
        True if cross-validation for regularization must be performed, False otherwise.
    cross_validate : bool, defaults to False.
        True if cross-validation must be performed, False otherwise.
    output_filename : string, defaults to 'GLMHMM_output.mat'.
        Output file name.
         
    Attributes
    ----------
    emit_w_ : array-like, shape (states, N - 1, regressors)
        The emission filter matrix.
    analog_emit_w_ : array-like, ...
        The continuous emission filter (for future extension).
    analog_emit_std_ : array-like, ...
        The continuous emission filter standard deviation (for future extension).
    trans_w_ : array-like, shape (states, states, regressors)
        The transition filter matrix.
    emit_w_init_ : array-like, shape (states, N - 1, regressors)
        The initial emission filter matrix.
    analog_emit_w_init_ : array-like, ...
        The initial continuous emission filter (for future extension).
    analog_emit_std_init : array-like, ...
        The initial continuous emission filter standard deviation (for future extension).
    trans_w_init_ : array-like, shape (states, states, regressors)
        The initial transition filter matrix.
    symb_lik_ : array-like (list)
        The likelihood of output symbols.
    analog_lik_ : array-like (list)
        The likelihood of continuous output states (for future extension).
    trans_lik_ : array-like (list)
        The likelihood of output states.
    regular_schedule_ : array-like
        The regularization schedule.
    regular_schedule_ind_ : int
        The regularization index.
    train_data_ : array-like
        The subset of stim (X) used for training.        
    test_data_ : array-like
        The subset of stim (X) used for validation.
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
                 filter_offset = 1,
                 init_loglik = -1e7,
                 smooth_lambda = 0,
                 emit_lambda = 0,
                 trans_lambda = 0,
                 AR_lambda = -1,
                 AR_vec = np.arange(510, 630),
                 stim_vec = np.setdiff1d(np.arange(0, 631), np.arange(510, 630)),
                 auto_anneal_vec = np.array([0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]),
                 auto_anneal_schedule = np.array([1, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
                 train_bins = np.array([]),
                 symb_exists = True,
                 use_ASD = True,
                 add_filters = False,
                 fit_emissions = True,
                 GLM_emissions = True,
                 GLM_transitions = True,
                 evaluate = False,
                 generate = False,
                 L2_smooth = False,
                 analog_flag = False,
                 auto_anneal = False,
                 anneal_lambda = False,
                 get_error_bars = False,
                 CV_regularize = False,
                 cross_validate = False,
                 output_filename = 'GLMHMM_output.mat'):
        
        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter
        self.num_samples = num_samples
        self.num_states = num_states
        self.num_emissions = num_emissions
        self.num_feedbacks = num_feedbacks
        self.num_filter_bins = num_filter_bins
        self.num_steps = num_steps
        self.filter_offset = filter_offset
        self.init_loglik = init_loglik      
        self.smooth_lambda = smooth_lambda
        self.emit_lambda = emit_lambda
        self.trans_lambda = trans_lambda
        self.AR_lambda = AR_lambda
        self.AR_vec = AR_vec
        self.stim_vec = stim_vec
        self.auto_anneal_vec = auto_anneal_vec
        self.auto_anneal_schedule = auto_anneal_schedule
        self.train_bins = train_bins
        self.symb_exists = symb_exists
        self.use_ASD = use_ASD
        self.add_filters = add_filters
        self.fit_emissions = fit_emissions
        self.GLM_emissions = GLM_emissions
        self.GLM_transitions = GLM_transitions
        self.evaluate = evaluate
        self.generate = generate
        self.L2_smooth = L2_smooth
        self.analog_flag = analog_flag
        self.auto_anneal = auto_anneal
        self.anneal_lambda = anneal_lambda
        self.get_error_bars = get_error_bars
        self.CV_regularize = CV_regularize
        self.cross_validate = cross_validate
        self.output_filename = output_filename
        
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
    
        total_trials = max(len(y), len(y_analog))  # How many different trials are we fitting?
    
        loglik = np.zeros(self.max_iter + 1)
        loglik[0] = self.init_loglik
    
        self.num_states = max(self.emit_w_.shape[0], self.analog_emit_w_.shape[0], self.trans_w_.shape[0])
        self.num_emissions = self.emit_w_.shape[1]
    
        if self.analog_flag == True:
            num_total_bins = max(self.emit_w_.shape[2], self.analog_emit_w_.shape[2], self.trans_w_.shape[2])           
            num_analog_params = y_analog[0].shape[0]
            num_analog_emit = np.zeros(num_analog_params)
        else:
            num_total_bins = max(self.emit_w_.shape[2], self.trans_w_.shape[2])
            num_analog_params = 0
        
        prior = []
        gamma = []
        xi = []
        
        for trial in range(0, total_trials):
            prior.append(np.ones(self.num_states) / self.num_states)     # Is this good?!?!
            
            if self.analog_flag == True:                
                for analog_num in range(0, num_analog_params):
                    num_analog_emit[analog_num] = num_analog_emit[analog_num] + np.nansum(y_analog[trial][analog_num, :], axis = 0)

                gamma.append(np.ones((self.num_states, y_analog[trial].shape[1])))
            else:
                if len(y[trial].shape) > 1:
                    gamma.append(np.ones((self.num_states, y[trial].shape[1])))
                else:
                    gamma.append(np.ones((self.num_states, 1)))
                    
            gamma[trial] = gamma[trial] / np.tile(np.sum(gamma[trial], axis = 0), (self.num_states, 1))
            
            xi.append([])
               
        ###################################################
        # Then the E-step
        ###################################################
       
        # First we need to know the likelihood of seeing each symbol given the filters as well as the likelihood of seeing a transition from state to state       
        effective_ind = 0
        last_try = False
        
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
            
            # Gradient descent for the emission filter
            print('Fitting categorical emission filters')

            if self.symb_exists == True and self.fit_emissions == True:
                            
                new_stim = []
                
                if self.cross_validate == True or self.CV_regularize == True:
                    if self.CV_regularize == True:
                        CV_schedule = self.regular_schedule_[max(self.regular_schedule_ind_ - 1, 1):min(self.regular_schedule_ind_ + 1, len(self.regular_schedule_))]
                    else:
                        CV_schedule = self.smooth_lambda
                        
                    CV_ind = -1
                    
                    new_emit_w = []
                    new_trans_w = []
                    
                    new_analog_emit_w = []
                    new_analog_emit_std = []
                    
                    test_emit_lik = []
                    test_trans_lik = []
                    test_analog_lik = []
                    test_gamma = []
                    test_xi = []
                    
                    test_log_lik = []
            
                    for this_lambda in CV_schedule:
                        CV_ind = CV_ind + 1
                        
                        # Check if we are doing smoothing regularization or Tikhonov
                        # Segment data into random subsets for cross validation
                        if self.smooth_lambda == -1:
                            self.trans_lambda = this_lambda
                            self.emit_lambda = this_lambda
                        else:
                            self.smooth_lambda = this_lambda
            
                        if self.symb_exists == True:
                            # _fit_emission_filters should take into account the test_data_
                            # field...
                            [new_emit_w_temp, pgd_lik, pgd_prob, pgd_prob2, pgd_prob3, pgd_prob4] = _fit_emission_filters(X, y, gamma, xi, self.emit_w_, self.get_params())
                            new_emit_w.append(new_emit_w_temp)
                        else:
                            pgd_lik = 0
            
                        [new_trans_w_temp, tgd_lik] = _fit_transition_filters(X, y, gamma, xi, self.trans_w_, self.get_params())
                        new_trans_w.append(new_trans_w_temp)
            
                        if self.analog_flag == True:
                            [new_analog_emit_w_temp, new_analog_emit_std_temp, arcorr, arcorr2] = _fit_analog_filters(X, y_analog, self.analog_emit_w_, self.get_params())
                            new_analog_emit_w.append(new_analog_emit_w_temp)
                            new_analog_emit_std.append(new_analog_emit_std_temp)
                        
                        test_symb = []
                        test_analog_symb = []
                        test_stim = []
                        
                        if self.analog_flag == True:
                            for i in range(0, len(self.test_data_)):
                                test_symb.append(y[self.test_data_[i]])
                                test_analog_symb.append(y_analog[self.test_data_[i]])
                                test_stim.append(X[self.test_data_[i]])
                                
                            [test_emit_lik_temp, test_trans_lik_temp, test_analog_lik_temp, test_gamma_temp, test_xi_temp] = _HMMGLM_likelihoods(test_symb, new_emit_w[CV_ind], new_trans_w[CV_ind], test_stim, new_analog_emit_w[CV_ind], test_analog_symb, self.get_params())
                        
                        else:
                            for i in range(0, len(self.test_data_)):
                                test_symb.append(y[self.test_data_[i]])
                                test_stim.append(X[self.test_data_[i]])
                                
                            [test_emit_lik_temp, test_trans_lik_temp, test_analog_lik_temp, test_gamma_temp, test_xi_temp] = _HMMGLM_likelihoods(test_symb, new_emit_w[CV_ind], new_trans_w[CV_ind], test_stim, [], [], self.get_params())
                
                        test_emit_lik.append(test_emit_lik_temp)
                        test_trans_lik.append(test_trans_lik_temp)
                        test_analog_lik.append(test_analog_lik_temp)
                        test_gamma.append(test_gamma_temp)
                        test_xi.append(test_xi_temp)
                                                
                        test_full_emit_lik  = 0
                        test_full_trans_lik  = 0
                        test_full_analog_emit_lik = 0
                        test_full_basic_lik = np.zeros((len(self.test_data_), 1))
                        
                        for i in range(0, len(self.test_data_)):
                            gamma[i][gamma[i][:, 0] == 0, 0] = np.finfo(float).eps
            
                            if self.symb_exists == True:
                                test_full_emit_lik = test_full_emit_lik - np.mean(np.sum(test_gamma[CV_ind][i] * np.log(test_emit_lik[CV_ind][i]), axis = 0), axis = 0)
                                test_full_trans_lik = test_full_trans_lik - np.mean(np.sum(np.sum(test_xi[CV_ind][i] * np.log(test_trans_lik[CV_ind][i][:, :, 1:]), axis = 0), axis = 0), axis = 0)
            
                            if self.analog_flag == True:
                                analog_prod = np.prod(test_analog_lik[CV_ind][i], axis = 0)
                                analog_prod[analog_prod == 0] = np.finfo(float).eps
                                
                                if self.num_states == 1:
                                    test_full_analog_emit_lik = test_full_analog_emit_lik - np.mean(np.sum(test_gamma[CV_ind][i] * np.log(analog_prod).T, axis = 0), axis = 0)
                                else:
                                    test_full_analog_emit_lik = test_full_analog_emit_lik - np.mean(np.sum(test_gamma[CV_ind][i] * np.log(analog_prod), axis = 0), axis = 0)
            
                            test_full_basic_lik[i] = -np.sum(test_gamma[CV_ind][i][:, 0] * np.log(test_gamma[CV_ind][i][:, 0]), axis = 0)
                        
                        test_log_lik.append(np.sum(test_full_basic_lik, axis = 0) + test_full_emit_lik + test_full_analog_emit_lik + test_full_trans_lik)
            
                    if self.CV_regularize == True:
                        CV_inds = np.arange(max(self.regular_schedule_ind_ - 1, 1), min(self.regular_schedule_ind_ + 1, len(self.regular_schedule_)))
                        good_ind = np.argwhere(test_log_lik == min(test_log_lik))
            
                        if self.symb_exists == True:
                            self.emit_w_ = new_emit_w[good_ind]
                            self.trans_w_ = new_trans_w[good_ind]
                            
                        if self.analog_flag == True:
                            self.analog_emit_w_ = new_analog_emit_w[good_ind]
                            self.analog_emit_std_ = new_analog_emit_std[good_ind]
            
                        this_lambda = self.regular_schedule_[CV_inds[good_ind]]
                        self.regular_schedule_ind_ = CV_inds[good_ind]
                        
                        if self.smooth_lambda == -1:
                            self.trans_lambda = this_lambda
                            self.emit_lambda = this_lambda
                        else:
                            self.smooth_lambda = this_lambda
                    else:
                        good_ind = 1
            
                    if self.cross_validate == True:
                        output[ind]['lambda'] = this_lambda
                        output[ind]['loglik_CV'] = test_log_lik[good_ind]
                        output[ind]['loglik_CV_lambda'] = CV_schedule
                        output[ind]['loglik_CV_all'] = test_log_lik
                    
                else:
                    for trial in range(0, len(X)):
                        # Please don't ask me why I decided it was a good idea to call the number of emissions 'num_states' here. Just roll with it!
                        new_stim.append({'emit' : y[trial], 'gamma' : gamma[trial], 'xi' : xi[trial], 'num_states' : self.num_emissions})
                        
                        if self.GLM_emissions == True:
                            new_stim[trial]['data'] = X[trial]
                            new_stim[trial]['num_total_bins'] = num_total_bins                            
                        else:
                            new_stim[trial]['data']  = X[trial][-1, :]
                            new_stim[trial]['num_total_bins'] = 1
                    
                    tmp_pgd1 = np.zeros((self.num_states, 1))
                    tmp_pgd2 = np.zeros((self.num_states, 1))
                    tmp_pgd3 = np.zeros((self.num_states, self.num_emissions + 1, self.num_emissions + 1))
                    tmp_pgd4 = np.zeros((self.num_states, self.num_emissions + 1, self.num_emissions + 1))
                    tmp_pgd_lik = np.zeros((self.num_states, 1))
                    
                    hess_diag_emit = np.zeros((self.num_states, self.num_emissions, num_total_bins))
                    
                    self._m_step_emission(new_stim)
                            
                    pgd_lik = np.sum(tmp_pgd1, axis = 0)
                    
                    if self.train_bins.size != 0:
                        for trial in range(0, len(X)):
                            new_stim[trial]['data'] = X[trial]
                    
                    for i in range(0, self.num_states):
                        [tmp_pgd1[i], hess_d] = _emit_learning_stats(np.reshape(self.emit_w_[i, :, :].T, (self.emit_w_.shape[1] * self.emit_w_.shape[2], 1), order = 'F'), new_stim, i, self.get_params())
                        hess_diag_emit[i, :, :] = np.reshape(hess_d, (hess_diag_emit.shape[2], hess_diag_emit.shape[1]), order = 'F').T
                        
                        [tmp_pgd_lik[i], tmp_pgd1[i], tmp_pgd2[i], tmp_pgd3_temp, tmp_pgd4_temp] = _emit_likelihood(np.reshape(self.emit_w_[i, :, :].T, (self.emit_w_.shape[1] * self.emit_w_.shape[2], 1), order = 'F'), new_stim, i)
                        tmp_pgd3[i, :, :] = tmp_pgd3_temp
                        tmp_pgd4[i, :, :] = tmp_pgd4_temp

                    pgd_prob1 = np.sum(tmp_pgd1, axis = 0)
                    pgd_prob2 = np.sum(tmp_pgd2, axis = 0)
                    pgd_prob3 = np.sum(tmp_pgd3, axis = 0)
                    pgd_prob4 = np.sum(tmp_pgd4, axis = 0)
            else:
                pgd_lik = 0
                pgd_prob1 = 0
                               
            # Gradient descent for the transition filter
            print('Fitting state transition filters')
        
            new_stim = []
            
            for trial in range(0, len(X)):
                new_stim.append({'gamma' : gamma[trial], 'xi' : xi[trial], 'num_states' : self.num_states})
                
                if self.GLM_transitions == True:
                    if self.train_bins.size == 0:
                        new_stim[trial]['data'] = X[trial]
                        new_stim[trial]['num_total_bins'] = num_total_bins
                    else:
                        new_stim[trial]['data'] = X[trial][self.train_bins, :]
                        new_stim[trial]['num_total_bins'] = len(self.train_bins)
                else:
                    new_stim[trial]['data'] = X[trial][-1, :]
                    new_stim[trial]['num_total_bins'] = 1
            
            tmp_tgd = np.zeros(self.num_states)
        
            if self.evaluate == True:
                for i in range(0, self.num_states):
                    tmp_tgd[i] = _trans_learning_fun(np.reshape(self.trans_w_[i, :, :].T, (self.trans_w_.shape[1] * self.trans_w_.shape[2], 1), order = 'F'), new_stim, i, self.get_params())[0]
                
                tgd_lik = np.sum(tmp_tgd, axis = 0)
            
            else:
                self._m_step_transition(new_stim)
                
                if self.train_bins.size != 0:
                    for trial in range(0, len(X)):
                        new_stim[trial]['data'] = X[trial]
                        
                hess_diag_trans = np.zeros((self.num_states, self.num_states, num_total_bins))
        
                for i in range(0, self.num_states):
                    tmp_tgd[i] = _trans_learning_fun(np.reshape(self.trans_w_[i, :, :].T, (self.trans_w_.shape[1] * self.trans_w_.shape[2], 1), order = 'F'), new_stim, i, self.get_params())[0]
        
                    if self.num_states > 1:
                        [tmp_tgd[i], hess_d] = _trans_learning_stats(np.reshape(self.trans_w_[i, :, :].T, (self.trans_w_.shape[1] * self.trans_w_.shape[2], 1), order = 'F'), new_stim, i, self.get_params())
                        hess_diag_trans[i, :, :] = np.reshape(hess_d, (hess_diag_trans.shape[2], hess_diag_trans.shape[1])).T
                    else:
                        hess_diag_trans = 0
                
                tgd_lik = np.sum(tmp_tgd, axis = 0)
                
            # We don't need to gradient descent for straight-up regressions, just need to regress!
            # we're just doing weighted least squares here: https://en.wikipedia.org/wiki/Least_squares#Weighted_least_squares
            # We need to see how much data we need to accurately reconstruct these filters and why the smooth asd is failing so badly so often
            if self.analog_flag == True:
                print('Fitting analog emission filters')
                
                new_stim = []
                
                ar_corr1 = np.zeros((self.num_states, num_analog_params))
                ar_corr2 = np.zeros(num_analog_params)
        
                if self.evaluate == True:
                                        
                    for analog_num in range(0, num_analog_params):
                        for trial in range(0, len(X)):
                            new_stim.append({'data' : X[trial], 'num_total_bins' : num_total_bins})

                            new_stim[trial]['symb'] = y_analog[trial][analog_num, :]
                            new_stim[trial]['good_emit'] = ~np.isnan(y_analog[trial][analog_num, :])
                   
                        [these_stim, these_symb, these_gamma] = _collect_WLS_info(new_stim)
                        
                        for states in range(0, self.num_states):
                            ar_corr1[states, analog_num] = _w_corr(these_stim * self.analog_emit_w_[states, analog_num, :], these_symb, these_gamma[states, :].T)
                        
                        ar_corr2[analog_num] = np.sum(np.mean(these_gamma, axis = 1) * ar_corr1[:, analog_num], axis = 0)
                    
                else:
                    for analog_num in range(0, num_analog_params):
                        print('Fitting filter ' + str(analog_num) + '/' + str(num_analog_params))
                        
                        for trial in range(0, len(X)):
                            new_stim.append({'num_total_bins' : num_total_bins, 'data' : X[trial]})

                            new_stim[trial]['symb'] = y_analog[trial][analog_num, :]
                            new_stim[trial]['good_emit'] = ~np.isnan(y_analog[trial][analog_num, :])
                            
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
                        ae_w = np.zeros((num_analog_iter, self.num_states, self.analog_emit_w_.shape[2]))
                        ae_std = np.zeros((num_analog_iter, self.num_states, self.analog_emit_w_.shape[2]))
                        iter_weight = np.zeros((num_analog_iter, self.num_states))
                        
                        for nai in range(0, num_analog_iter):
                            use_stim = randomized_stim[iter_stim[nai, 0]:iter_stim[nai, 1]]
                            
                            for states in range(0, self.num_states):
                                if self.use_ASD == True:
                                    if (these_stim.shape[1] % self.num_filter_bins) == 1:
                                        [out_weights, ASD_stats] = _fast_ASD_weighted_group(these_stim[use_stim, :], these_symb[use_stim], these_gamma[states, use_stim], [np.ones((np.round(these_stim.shape[1] / self.num_filter_bins), 1)) * self.num_filter_bins, [1]], 2)
                                    else:
                                        [out_weights, ASD_stats] = _fast_ASD_weighted_group(these_stim[use_stim, :], these_symb[use_stim], these_gamma[states, use_stim], [np.ones((np.round(these_stim.shape[1] / self.num_filter_bins), 1)) * self.num_filter_bins], 2)

                                    ae_w[nai, states, :] = out_weights
                                    ae_std[nai, states, :] = ASD_stats['L_post_diag']
                                else:
                                    [out_weights, out_std] = _weighted_LS_by_state(these_stim[use_stim, :], these_symb[use_stim], these_gamma[states, use_stim], 10)
                                    ae_w[nai, states, :] = out_weights
                                    ae_std[nai, states, :] = out_std

                                iter_weight[nai, states] = np.sum(these_gamma[states, use_stim], axis = 0)
                                ar_corr1[states, analog_num] = 0
 
                        for states in range(0, self.num_states):
                            self.analog_emit_w_[states, analog_num, :] = np.sum(ae_w[:, states, :] * np.tile(np.expand_dims(np.expand_dims(iter_weight[:, states] / np.sum(iter_weight[:, states], axis = 0), axis = 1), axis = 2), (1, 1, self.analog_emit_w_.shape[2])), axis = 0)
                            self.analog_emit_std_[states, analog_num, :] = np.sum(ae_std[:, states, :] * np.tile(np.expand_dims(np.expand_dims(iter_weight[:, states] / np.sum(iter_weight[:, states], axis = 0), axis = 1), axis = 2), (1, 1, self.analog_emit_std_.shape[2])), axis = 0)
        
                        ar_corr1[states, analog_num] = 0
                        ar_corr2[analog_num] = 0
                        
            # Now we have done the E and the M steps! Just save the likelihoods...
            symb_likelihood = 0
            analog_likelihood = 0
            trans_likelihood = 0
                        
            if self.analog_flag == True:
                self.analog_lik_ = _GLMHMM_analog_lik(self.analog_emit_w_, X, y_analog, num_analog_emit)
            
            for trial in range(0, total_trials):
                if self.symb_exists == True and self.fit_emissions == True:
                    self.symb_lik_.append(_GLMHMM_symb_lik(self.emit_w_, X[trial], y[trial]))
                    self.trans_lik_.append(_GLMHMM_trans_lik(self.trans_w_, X[trial]))
                    symb_likelihood = symb_likelihood + -np.sum(np.sum(gamma[trial] * np.log(self.symb_lik_[trial]), axis = 0), axis = 0)
                    trans_likelihood = trans_likelihood + -np.sum(np.sum(np.sum(xi[trial] * np.log(self.trans_lik_[trial][:, :, 1:]), axis = 0), axis = 0), axis = 0)
                
                if self.analog_flag == True:
                    analog_prod = np.prod(np.array(self.analog_lik_[trial]), axis = 0)
                    analog_prod[analog_prod < np.finfo(float).eps] = np.finfo(float).eps
                    analog_likelihood = analog_likelihood + -np.sum(np.sum(gamma[trial] * np.log(analog_prod), axis = 0), axis = 0)
    
            # Basic log likelihood: sum(gamma(n) * log(gamma(n))) + tgd_lik + pgd_lik
            basic_likelihood = np.zeros(total_trials)
            
            for i in range(0, total_trials):
                gamma[i][gamma[i][:, 0] == 0, 0] = np.finfo(float).eps
                basic_likelihood[i] = -np.sum(gamma[i][:, 0] * np.log(gamma[i][:, 0]), axis = 0)
    
            loglik[ind + 1] = np.sum(basic_likelihood, axis = 0) + symb_likelihood + trans_likelihood + analog_likelihood
            
            # Saving variables
            output.append({'emit_w' : self.emit_w_, 'trans_w': self.trans_w_})
            
            if self.symb_exists == True and self.fit_emissions == True:
                output[ind]['symb_likelihood'] = symb_likelihood
                output[ind]['trans_likelihood'] = trans_likelihood
                
                output[ind]['pgd_prob1'] = pgd_prob1
                output[ind]['pgd_prob2'] = pgd_prob2
                output[ind]['pgd_prob3'] = pgd_prob3
                output[ind]['pgd_prob4'] = pgd_prob4
                
                if 'hess_diag_emit' in locals():
                    output[ind]['hess_diag_emit'] = np.lib.scimath.sqrt(hess_diag_emit)
                    
                if 'hess_diag_trans' in locals():
                    output[ind]['hess_diag_trans'] = np.lib.scimath.sqrt(hess_diag_trans)
            
            if self.analog_flag == True:
                output[ind]['analog_emit_w'] = self.analog_emit_w_
                output[ind]['analog_emit_std'] = np.sqrt(self.analog_emit_std_)
                output[ind]['analog_likelihood'] = analog_likelihood
                output[ind]['ar_corr1'] = ar_corr1
                output[ind]['ar_corr2'] = ar_corr2
                        
            output[ind]['tgd_lik'] = tgd_lik
            output[ind]['pgd_lik'] = pgd_lik
            output[ind]['loglik'] = loglik[1:ind + 1]

            parameters_dict = self.get_params()
            parameters_dict.pop('random_state')
            savemat(self.output_filename, {'output' : output, 'parameters' : parameters_dict})
    
            print('Log likelihood: ' + str(loglik[ind + 1]))
        
            # Now do this for not just the loglik but *each* of the likelihoods individually
            # I have been stopping if the % change in log likelihood is below some threshold
            if (abs(loglik[ind + 1] - loglik[ind]) / abs(loglik[ind]) < self.tol):
                if last_try == True:
                    loglik = loglik[1: ind + 1]
                    
                    analog_emit_w_ASD = np.zeros((self.num_states, num_analog_params, 1))
                    analog_emit_std_ASD = np.zeros((self.num_states, num_analog_params, 1))
        
                    for analog_num in range(0, num_analog_params):
                        print('Fitting filter ' + str(analog_num) + '/' + str(num_analog_params))
                        
                        new_stim = []
                        
                        for trial in range(0, len(X)):
                            new_stim.append({'symb' : y_analog[trial][analog_num, :]})
                            new_stim[trial]['good_emit'] = ~np.isnan(y_analog[trial][analog_num, :])
                            
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
                        ae_w = np.zeros((num_analog_iter, self.num_states, self.analog_emit_w_.shape[2]))
                        ae_std = np.zeros((num_analog_iter, self.num_states, self.analog_emit_w_.shape[2]))
                        iter_weight = np.zeros((num_analog_iter, self.num_states))
                        ar_corr1 = np.zeros((self.num_states, num_analog_params))
                        
                        for nai in range(0, num_analog_iter):
                            use_stim = randomized_stim[iter_stim[nai, 0]:iter_stim[nai, 1]]
                            
                            for states in range(0, self.num_states):
                                [out_weights, ASD_stats] = _fast_ASD_weighted_group(these_stim[use_stim, :], these_symb[use_stim], these_gamma[states, use_stim], [np.ones((np.round(these_stim.shape[1] / self.num_filter_bins), 1)) * self.num_filter_bins, [1]], 2)

                                ae_w[nai, states, :] = out_weights
                                ae_std[nai, states, :] = ASD_stats['L_post_diag']

                                iter_weight[nai, states] = np.sum(these_gamma[states, use_stim], axis = 0)
                                ar_corr1[states, analog_num] = 0
                                
                        for states in range(0, self.num_states):
                            analog_emit_w_ASD[states, analog_num, :] = np.sum(ae_w[:, states, :] * np.tile(np.expand_dims(np.expand_dims(iter_weight[:, states] / np.sum(iter_weight[:, states], axis = 0), axis = 1), axis = 2), (1, 1, self.analog_emit_w_.shape[2])), axis = 0)
                            analog_emit_std_ASD[states, analog_num, :] = np.sum(ae_std[:, states, :] * np.tile(np.expand_dims(np.expand_dims(iter_weight[:, states] / np.sum(iter_weight[:, states], axis = 0), axis = 1), axis = 2), (1, 1, self.analog_emit_std_.shape[2])), axis = 0)
        
                    output[ind]['analog_emit_w_ASD'] = analog_emit_w_ASD
                    output[ind]['analog_emit_std_ASD'] = analog_emit_std_ASD
        
                    print('Change in log likelihood is below threshold!')
                    self.converged_ = True
                    self.n_iter_ = ind + 1
                    
                    break
                
                else:
                    last_try = True
                    
                    if effective_ind < 4:
                        # Since the regularization schedule starts here...
                        effective_ind = 5
                    else:
                        effective_ind = effective_ind + 1
            
            else:
                effective_ind = effective_ind + 1
                last_try = False
        
            if self.auto_anneal == True:
                this_lambda = _regularization_schedule(effective_ind)
                
                self.trans_lambda = this_lambda
                self.emit_lambda = this_lambda
        
            if self.evaluate == True or self.get_error_bars == True:
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
        
        if type(self.filter_offset) != int or self.filter_offset < 1:
            raise ValueError("Invalid value for 'filter_offset': %d "
                             "It must be an integer value greater than one"
                             % self.filter_offset)
        
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
        self.analog_emit_std_ = np.array([])
        self.trans_w_ = np.zeros((self.num_states, self.num_states, self.num_filter_bins * self.num_feedbacks + 1))     # states x states x filter bins (diagonals are ignored!)
                        
        for ss1 in range(0, self.num_states):
            for ff in range(0, self.num_feedbacks):           
                for ee in range(0, self.num_emissions - 1):
                    self.emit_w_[ss1, ee, (ff - 1) * self.num_filter_bins + np.arange(self.num_filter_bins)] = np.exp(-np.arange(self.num_filter_bins) / self.num_filter_bins) * np.round(self.random_state.rand(1) * 2 - 1)
            
                for ss2 in range(0, self.num_states):
                    self.trans_w_[ss1, ss2, (ff - 1) * self.num_filter_bins + np.arange(self.num_filter_bins)] = np.exp(-np.arange(self.num_filter_bins) / self.num_filter_bins) * np.round(self.random_state.rand(1) * 2 - 1)
        
        self.symb_lik_ = []
        self.analog_lik_ = []
        self.trans_lik_ = []
              
        if self.CV_regularize == False:
            self.regular_schedule_ = 1
            self.regular_schedule_ind_ = 0
            
            self.train_data_ = np.arange(0, len(X))
            self.test_data_ = np.array([])
        else:
            self.regular_schedule_ = np.logspace(-4, 1, num = 10)
            self.regular_schedule_ind_ = 7

            rp_data = np.random.permutation(len(X))
            self.train_data_ = rp_data[np.ceil(0.25 * len(rp_data)):]
            self.test_data_ = rp_data[0:np.ceil(0.25 * len(rp_data))]
            
        if self.cross_validate == False:
            if self.CV_regularize == False:
                self.train_data_ = np.arange(0, len(X))
                self.test_data_ = np.array([])
        else:
            rp_data = np.random.permutation(len(X))
            self.train_data_ = rp_data[np.ceil(0.25 * len(rp_data)):]
            self.test_data_ = rp_data[0:np.ceil(0.25 * len(rp_data))]
        
        if self.add_filters == False:
            self.emit_w_init_ = copy.copy(self.emit_w_)
            self.analog_emit_w_init_ = copy.copy(self.analog_emit_w_)
            self.analog_emit_std_init_ = copy.copy(self.analog_emit_std_)
            self.trans_w_init_ = copy.copy(self.trans_w_)
        else:
            self.emit_w_init_ = np.array([])
            self.analog_emit_w_init_ = np.array([])
            self.analog_emit_std_init_ = np.array([])
            self.trans_w_init_ = np.array([])
         
    def _e_step(self, X, prior, gamma, xi, total_trials):
        """E step.
        """
        
        for trial in range(0, total_trials):
             # Maybe first compute likelihoods for the symbols?
             if self.analog_flag == True and self.symb_exists == True:
                 emit_likelihood = self.symb_lik_[trial] * np.prod(self.analog_lik_[trial], axis = 0)
             elif self.symb_exists == True:
                 emit_likelihood = self.symb_lik_[trial]
             elif self.analog_flag == True:
                 emit_likelihood = np.prod(self.analog_lik_[trial], axis = 0)

             # Things get funky if the likelihood is exactly 0
             emit_likelihood[emit_likelihood < np.finfo(float).eps * 1e3] = np.finfo(float).eps * 1e3    # Do we need this?

             # Use the forward-backward algorithm to estimate the probability of being in a given state (gamma),
             # probability of transitioning between states (xi), and hold the prior for initialization of next round of fitting
             prior[trial], gamma[trial], xi[trial], alpha1, alpha2, scale, scale_a, score = _compute_trial_expectation(prior[trial], emit_likelihood, self.trans_lik_[trial])
    
        return prior, gamma, xi        
       
    def _m_step_emission(self, new_stim):
        """M step for emitted symbols.
        """
        
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        # I am using the scipy minimization function and passing the analytic value and the gradient to it
        # NOTE: I also could compute the Hessian but in my experience, it ends up not speeding the fitting up much because it is very slow and memory intensive to compute on each iteration
        # NOTE: This also returns the inverse Hessian so we can get the error bars from that if we want to       
        for i in range(0, self.num_states):
            if self.num_steps == 1:
                outweights = minimize(lambda x: _emit_learning_fun(x, new_stim, i, self.get_params()), np.reshape(self.emit_w_[i, :, :].T, (self.emit_w_.shape[1] * self.emit_w_.shape[2]), order = 'F'), jac = True, method = 'BFGS', options = {'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-08, 'maxiter': None, 'disp': True, 'return_all': True})
            else:
                outweights = minimize(lambda x: _emit_multistep_learning_fun(x, new_stim, i, self.get_params()), np.reshape(self.emit_w_[i, :, :].T, (self.emit_w_.shape[1] * self.emit_w_.shape[2]), order = 'F'), jac = True, method = 'BFGS', options = {'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-08, 'maxiter': None, 'disp': True, 'return_all': True})
            
            self.emit_w_[i, :, :] = np.reshape(outweights['x'], (self.emit_w_.shape[2], self.emit_w_.shape[1]), order = 'F').T   # Make sure this is reformatted properly!!!
    
    def _m_step_transition(self, new_stim):
        """M step for state transition.
        """
        
        for i in range(0, self.num_states):
            if self.train_bins.size == 0:
                outweights = minimize(lambda x: _trans_learning_fun(x, new_stim, i, self.get_params()), np.reshape(self.trans_w_[i, :, :].T, (self.trans_w_.shape[1] * self.trans_w_.shape[2]), order = 'F'), jac = True, method = 'BFGS', options = {'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-08, 'maxiter': None, 'disp': True, 'return_all': True})
                self.trans_w_[i, :, :] = np.reshape(outweights['x'], (self.trans_w_.shape[2], self.trans_w_.shape[1]), order = 'F').T
            else:
                outweights = minimize(lambda x: _trans_learning_fun(x, new_stim, i, self.get_params()), np.reshape(self.trans_w_[i, :, self.train_bins].T, (self.trans_w_.shape[1] * len(self.train_bins)), order = 'F'), jac = True, method = 'L-BFGS-B', options = {'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': 101, 'maxls': 20})
                self.trans_w_[i, :, self.train_bins] = np.reshape(outweights['x'], (len(self.train_bins), self.trans_w_.shape[1]), order = 'F').T
            
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
                "filter_offset" : self.filter_offset,
                "init_loglik" : self.init_loglik,     
                "smooth_lambda" : self.smooth_lambda,
                "emit_lambda" : self.emit_lambda,
                "trans_lambda" : self.trans_lambda,
                "AR_lambda" : self.AR_lambda,
                "AR_vec" : self.AR_vec,
                "stim_vec" : self.stim_vec,
                "auto_anneal_vec" : self.auto_anneal_vec,
                "auto_anneal_schedule" : self.auto_anneal_schedule,
                "train_bins" : self.train_bins,
                "symb_exists" : self.symb_exists,
                "use_ASD" : self.use_ASD,
                "add_filters" : self.add_filters,
                "fit_emissions" : self.fit_emissions,
                "GLM_emissions" : self.GLM_emissions,
                "GLM_transitions" : self.GLM_transitions,
                "evaluate" : self.evaluate,
                "generate" : self.generate,
                "L2_smooth" : self.L2_smooth,
                "analog_flag" : self.analog_flag,
                "auto_anneal" : self.auto_anneal,
                "anneal_lambda" : self.anneal_lambda,
                "get_error_bars" : self.get_error_bars,
                "CV_regularize" : self.CV_regularize,
                "cross_validate" : self.cross_validate,
                "output_filename" : self.output_filename}
            
    def set_params(self, **parameters):
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        return self