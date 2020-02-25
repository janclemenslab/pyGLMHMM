# pyGLMHMM

A scikit-learn estimator which fits a combination of Multinomial Generalized Linear Model (GLM) and Hidden Markov Model (HMM) to behavioral data.

This repository is the Python implementation of all of the MATLAB code provided here (https://github.com/murthylab/GLMHMM) implementing the GLMHMM method. It was written in a way to maximally fit the object-oriented framework of scikit-learn estimator API while being faithful to the original implementation of the MATLAB code, so the cross-communication between the two kinds of codes would be possible in future.

Here are the quick descriptions of the two main files of this repository:
1) _glm_hmm: is the main program which implements the GLMHMMEstimator class with a few main methods including the "fit" method.
2) plot_glm_hmm: which generates an instance of GLMHMMEstimator object with an initial set of parameters and runs the "fit" method of the object on a random sample data to generate the results. So to test the _glm_hmm code, you should run this code with your prefered initial parameters and input data.

References:
1) Calhoun, A.J., Pillow, J.W. & Murthy, M. Unsupervised identification of the internal states that shape natural behavior. Nat Neurosci 22, 2040–2049 (2019).
2) Escola, S., Fontanini, A., Katz, D. & Paninski, L. Hidden Markov models for the stimulus-response relationships of multistate neural systems. Neural Comput 23, 1071–1132 (2011).

This is the overall structure of the variables of the GLMHMMEstimator class:
    
    Inputs
    ----------
    stim (X) : The stimulus to be used for fitting. These should be in the form of (regressors, time) per trial in a list.
    symb (y) : The emitted discrete symbols to be fitted. These should be numbers from 0...N-1 (N: the number of possible outputs, i.e.     song types) per trial in a list.
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
