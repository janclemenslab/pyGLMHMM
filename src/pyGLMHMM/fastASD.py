import numpy as np

from scipy.sparse import spdiags, csr_matrix
from scipy.optimize import least_squares

def _fast_ASD_weighted_group(x, y, w, nk_grp, min_length, nx_circ = np.nan):
    # Infers independent groups of smooth coefficients using diagonalized ASD prior 
    #
    # Empirical Bayes estimate of regression coefficients under automatic
    # smoothness determination (ASD) prior (also known as Gaussian or
    # squared-exponential kernel), with maximum marginal likelihood estimate of
    # prior covariance parameters.  
    #
    # Implementation: uses Fourier representation of stimuli and ASD covariance
    # matrix, so prior is diagonal.  
    # 
    # INPUT:
    # x - stimulus
    # y - symbol
    # w - gamma
    # nk_grp [1 * 1] - number of elements in each group (assumed to include all coefficients)
    # min_length [1 * 1] or [n_grp * 1], minimum length scale (for all or for each group)
    # nx_circ [n_grp * 1] - circular boundary for each dimension (OPTIONAL)
    #
    # OUTPUT:
    # kest [nk * 1] - ASD estimate of regression weights
    # ASD_stats - dictionary with fitted hyperparameters, Hessian, and posterior covariance
    #
    # Note: does not include a DC term, so should be applied when response and regressors have been standardized to have mean zero
        
    ## ========== Parse inputs and determine what hyperparams to initialize ==========
    
    COND_THRESH = 1e8  # threshold for small eigenvalues
    
    # Compute sufficient statistics
    w = np.diag(w)
    
    dd = {'xx' : np.matmul(np.matmul(x.T, w), x)} # stimulus auto-covariance
    dd['xy'] = np.matmul(np.matmul(x.T, w), y) # stimulus-response cross-covariance
    # dd['yy'] = np.matmul(np.matmul(y.T, w), y) # marginal response variance
    dd['yy'] = np.matmul(y.T, y) # marginal response variance
    dd['n_samps'], nx = x.shape # total number of samples and number of coeffs
    n_grp = len(nk_grp) # number of groups
    
    # Check to make sure same number of elements in 'xx' as group indices
    if np.sum(nk_grp, axis = 0) != nx: 
        print('Stimulus size dd[xx] does not match number of indices in group_id')
    
    # Replicate min_length to vector if needed
    if len(min_length) == 1:
        min_length = np.tile(min_length, (n_grp, 1))
    
    # Set circular boundary for each group of coeffs
    if nx_circ == np.nan:
        nx_circ = np.zeros((n_grp, 1)) # initialize nx_circ
        for i in range(0, n_grp):
            nx_circ_MAX = 1.25 * nk_grp[i] # maximum based on number of coeffs in group
            nx_circ[i] = np.ceil(max(nk_grp[i] + 2 * min_length[i], nx_circ_MAX)) # set based on maximal smoothness
    
    # ----- Initialize range for hyperparameters -----
    
    # Length scale range
    max_length = np.maximum(min_length * 2, nk_grp / 4).T
    length_range = [min(min_length), max(max_length)]
    
    # Rho range
    rho_max = 2 * (dd['yy'] / dd['n_samps']) / np.mean(np.diag(dd['xx']), axis = 0) # ratio of variance of output to intput
    rho_min = min(1, 0.1 * rho_max) # minimum to explore
    rho_range = [rho_min, rho_max]
    
    # Noise variance sigma_n^2
    nsevar_max = dd['yy'] / dd['n_samps'] # marginal variance of y
    nsevar_min = min(1, 0.01 * nsevar_max) # var ridge regression residuals
    nsevar_range = [nsevar_min, nsevar_max]
    
    # Change of variables to tilde rho (which separates rho and length scale)
    trho_range = np.sqrt(2 * np.pi) * rho_range * [min(length_range), max(length_range)]
    
    ## ========== Diagonalize by converting to FFT basis ==========
    
    opts = {'nx_circ' : nx_circ}
    opts['cond_thresh'] = COND_THRESH
    opt1 = opts
    
    # Generate Fourier basis for each group of coeffs
    B_mats = [] # Fourier basis for each group
    w_vecs_per_grp = [] # frequency vector for each group
    
    B_mats_shape_0 = np.zeros(n_grp)
    B_mats_shape_1 = np.zeros(n_grp)
    
    n_w_vec = np.zeros(n_grp)
    
    for i in range(0, n_grp):
        opt1['nx_circ'] = opts['nx_circ'][i] # pass in ju
        _, B_mats_temp, w_vecs_per_grp_temp = _mkcov_ASD_factored([min_length[i], 1], nk_grp(i), opt1)
        
        B_mats.append(B_mats_temp)
        w_vecs_per_grp.append(w_vecs_per_grp_temp)
        
        B_mats_shape_0[i] = B_mats_temp.shape[0]
        B_mats_shape_1[i] = B_mats_temp.shape[1]
        
        n_w_vec[i] = len(w_vecs_per_grp)
        
    Bfft = np.zeros((np.sum(B_mats_shape_0, axis = 0), np.sum(B_mats_shape_1, axis = 0)))
    w_vec = np.array([])
     
    for i in range(0, n_grp):
        Bfft[np.sum(B_mats_shape_0[0:i], axis = 0):np.sum(B_mats_shape_0[0:i + 1], axis = 0), np.sum(B_mats_shape_1[0:i], axis = 0):np.sum(B_mats_shape_1[0:i + 1], axis = 0)] = B_mats[i] # Fourier basis matrices assembled into block diag
        w_vec = np.concatenate((w_vec, w_vecs_per_grp[i]), axis = 0) # group Fourier frequencies assembled into one vec   
        
    dd['xx'] = np.matmul(np.matmul(Bfft.T, dd['xx']), Bfft) # project xx into Fourier basis for each group of coeffs
    dd.xy = np.matmul(Bfft.T, dd['xy']) # project xy into Fourier basis for each group
    
    # Make matrix for mapping hyperparams to Fourier coefficients for each group
    B_grp = np.zeros((np.sum(n_w_vec, axis = 0), n_grp))
    for i in range(0, n_grp):
        B_grp[np.sum(n_w_vec[0:i], axis = 0):np.sum(n_w_vec[0:i + 1], axis = 0), i] = np.ones((n_w_vec[i], 1))
    
    # Compute vector of normalized squared Fourier frequencies
    ww_nrm = np.power(2 * np.pi / np.matmul(B_grp, nx_circ), 2) * np.power(w_vec, 2) # compute normalized DFT frequencies squared
        
    ## ========== Grid search for initial hyperparameters ==========
    
    # Set loss function for grid search
    ii_grp = [[np.ones((n_grp, 1))], [2 * np.ones((n_grp, 1))], [3]] # indices for setting group params
    l_fun_0 = lambda prs: _neglogev_ASD_spectral_group(prs(ii_grp), dd, B_grp, ww_nrm, COND_THRESH) # loss function
    
    # Set up grid
    n_grid = 4 # search a 4 * 4 * 4 grid for initial value of hyperparameters
    rng = [length_range, trho_range, nsevar_range]
    
    # Do grid search and find minimum
    nll_vals, grid_pts = _grid_eval(n_grid, rng, l_fun_0)
    h_prs_00, _, _ = argmin(nll_vals, grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2]) # find minimum
    h_prs_0 = h_prs_00[ii_grp] # initialize hyperparameters for each group
    
    ## ========== Optimize evidence using fmincon ==========
    
    l_fun = lambda prs: _neglogev_ASD_spectral_group(prs, dd, B_grp, ww_nrm, COND_THRESH) # loss function
    LB = [[min_length], [1e-2 * np.ones((n_grp + 1, 1))]] # lower bounds
    UB = np.inf * np.ones((n_grp * 2 + 1, 1)) # upper bounds
    
    h_prs_hat = least_squares(l_fun, h_prs_0, jac = '2-point', bounds = (LB, UB), method = 'trf', 
                  ftol = 1e-08, xtol = 1e-08, gtol = 1e-08, x_scale = 1.0, loss = 'linear', f_scale = 1.0, 
                  diff_step = None, tr_solver = None, tr_options = {}, jac_sparsity = None, max_nfev = None, verbose = 0) # run optimization
        
    ## ========== Compute posterior mean and covariance at maximizer of hyperparams ==========
    
    neglogEv, _, H, mu_FFT, L_post_FFT, ii = l_fun(h_prs_hat)
    kest = np.matmul(Bfft[:, ii], mu_FFT) # inverse Fourier transform of Fourier-domain mean
    
    # Report rank of prior at termination
    print('fast_ASD_weighted_group: terminated with rank of C_prior = ' + str(np.sum(ii, axis = 0)))
    
    # Check if length scale is at minimum allowed range
    if np.any(h_prs_hat[0:n_grp] <= min_length + 0.001):
        print(['Solution is at minimum length scale >>> Consider re-running with shorter min_length'])
    
    # Assemble summary statistics for output 
    # Transform trho back to standard rho
    l_hat = h_prs_hat[0, n_grp] # transformed rho param
    trho_hat = h_prs_hat[n_grp:2 * n_grp] # length scale
    a = np.sqrt(2 * np.pi)
    rho_hat = trho_hat / (a * l_hat) # original rho param
    A_jcb = [[np.identity(n_grp), np.zeros((n_grp, n_grp + 1))], [a * [np.diag(rho_hat), np.diag(l_hat)], np.zeros((n_grp, 1))], [np.zeros((1, n_grp * 2)), 1]] # Jacobian for parameters
    
    ASD_stats = {'rho' : rho_hat} # rho hyperparameter
    ASD_stats['len'] = l_hat # length scale hyperparameter
    ASD_stats['nsevar'] = h_prs_hat[-1] # noise variance
    # ASD_stats['H'] = H # Hessian of hyperparameters
    ASD_stats['H'] = np.matmul(np.matmul(A_jcb.T, H), A_jcb) # Hessian of hyperparameters
    ASD_stats['ci'] = np.sqrt(np.diag(np.linalg.inv(ASD_stats['H']))) # 1SD posterior CI for hyperparameters
    
    ASD_stats['neglogEv'] = neglogEv # negative log evidence at solution
    
    # Just compute diagonal of posterior covariance
    ASD_stats['L_post_diag'] = np.sum(Bfft[:, ii].T * np.matmul(L_post_FFT, Bfft[:, ii].T), axis = 0).T
    
    # If full posterior cov for filter is desired
    ASD_stats['L_post'] = np.matmul(np.matmul(Bfft[:, ii], L_post_FFT), Bfft[:, ii].T)
    
    return kest, ASD_stats

def _mkcov_ASD_factored(prs, nx, opts = np.nan):
    # Factored representation of ASD covariance matrix in Fourier domain
    #
    # Covariance represented as C = U * s_diag * U'
    # where U is unitary (in some larger basis) and s_diag is diagonal
    #
    # C_ij = rho * exp(((i - j)^2 / (2 * l^2))
    #
    # INPUT:
    # prs [2 * 1] - ASD parameters [len = length scale, rho - maximal variance]
    # nx [1 * 1] - number of regression coeffs
    # opts [dict] - options dictionary with:
    #               .nx_circ - number of coefficients for circular boundary
    #               .cond_thresh - threshold for condition number of K (default = 1e8)
    # 
    # Note: nx_circ = nx gives circular boundary
    #
    # OUTPUT:
    # c_diag [ni * 1] - vector with thresholded eigenvalues of C
    # U [ni * nx_circ] - column vectors define orthogonal basis for C (on Reals)
    # w_vec [nx_circ * 1] - vector of Fourier frequencies
    
    length = prs[0]
    rho = prs[1]
    
    # Parse inputs
    if opts == np.nan:
        opts['nx_circ'] = nx + np.ceil(4 * length) # extends support by 4 st_devs of ASD kernel width
        opts['cond_thresh'] = 1e8  # threshold for small eigenvalues 
    
    # Check that nx_circ isn't bigger than nx
    if opts['nx_circ'] < nx:
        print('WARNING >>> mkcov_ASD_factored: nx_circ < nx. Some columns of x will be ignored.')
    
    # Compute vector of Fourier frequencies
    max_freq = np.floor(opts['nx_circ'] / (np.pi * length) * np.sqrt(0.5 * np.log(opts['cond_thresh']))) # max
    
    if max_freq < opts['nx_circ'] / 2:
        w_vec = np.concatenate((np.arange(0, max_freq).T, np.arange(-max_freq, -1).T))
    else:
        # In case cutoff is above max number of frequencies
        n_cos = np.ceil((opts['nx_circ'] - 1) / 2) # neg frequenceis
        n_sin = np.floor((opts['nx_circ'] - 1) / 2) # pos frequencies
        w_vec = np.concatenate((np.arange(0, n_cos), np.arange(-n_sin, -1))).T # vector of frequencies
    
    # Compute diagonal in Fourier domain
    c_diag, _, _, _ = _mkcovdiag_ASD(length, rho, opts['nx_circ'], w_vec ** 2) # compute diagonal and Fourier freqs
    
    # Compute real-valued discrete Fourier basis U
    U, _ = _realfftbasis(nx, opts['nx_circ'], w_vec).T
    
    return c_diag, U, w_vec

def _mkcovdiag_ASD(length, rho, nx_circ, w_vec_sq):
    # Eigenvalues of ASD covariance (as diagonalized in Fourier domain)
    #
    # Compute discrete ASD (RBF kernel) eigenspectrum using frequencies in [0, nx_circ]
    # See _mkcov_ASD_factored for more info
    #
    # INPUT:
    # length - length scale of ASD kernel (determines smoothness)
    # rho - maximal prior variance ("overall scale")
    # nx_circ - number of coefficients to consider for circular boundary 
    # w_vec_sq - vector of squared frequencies for DFT 
    #        
    # OUTPUT:
    # c_diag [nx_circ * 1] - vector of eigenvalues of C for frequencies in w
    # dc_inv [nx_circ * 2] - 1st derivs [dC^-1 / drho, dC^-1 / dlen]
    # dc [nx_circ * 2] - 1st derivs [dC / drho , dC / dlen]
    # ddc_inv [nx_circ * 3] - 2nd derivs of C-1 w.r.t [drho^2, drho * dlen, dlen^2]
    #
    # Note: nx_circ = nx corresponds to having a circular boundary 
        
    # Compute diagonal of ASD covariance matrix
    const = (2 * np.pi / nx_circ) ** 2 # constant 
    ww = w_vec_sq * const # effective frequency vector
    c_diag = np.sqrt(2 * np.pi) * rho * length * np.exp(-0.5 * ww * (length ** 2))
    
    # 1st derivative of inv(c_diag)
    nw = len(w_vec_sq)
    dc_inv = [(-1 / length + length * ww) / c_diag, -(1 / rho) / c_diag] # dC^-1 / dl, dC^-1 / drho
    
    # 1st derivative of c_diag
    dc = [(1 / length - length * ww) * c_diag[:, [0, 0]], (1 / rho) * np.ones(nw)] # dC / dl, dC / drho
                               
    # 2nd derivative of inv(c_diag)
    ddc_inv = [(2 / length ** 2 - ww + (length ** 2) * np.power(ww, 2)), (1 / rho) * (1 / length - length * ww), (2 / rho ^ 2) * np.ones(nw)] / c_diag[:, np.ones(3)] # d^2 C^-1 / dl^2, d^2 C^-1 / drho dl, d^2 C^-1 / drho^2

    return c_diag, dc_inv, dc, ddc_inv

def _mkcovdiag_ASD_std(length, trho, ww_nrm):
    # Eigenvalues of ASD covariance (as diagonalized in Fourier domain)
    #
    # Compute discrete ASD (RBF kernel) eigenspectrum in Fourier domain, and its derivatives w.r.t. to the model parameters trho and length
    #
    # INPUT:
    # length [1 * 1] or [n * 1] - ASD length scales
    # trho [1 * 1] or [n * 1] - Fourier domain prior variance
    # ww_nrm [n * 1] - vector of squared normalized Fourier frequencies
    #        
    # OUTPUT:
    # c_diag [n * 1] - vector of eigenvalues of C for frequencies in w
    # dc_invd_thet [n * 2] - 1st derivs [dC^-1 / dlen, dC^-1 / drho]
    # dcd_thet [n * 2] - 1st derivs [dC / dlen, dC / drho]
    # ddc_invd_thet [n * 3] - 2nd derivs of C^-1 w.r.t [dlen^2, dlen * drho, drho^2]
        
    # Compute diagonal of ASD covariance matrix
    c_diag = trho * np.exp(-0.5 * ww_nrm * (length ** 2))
    
    # 1st derivative of inv(c_diag)
    dc_invd_thet = [(length * ww_nrm) / c_diag, -(1 / trho) / c_diag] # dC^-1 / dl, dC^-1 / drho
    
    # 1st derivative of Cdiag 
    dcd_thet = [-length * ww_nrm * c_diag, (c_diag / trho)] # dC / dl, dC / drho
    
    # 2nd derivative of inv(c_diag)
    ddc_invd_thet = [(ww_nrm + (length ** 2) * np.power(ww_nrm, 2)) / c_diag, -(1 / trho) * (length * ww_nrm) / c_diag, 2 / (np.power(trho, 2) * c_diag)] # d^2 C^-1 / dl^2, d^2 C^-1 / drho dl, d^2 C^-1 / drho^2
                 
    return c_diag, dc_invd_thet, dcd_thet, ddc_invd_thet

def _neglogev_ASD_spectral(prs, dd, ww_nrm, cond_thresh):
    # Negative log evidence for ASD regression model in Fourier domain
    #
    # Computes negative log evidence: 
    #       -log P(Y|X, sig^2, C) 
    # under linear-Gaussian model: 
    #       y = x' * w + n, n ~ N(0, sig^2)
    #       w ~ N(0, C),
    # where C is ASD (or RBF or "squared exponential") covariance matrix
    # 
    # INPUT:
    # prs [3 * 1] - ASD parameters [len; rho; nsevar]
    # dd [dict] - sufficient statistics for regression:
    #             .xx - Fourier domain stimulus autocovariance matrix X' * X 
    #             .xy - Fourier domain stimulus-response cross-covariance X' * Y 
    #             .yy - response variance Y' * Y
    #             .n_samps - number of samples 
    # ww_nrm [m * 1] - vector of normalized DFT frequencies, along each dimension
    # cond_thresh [1 * 1] - threshold for condition number of K (default = 1e8)
    #
    # OUTPUT:
    # neglogev - negative marginal likelihood
    # grad - gradient
    # H - Hessian
    # mu_post - mean of posterior over regression weights
    # L_post - posterior covariance over regression weights
    # ii - logical vector indicating which DFT frequencies are not pruned
    # C_inv - inverse prior covariance in diagonalized, pruned Fourier space
        
    # Unpack parameters
    length = prs[0]
    trho = prs[1]
    nsevar = prs[2]
    
    # Compute diagonal representation of prior covariance (Fourier domain)
    ii = ww_nrm < 2 * np.log(cond_thresh) / (length ** 2) 
    ni = np.sum(ii, axis = 0) # rank of covariance after pruning
    
    # Build prior covariance matrix from parameters
    # Compute derivatives if gradient and Hessian are requested
    # Compute diagonal of C, 1st and 2nd derivs of C^-1 and C
    [c_diag, dc_inv, dc, ddc_inv] = _mkcovdiag_ASD_std(length, trho, ww_nrm[ii]) # compute diagonal and Fourier freqs
    
    # Prune XX and XY Fourier coefficients and divide by nsevar
    C_inv = spdiags(1 / c_diag, 0, ni, ni) # inverse cov in diagonalized space
    XX = dd['xx'][ii, ii] / nsevar 
    XY = dd['xy'][ii] / nsevar
    
    # Compute neglogli 
    trm1 = -0.5 * (logdet(XX + C_inv) + np.sum(np.log(c_diag), axis = 0) + (dd['n_samps']) * np.log(2 * np.pi * nsevar)) # log-determinant term
    trm2 = 0.5 * (-dd['yy'] / nsevar + np.matmul(XY.T, np.linalg.lstsq(XX + C_inv, XY))) # quadratic term
    neglogev = -trm1 - trm2 # negative log evidence
    
    # Compute neglogli and gradient
    # Compute matrices we need
    L_post_inv = XX + C_inv
    L_post = np.linalg.inv(L_post_inv)
    L_p_diag = np.diag(L_post)
    mu_post = np.matmul(L_post, XY)
        
    # Compute gradient
    # Derivs w.r.t hyperparams rho and length
    dL_dthet = -0.5 * np.matmul(dc_inv.T, c_diag - (L_p_diag + np.power(mu_post, 2)))
    # Deriv w.r.t noise variance 'nsevar'
    RR = 0.5 * (dd['yy'] / nsevar - 2 * np.matmul(mu_post.T, XY) + np.matmul(np.matmul(mu_post.T, XX), mu_post)) / nsevar # squared_residuals / (2 * nsevar^2)
    trace_trm = 0.5 * (ni - dd['n_samps'] - np.sum(L_p_diag / c_diag, axis = 0)) / nsevar
    dL_dnsevar = -trace_trm - RR
    # Combine them into gardient vector
    grad = [dL_dthet, dL_dnsevar]

    # Compute Hessian
    # theta terms (rho and length)
    n_thet = 2  # number of theta variables (rho and length)
    vn = np.ones((1, n_thet)) # vector of 1s of length n_theta
    dL_p_diag = np.matmul(-np.power(L_post, 2), dc_inv) # deriv of diag(L_post) w.r.t thetas
    dmu_post = np.matmul(-(L_post), (dc_inv * mu_post)) # deriv of mu_post w.r.t thetas
    ri, ci, _ = _triuinds(n_thet)  # get indices for rows and columns of upper triangle
    trm1_stuff = -0.5 * (dc - (dL_p_diag + 2 * dmu_post * mu_post[:, vn]))
    ddL_ddthet_trm1 = np.sum(trm1_stuff[:, ri] * dc_inv[:, ci], axis = 0).T
    ddL_ddthet_trm2 = -0.5 * ddc_inv.T * (c_diag - (L_p_diag + np.power(mu_post, 2)))
    ddL_ddthet = ddL_ddthet_trm1 + ddL_ddthet_trm2

    # nsevar term
    dL_p_diag_v = np.sum(L_post * np.matmul(L_post, XX), axis = 1) / nsevar # deriv of diag(L_post) w.r.t nsevar
    dmu_post_v = -np.matmul(L_post, (mu_post / c_diag)) / nsevar # deriv of mu_post w.r.t nsevar
    ddL_dv = -(dL_dnsevar / nsevar - RR / nsevar - np.sum(dL_p_diag_v / c_diag, axis = 0) / (2 * nsevar) + np.matmul((-XY + np.matmul(XX, mu_post)).T, dmu_post_v) / nsevar)  # 2nd deriv w.r.t. nsevar
        
    # Cross term theta - nsevar
    ddL_dtheta_v = 0.5 * np.matmul(dc_inv.T, (dL_p_diag_v + 2 * dmu_post_v * mu_post))
    
    # Assemble Hessian 
    H = _unvec_sym_mtx_from_triu([ddL_ddthet, ddL_dtheta_v, ddL_dv])

    return neglogev, grad, H, mu_post, L_post, ii, C_inv

def _neglogev_ASD_spectral_group(prs, dd, B_grp, ww_nrm, cond_thresh):
    # Negative log evidence for ASD regression model with pre-diagonalized inputs
    #
    # Computes negative log evidence: 
    #    -log P(Y|X, sig^2, C) 
    # under linear-Gaussian model: 
    #       y = x' * w + n, n ~ N(0, sig^2)
    #       w ~ N(0, C)
    # where C is ASD (or RBF or "squared exponential") covariance matrix
    # 
    # INPUT:
    # prs [2 * n_grp + 1 * 1] - ASD parameters [rho (marginal var), length(length), nsevar]
    # dd [dict] - data dictionary with:
    #             .xx - stimulus autocovariance matrix X' * X in Fourier domain
    #             .xy - stimulus-response cross-covariance X' * Y in Fourier domain
    #             .yy - response variance Y' * Y
    #             .n_samps - number of samples 
    # B_grp - sparse matrix mapping the rho and length hyperparam vectors to coeffs
    # w_vec_sq - vector of squared Fourier frequencies
    # cond_thresh - threshold for condition number of K (default = 1e8)
    #
    # OUTPUT:
    # neglogev - negative marginal likelihood
    # grad - gradient
    # H - Hessian
    # mu_post - mean of posterior over regression weights
    # L_post - posterior covariance over regression weights
    # ii - logical vector indicating which DFT frequencies are not pruned
   
    n_grp = B_grp.shape[1] # number of groups of coefficients
    
    # Unpack parameters
    lens = B_grp * prs[0:n_grp] # vector of rhos for each coeff
    trhos = B_grp * prs[n_grp:2 * n_grp]  # vector of lens for each coeff
    nsevar = prs[-1]
    
    # Find indices for which eigenvalues too small
    ii = ww_nrm < (2 * np.log(cond_thresh) / np.power(lens, 2))
    ni = np.sum(ii, axis = 0) # number of non-zero DFT coefs / rank of covariance after pruning
    
    # Prune XX and XY Fourier coefficients and divide by nsevar
    B_grpred = B_grp[ii, :] # reduced group indices matrix
    XX = dd['xx'][ii, ii] / nsevar 
    XY = dd['xy'][ii] / nsevar
    
    # Build prior covariance matrix from parameters
    # Compute diagonal of C, 1st and 2nd derivs of C^-1 and C
    [c_diag, dc_inv, dc_diag, ddc_inv] = _mkcovdiag_ASD_std(lens[ii], trhos[ii], ww_nrm[ii]) 
    
    C_inv = spdiags(1 / c_diag, 0, ni, ni) # inverse cov in diagonalized space
    
    # Compute negative loglikelihood only    
    trm1 = -0.5 * (logdet(XX + C_inv) + np.sum(np.log(c_diag), axis = 0) + (dd['n_samps']) * np.log(2 * np.pi * nsevar)) # log-determinant term
    trm2 = 0.5 * (-dd['yy'] / nsevar + np.matmul(XY.T, np.linalg.lstsq(XX + C_inv, XY))) # quadratic term
    neglogev = -trm1 - trm2 # negative log evidence
    
    # Compute negative loglikelihood and gradient 
    # Make stuff we will need
    L_post_inv = XX + C_inv
    L_post = np.linalg.inv(L_post_inv)
    L_p_diag = np.diag(L_post)
    mu_post = np.matmul(L_post, XY)
    
    # Compute negative logevidence
    trm1 = -0.5 * (logdet(L_post_inv) + np.sum(np.log(c_diag), axis = 0) + (dd['n_samps']) * np.log(2 * np.pi * nsevar)) # log-determinant term
    trm2 = 0.5 * (-dd['yy'] / nsevar + np.matmul(XY.T, np.linalg.lstsq(L_post, XY))) # quadratic term
    neglogev = -trm1 - trm2 # negative log evidence
    
    # Compute gradient
    # Derivs w.r.t hyperparams rho and length
    dL_dthet = -0.5 * B_grpred * np.matmul(dc_inv, c_diag - (L_p_diag + np.power(mu_post, 2)))
    # Deriv w.r.t noise variance 'nsevar'
    RR = 0.5 * (dd['yy'] / nsevar - 2 * np.matmul(mu_post.T, XY) + np.matmul(np.matmul(mu_post.T, XX), mu_post)) / nsevar # squared_residuals / (2 * nsevar^2)
    trace_trm = 0.5 * (ni - dd['n_samps'] - np.sum(L_p_diag / c_diag, axis = 0)) / nsevar
    dL_dnsevar = -trace_trm - RR
    # Combine them into gardient vector
    grad = [dL_dthet, dL_dnsevar]
        
    # Compute Hessian
    # theta terms (rho and length)
    n_thet = 2  # number of theta variables (rho and length)
    vn = np.ones((1, n_thet * n_grp)) # vector of 1s of length n_theta
    
    # Make matrix with dc_inv for each parameter in a second column
    M_dc_inv = [[B_grpred * dc_inv[:, 0]], [B_grpred * dc_inv[:, 1]]]
    M_dc_diag = [[B_grpred * dc_diag[:, 0]], [B_grpred * dc_diag[:, 1]]]
    M_ddc_inv = [[B_grpred * ddc_inv[:, 0]], [B_grpred * ddc_inv[:, 1]], [B_grpred * ddc_inv[:, 2]]]
    
    # Derivs of posterior covariance diagonal and posterior mean w.r.t theta
    dL_p_diag = -np.matmul(np.power(L_post, 2), M_dc_inv) # deriv of diag(L_post) w.r.t thetas
    dmu_post = -np.matmul(L_post, np.matmul(M_dc_inv, mu_post)) # deriv of mu_post w.r.t thetas
    ri, ci, _ = _triuinds(n_thet * n_grp)  # get indices for rows and columns of upper triangle
    trm1_stuff = -0.5 * (M_dc_diag - (dL_p_diag + 2 * dmu_post * mu_post[:, vn]))
    ddL_ddthet_trm1 = np.sum(trm1_stuff[:, ri] * M_dc_inv[:, ci], axis = 0).T 
    ddL_ddthet_trm2 = -0.5 * np.matmul(M_ddc_inv.T, c_diag - (L_p_diag + np.power(mu_post, 2)))
    ddL_ddthet = _unvec_sym_mtx_from_triu(ddL_ddthet_trm1) # form Hessian

    # Generate indices needed to insert the trm2 (those in upper triangle)
    ii1 = sub2ind(n_grp * n_thet * [1, 1], np.arange(1, n_grp), np.arange(1, n_grp)).T # indices for drho^2
    ii2 = sub2ind(n_grp * n_thet * [1, 1], np.arange(1, n_grp), n_grp + np.arange(1, n_grp)).T # indices for drho, dlen
    ii3 = sub2ind(n_grp * n_thet * [1, 1], n_grp + np.arange(0, n_grp), n_grp + np.arange(0, n_grp)).T
    ii_dep = [ii1, ii2, ii3]
    ddL_ddthet[ii_dep] = ddL_ddthet[ii_dep] + ddL_ddthet_trm2
    ddL_ddthet = _vec_mtx_triu(ddL_ddthet)

    # nsevar term          
    dL_p_diag_v = np.sum(L_post * np.matmul(L_post, XX), axis = 1) / nsevar # deriv of diag(L_post) w.r.t nsevar
    dmu_post_v = -np.matmul(L_post, (mu_post / c_diag)) / nsevar # deriv of mu_post w.r.t nsevar
    ddL_dv = -(dL_dnsevar / nsevar - RR / nsevar - np.sum(dL_p_diag_v / c_diag, axis = 0) / (2 * nsevar) + np.matmul((-XY + np.matmul(XX, mu_post)).T, dmu_post_v) / nsevar)  # 2nd deriv w.r.t. nsevar

    # Cross term theta - nsevar
    ddL_dtheta_v = 0.5 * np.matmul(M_dc_inv.T, dL_p_diag_v + 2 * dmu_post_v * mu_post)
    
    # Assemble Hessian 
    H = _unvec_sym_mtx_from_triu([ddL_ddthet, ddL_dtheta_v, ddL_dv])
    
    return neglogev, grad, H, mu_post, L_post, ii

def _realfftbasis(nx, nn = np.nan, w_vec = np.nan):
    # Basis of sines + cosines for nn-point discrete fourier transform (DFT)
    # For real-valued vector x, realfftbasis(nx, nn) * x is equivalent to realfft(x, nn) 
    #
    # INPUT:
    # nx - number of coefficients in original signal
    # nn - number of coefficients for FFT (should be >= nx, so FFT is zero-padded)
    # w_vec (optional) - frequencies: positive = cosine
    #
    # OUTPUT:
    # B [nn * nx] or [nw * nx] - DFT basis 
    # w_vec - frequencies associated with rows of B
    #
    # See also: realfft, realifft
    
    if nn == np.nan:
        nn = nx
        
    if w_vec == np.nan:
        # Make frequency vector
        n_cos = np.ceil((nn + 1) / 2) # number of cosine terms (positive freqs)
        n_sin = np.floor((nn - 1) / 2) # number of sine terms (negative freqs)
        w_vec = np.concatenate(np.arange(0, (n_cos - 1)), np.arange(-n_sin, -1)).T # vector of frequencies
    
    # Divide into pos (for cosine) and neg (for sine) frequencies
    w_cos = w_vec[w_vec >= 0] 
    w_sin = w_vec[w_vec < 0]
    
    x = np.arange(0, nx - 1).T # spatial pixel indices
    
    if w_sin.size != 0:
        B = [np.cos((w_cos * 2 * np.pi / nn) * x.T), np.sin((w_sin * 2 * np.pi / nn) * x.T)] / np.sqrt(nn / 2)
    else:
        B = np.cos((w_cos * 2 * np.pi / nn) * x.T) / np.sqrt(nn / 2)
    
    # Make DC term into a unit vector
    izero = (w_vec == 0) # index for DC term
    B[izero, :] = B[izero, :] / np.sqrt(2)
    
    # If nn is even, make Nyquist term (highest cosine term) a unit vector
    if nn / 2 == max(w_vec):
        n_cos = np.ceil((nn + 1) / 2) # this is the index of the Nyquist freq
        B[n_cos, :] = B[n_cos, :] / np.sqrt(2)
    
    return B, w_vec

def _triuinds(nn, k = 0):
    # Extracts row and column indices of upper triangular elements of a matrix of size nn (default k = 0 if not provided)
    # 
    # INPUT:
    # nn - side-length of square matrix
    # k - which diagonal to start at (0 = main diagonal) (OPTIONAL)
    #
    # OUTPUT:
    # ii - indices of entries of upper triangle (from 1 to nn^2)
    # ri, ci - row and column indices of  upper triangle
 
    ri, ci = np.nonzero(np.triu(np.ones(nn), k))
    ii = (ci + 1) * nn + (ri + 1) - 1

    return ri, ci, ii

def _unvec_sym_mtx_from_triu(v):
    # Takes entries in vector v, which are the upper triangular elements in a symmetric matrix, and forms the symmetric matrix
    # 
    # INPUT:
    # v - vector of upper diagonal entries in symmetric matrix
    
    n = np.floor(np.sqrt(2 * len(v))) # side-length of matrix
    M = np.zeros(n); # initialize matrix
    
    ri, _, _ = _triuinds(n) # get indices into matrix
    M[ri] = v
    M = M + M.T - np.diag(np.diag(M))
    
    return M

def _vec_mtx_triu(M, k = 0):
    # Takes the upper triangular of a matrix and return it as a vector
    #
    # INPUT:
    # M - matrix
    # k - which diagonal to use (optional, default = 0)

    ri = _triuinds(M.shape[0], k)
    v = M(ri)
    
    return v

def _grid_eval(n_grid, grid_ranges, fptr):
    # Evaluates a function on a grid of points
    #
    # INPUT:
    # n_grid [1 * 1] - number of grid points
    # grid_ranges [n * 2] - specify range for each dimension of grid
    # fptr - handle for function to evaluate
    #
    # OUTPUT:
    # f_vals - full grid of function values
    # grid_vecs - array whose columns are grid coordinate vectors along each dimension
    
    # Set grid 
    n_dim = grid_ranges.shape[0] # number of dimensions in grid
    grid_vecs = np.zeros((n_grid, n_dim))
    
    for i_dim in range(0, n_dim):
        end_points = grid_ranges[i_dim, :] # endpoints for grid
        grid_vecs[:, i_dim] = np.linspace(end_points[0], end_points[1], n_grid)
    
    # Evaluate function on grid
    size = n_grid * np.ones((1, n_dim)) # size of grid
    n_points = np.prod(size) # total number of points in grid
    f_vals = np.zeros(size) # initialize space for grid values
    
    for idx in range(0, n_points):
        v_inds = ind2subv(size, idx) # indices for grid
        grid_points = grid_vecs[csr_matrix((v_inds, np.arange(0, n_dim), True), shape = [n_grid, n_dim])] # make input vector
        f_vals[idx] = fptr(grid_points) # evaluate
    
    return f_vals, grid_vecs

def logdet(A):
    # Computes the log-determinant of a matrix A
    #
    # This is faster and more stable than using log(det(A))
    #
    # INPUT:
    # A [N * N] - A must be sqaure and positive semi-definite
    
    x = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(A))), axis = 0)
    
    return x

def argmin(yy, **kwargs):
    # Finds argmin of function: values of x1, x2, etc., at which yy achieves its maximum.
    # 
    # INPUT:
    # yy [m * n * p ...] - array of function values
    # x1 [m * 1] - coordinates along 1st dimension of yy
    # x2 [n * 1] - coordinates along 2nd dimension of yy (optional)
    # x3 [p * 1] - coordinates along 3rd dimension of yy (optional)
    # x4 [q * 1] - coordinates along 4th dimension of yy (optional)
    #  
    # OUTPUT:
    # x_val - vector of coordinates at which min is obtained
    # inds - indices at which x_val is maximal
    # y_val - value of function at min
    
    y_size = yy.shape  # get dimensions of yy
    
    # Create indices as needed
    if len(kwargs) == 0:
        for i in range(0, len(y_size)):
            kwargs.append(np.arange(0, y_size[i]).T)
    
    # Check if it is a vector
    if y_size[0] == 1 or y_size[1] == 1:
        y_size = len(y_size)
    
    y_val = np.amin(yy) # find minimum of yy
    inds = np.argmin(yy) # find index of minimum of yy
    
    if len(y_size) == 1: # vector
        x_val = kwargs[0][inds]
    elif len(y_size) == 2: # matrix 
        [i1, i2] = ind2sub(y_size, inds) 
        x_val = [kwargs[0][i1], kwargs[1][i2]].T
        inds = [i1, i2].T
    elif len(y_size) == 3: # 3D array
        [i1, i2, i3] = ind2sub(y_size, inds)
        x_val = [kwargs[0][i1], kwargs[1][i2], kwargs[2][i3]].T
        inds = [i1, i2, i3].T
    elif len(y_size) == 4: # 4D array
        [i1, i2, i3, i4] = ind2sub(y_size, inds)
        x_val = [kwargs[0][i1], kwargs[1][i2], kwargs[2][i3], kwargs[3][i4]].T
        inds = [i1, i2, i3, i4].T

    return x_val, inds, y_val

def ind2subv(size, index):
    # Subscript vector from linear index
    # Returns a vector of the equivalent subscript values corresponding to a single index into an array of size SIZE
    # If INDEX is a vector, then the result is a matrix, with subscript vectors as rows

    n = len(size)
    cum_size = np.cumprod(size)
    prev_cum_size = [1, cum_size[0:-1]]
    index = index - 1
    sub = np.tile(index, [1, n]) % np.tile(cum_size, [len(index), 1])
    sub = np.floor(sub / np.tile(prev_cum_size, [len(index), 1])) + 1
    
    return sub

def sub2ind(array_shape, rows, cols):
    ind = rows * array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    
    return rows, cols