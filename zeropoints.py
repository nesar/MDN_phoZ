import time
import numpy as np
from scipy.stats import norm


def gaussian_vect(zgrid, mu, sigma, weight):
    exp = np.exp(-0.5 * np.add.outer(zgrid, -mu)**2/sigma**2)
    return weight / (sigma * np.sqrt(2*np.pi)) * exp

def exact_likelihood(
    preproc, 
    preproc_y, 
    model_train, 
    ztrue, 
    f, 
    fe, 
    Nintegral,
    cutoff,
    zgrid,
    ztrue_arg
):
    """Function to calculate the likelihood for the zero points.
    Calculates p(z|f) = sum_F p(F|f) * p(z|F) by drawing from p(F|f),
    calculating p(z|F) with MDN, and summing the importance sampled F points.
    
    Parameters
    ----------
    preproc : callable
        Rescaling function for magnitudes and colors.
        
    preproc_y : callable
        Rescaling function for redshift.
        
    model_train : callable
        MDN trained network.
    
    ztrue : ndarray of shape (n_galaxies, )
        Array with the true redshift.
        
    f : ndarray of shape (n_galaxies, n_features)
        Array with the input features.
        
    fe : ndarray of shape (n_galaxies, n_features)
        Array with the error of the input features.   
    
    Nintegral : int
        Number of samples to integrate the photometric noise. 
        
    cutoff : float
        Number of sigmas (roughly) at where to clip the likelihood of outlier galaxies. Default is 5.
        
    zgrid : ndarray of shape (n_grid).
        Redshift grid to perform the exact likelihood computation. 

    ztrue_arg : ndarray of shape (n_galaxies, ).
        Nearest positional argument of ztrue in zgrid.
        
    Returns
    -------
    loglike : ndarray of shape (n_galaxies, )
        The zero point loglikelihood value for each galaxy   
    """
    
    pz = np.zeros((Nintegral,len(f),len(zgrid)))
    
    for i in range(Nintegral):
        f_real = np.random.normal(f, fe)
        
        f_real = preproc.transform(f_real)
        y_pred = np.array(model_train(f_real))
        
        weight = y_pred[2]
        mu = y_pred[0]
        sig = np.sqrt(np.log(y_pred[1]))

        mu = preproc_y.inverse_transform(mu)
        sig = preproc_y.inverse_transform(sig)
        sig = np.clip(sig,0.001, np.inf)
        
        zipped = zip(mu.T, 
                 sig.T, 
                 weight.T)
        
        pzsub = np.sum([gaussian_vect(zgrid, m,s,w) for (m,s,w) in zipped], axis=0).T
        sum_pz = np.sum(pzsub,axis=1)[:,None]
        sum_pz[sum_pz==0] = 1.
        pzsub /= sum_pz
        pz[i] = pzsub
        
    pz = np.sum(pz,axis=0)
    sum_pz = np.sum(pz,axis=1)[:,None]
    sum_pz[sum_pz==0] = 1.
    pz /= sum_pz
    
    argmax = np.argmax(pz,axis=1)
    like_max = pz[range(len(argmax)), argmax]
    like = pz[range(len(ztrue)), ztrue_arg]
    
    loglike = -2.0*np.log(like) + 2.0 * np.log(like_max)
    loglike = np.clip(loglike, -np.inf, cutoff)
    loglike[np.isnan(loglike)] = cutoff
    return loglike
    
def approximate_likelihood(
    preproc, 
    preproc_y, 
    model_train, 
    ztrue, 
    f, 
    fe, 
    Nintegral,
    cutoff
):
    """Function to calculate an approximate likelihood for the zero points.
    Assumes p(z|f) = sum_F p(F|f) * p(z|F) to be a Gaussian distribution.
    F values are drawn from p(F|f), and the most likely z according to MDN=p(z|F) 
    is stored: {z}. The average and variance of {z} determines a Gaussian model for
    p(z|f).
    
    Parameters
    ----------
    preproc : callable
        Rescaling function for magnitudes and colors.
        
    preproc_y : callable
        Rescaling function for redshift.
        
    model_train : callable
        MDN trained network.
    
    ztrue : ndarray of shape (n_galaxies, )
        Array with the true redshift.
        
    f : ndarray of shape (n_galaxies, n_features)
        Array with the input features.
        
    fe : ndarray of shape (n_galaxies, n_features)
        Array with the error of the input features.   
    
    Nintegral : int
        Number of samples to integrate the photometric noise. 
        
    cutoff : float
        Number of sigmas (roughly) at where to clip the likelihood of outlier galaxies. Default is 5.
        
    Returns
    -------
    loglike : ndarray of shape (n_galaxies, )
        The zero point loglikelihood value for each galaxy
    """

    photoz = np.zeros((Nintegral,len(f)))
    for i in range(Nintegral):
        f_real = np.random.normal(f, fe)

        ## Prediction 
        f_real = preproc.transform(f_real)
        y_pred = np.array(model_train(f_real))

        y_pred_arg = np.argmax(y_pred[2, :, :], axis = 1)
        y_pred_mean = y_pred[0, :, :][:, y_pred_arg][:, 0]
        #y_pred_arg = np.argmax(y_pred[2], axis = 1)
        #y_pred_mean = y_pred[0][range(len(y_pred_arg)), y_pred_arg]

        photoz[i] = preproc_y.inverse_transform(y_pred_mean.reshape(-1, 1))[:, 0]
        
    mean_z = np.mean(photoz,axis=0)
    std_z = np.std(photoz,axis=0, ddof=1)
    
    like = norm.pdf(ztrue, loc=mean_z, scale=std_z)
    like = np.clip(like,1e-200,np.inf)
    like_max = norm.pdf(mean_z, loc=mean_z, scale=std_z)
    loglike = -2.0*np.log(like) + 2.0 * np.log(like_max)
    loglike = np.clip(loglike, -np.inf, cutoff)
    loglike[np.isnan(loglike)] = cutoff
    return loglike

def _likelihood_wrapper(
    preproc, 
    preproc_y, 
    model_train, 
    ztrue, 
    f, 
    fe, 
    Nintegral,
    method,
    cutoff,
    zgrid, 
    ztrue_arg
):
    """Function wrapper to call the appropriate likelihood function to integrate 
    the photometric noise.
 
    Parameters
    ----------
    preproc : callable
        Rescaling function for magnitudes and colors.
        
    preproc_y : callable
        Rescaling function for redshift.
        
    model_train : callable
        MDN trained network.
    
    ztrue : ndarray of shape (n_galaxies, )
        Array with the true redshift.
        
    f : ndarray of shape (n_galaxies, n_features)
        Array with the input features.
        
    fe : ndarray of shape (n_galaxies, n_features)
        Array with the error of the input features.   
    
    Nintegral : int
        Number of samples to integrate the photometric noise.
    
    method : str
        Method to integrate the photometric noise. Options are 'exact' or 'approximate'.
        
    cutoff : float
        Number of sigmas (approx) at where to clip the likelihood of outlier galaxies. Default is 5.
        
    zgrid : ndarray of shape (n_grid).
        Redshift grid to perform the exact likelihood computation. 

    ztrue_arg : ndarray of shape (n_galaxies, ).
        Nearest positional argument of ztrue in zgrid.
        
    Returns
    -------
    loglike : ndarray of shape (n_galaxies, )
        The zero point loglikelihood value for each galaxy
    """
    
    if (method == "exact"):
        return exact_likelihood(preproc, preproc_y, model_train, ztrue, f, fe, Nintegral, 
                                cutoff, zgrid, ztrue_arg)
        
    elif (method == "approximate"):
        return approximate_likelihood(preproc, preproc_y, model_train, ztrue, f, fe, Nintegral, cutoff)

def run_zp_chain(
    preproc,
    preproc_y,
    model_train,
    ztrue, 
    f, 
    fe, 
    Nintegral,
    Nchain,
    method = "exact",
    params_init = None,
    cutoff = 5., 
    extra_cov = 9.,
    step_size = 0.005,
    zgrid = None,
    D = 5
):
    """Function that runs an MCMC over the zero-point space. Runs a Metropolis-Hastings walk 
    with a Gaussian as the proposal distribution. With zeropoints aka 'zp' and some data:
    -- p(zp|data) propto p(data|zp) * p(zp)
    We assume p(zp) to be uniform for now. Data is the true redshifts and features: {ztrue,f}.
    -- p(data|zp) = p(ztrue,f|zp) = p(ztrue|f, zp) * p(f|zp) = p(ztrue|f+zp)
    Where p(f|zp) is simply a delta centered at f+zp. And as usual:
    -- p(ztrue|f+zp) = sum_F p(F|f+zp) * p(z|F)
 
    Parameters
    ----------
    preproc : callable
        Rescaling function for magnitudes and colors.
        
    preproc_y : callable
        Rescaling function for redshift.
        
    model_train : callable
        MDN trained network.
    
    ztrue : ndarray of shape (n_galaxies, )
        Array with the true redshift.
        
    f : ndarray of shape (n_galaxies, n_features)
        Array with the input features.
        
    fe : ndarray of shape (n_galaxies, n_features)
        Array with the error of the input features.   
    
    Nintegral : int
        Number of samples to integrate the photometric noise.
    
    Nchain : int
        Number of steps in the zero-point chain.
        
    method : str
        Method to integrate the photometric noise. Options are 'exact' or 'approximate'.
        
    params_init : ndarray of shape (n_params, )
        Initial guess at the parameters. Default is zero.
        
    cutoff : float
        Number of sigmas (approx) at where to clip the likelihood of outlier galaxies. Default is 5.
        
    extra_cov : float
        Factor to increase the nominal variance of the zero-point likelihood. Default is 3**2.
        
    step_size : float
        Width size of the Gaussian proposal for the MCMC. Default is 0.005
        
    zgrid : ndarray of shape (n_grid). Optional.
        Redshift grid to perform the exact likelihood computation.
        
    Returns
    -------
    chain : ndarray of shape (n_iterations, n_params)
        The values of the zero point params at each step
    P : ndarray of shape (n_iterations, )
        The value of the zero point likelihood at each step
    """
    
    if (method != "exact") & (method != "approximate"):
        raise NotImplementedError("Method has to be either 'exact' or 'approximate'.")
        
    if method == "exact":
        ztrue_arg = np.argmin(np.add.outer(ztrue,-zgrid)**2,axis=1)
    else:
        ztrue_arg = None
    
    n_params = f.shape[1] - 1
    print(f.shape)
    if params_init is None:
        params_init = np.zeros(n_params)
    
    chain = np.zeros((Nchain, n_params))
    P = np.zeros(Nchain)

    # Initial step
    params = params_init.copy()
    xdata = f.copy()
    xdata[:,:D-1] = xdata[:,:D-1] + params
    t0 = time.time()
    loglike = _likelihood_wrapper(preproc, preproc_y, model_train, ztrue, xdata, fe, Nintegral, 
                                 method, cutoff, zgrid, ztrue_arg)
    t1 = time.time()
    chain[0] = params
    P[0] = loglike.sum()
    print(t1-t0)
    print(0, np.round(chain[0],4), np.round(P[0],2))

    # Simple Metropolis-Hastings
    for i in range(Nchain-1):
        params = chain[i] + np.random.normal(0, step_size, n_params)

        xdata = f.copy()
        xdata[:,:D-1] = xdata[:,:D-1] + params
        t0 = time.time()
        loglike = _likelihood_wrapper(preproc, preproc_y, model_train, ztrue, xdata, fe, Nintegral, 
                                     method, cutoff, zgrid, ztrue_arg)
        t1 = time.time()
        
        # extra_cov reduces the significance of the calculated probability ratio.
        # Makes it easy to jump around.
        # Useful when the likelihood is not accurate.
        A = min(1, np.exp(-0.5*(loglike.sum()-P[i])/extra_cov))
        R = np.random.rand(1)[0]

        if R <= A:
            chain[i+1] = params
            P[i+1] = loglike.sum()
        else:
            chain[i+1] = chain[i]
            P[i+1] = P[i]

        print(t1-t0)
        print(i, np.round(chain[i+1],4), np.round(P[i+1],2))
    

    return chain, P
