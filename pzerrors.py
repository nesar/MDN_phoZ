import time
import numpy as np
from scipy.stats import qmc, norm

def f2m(f, fe):
    m = -2.5 * np.log10(f)
    me = (2.5 / np.log(10)) * (fe / abs(f)) # Taylor first order expansion.
    return m, me

def m2f(m, me):
    f = 10 ** (-0.4 * m)
    fe = f * me * (np.log(10) / 2.5) # Taylor first order expansion.
    return f, fe

def mag2flux(m, me):
    i = m[:, 4]
    z = i - m[:, 3] # z = i - (i-z)
    r = m[:, 2] + i # r = (r-i) + i
    g = m[:, 1] + r # g = (g-r) + r
    u = m[:, 0] + g # u = (u-g) + g
    
    ie = me[:, 4]
    ze = np.sqrt(me[:, 3]**2 - ie**2) # Var(z) = Var(i-z) - Var(i)
    re = np.sqrt(me[:, 2]**2 - ie**2) # Var(r) = Var(r-i) - Var(i)
    ge = np.sqrt(me[:, 1]**2 - re**2) # Var(g) = Var(g-r) - Var(r)
    ue = np.sqrt(me[:, 0]**2 - ge**2) # Var(u) = Var(u-g) - Var(g)
    
    mags = np.array([u, g, r, i, z]).T
    mags_err = np.array([ue, ge, re, ie, ze]).T
    fluxes, fluxes_err = m2f(mags, mags_err)
    
    return fluxes, fluxes_err

def flux2mag(fluxes, fluxes_err):
    
    mags, mags_err = f2m(fluxes, fluxes_err)
    u, g, r, i, z = mags.T
    ue, ge, re, ie, ze = mags_err.T
    
    m = np.zeros_like(fluxes)
    me = np.zeros_like(fluxes)
    
    m[:, 0] = u - g
    m[:, 1] = g - r
    m[:, 2] = r - i
    m[:, 3] = i - z
    m[:, 4] = i
    
    me[:, 0] = np.sqrt(ue**2 + ge**2)
    me[:, 1] = np.sqrt(ge**2 + re**2)
    me[:, 2] = np.sqrt(re**2 + ie**2)
    me[:, 3] = np.sqrt(ie**2 + ze**2)
    me[:, 4] = ie
    
    return m, me


def gaussian_vect(zgrid, mu, sigma, weight):
    exp = np.exp(-0.5 * np.add.outer(zgrid, -mu)**2/sigma**2)
    return weight / (sigma * np.sqrt(2*np.pi)) * exp

def pz_error_batch_magnitude(
    preproc, 
    preproc_y, 
    model_train, 
    feature, 
    feature_error, 
    Nintegral,
    zgrid,
    zeropoint=None,
):
    """Function to calculate the p(z) of a galaxy marginalizing over the feature errors.
    Features are assumed to be: (u-g, g-r, r-i, i-z, i) magnitudes.
    The errors are sampled from 5 independent Gaussians in (u-g, g-r, r-i, i-z, i) space.
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
        
    feature : ndarray of shape (n_galaxies, n_features)
        Array with the input features.
        
    feature_error : ndarray of shape (n_galaxies, n_features)
        Array with the error of the input features.   
    
    Nintegral : int
        Number of samples to integrate the photometric noise. 
        
    zgrid : ndarray of shape (n_grid).
        Redshift grid to perform the exact likelihood computation. 
        
    zeropoint : (n_features, )
        Zero point shifts that correct the leading order bias between models and data. 
        
    Returns
    -------
    loglike : ndarray of shape (n_galaxies, )
        The zero point loglikelihood value for each galaxy   
    """
    pz = np.zeros((Nintegral, len(feature), len(zgrid)))
    
    if zeropoint is None:
        zeropoint = np.zeros(feature.shape[1])
    feature_zp = feature + zeropoint
    
    for i in range(Nintegral):
        print("%d out of %d"%(i, Nintegral))
        f_real = np.random.normal(feature_zp, feature_error)
        
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
    
    return pz


def pz_error_batch_flux(
    preproc, 
    preproc_y, 
    model_train, 
    feature, 
    feature_error, 
    Nintegral,
    zgrid,
    zeropoint=None,
):
    """Function to calculate the p(z) of a galaxy marginalizing over the feature errors.
    Features are assumed to be: (u-g, g-r, r-i, i-z, i) magnitudes.
    The errors are sampled from 5 independent Gaussians in (u, g, r, i, z) space 
    and then converted to (u-g, g-r, r-i, i-z, i) space.
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
        
    feature : ndarray of shape (n_galaxies, n_features)
        Array with the input features.
        
    feature_error : ndarray of shape (n_galaxies, n_features)
        Array with the error of the input features.    
    
    Nintegral : int
        Number of samples to integrate the photometric noise. 
        
    zgrid : ndarray of shape (n_grid).
        Redshift grid to perform the exact likelihood computation. 
        
    zeropoint : (n_features, )
        Zero point shifts that correct the leading order bias between models and data. 
        
    Returns
    -------
    loglike : ndarray of shape (n_galaxies, )
        The zero point loglikelihood value for each galaxy   
    """
    
    pz = np.zeros((Nintegral, len(feature), len(zgrid)))
    
    if zeropoint is None:
        zeropoint = np.zeros(feature.shape[1])
    feature_zp = feature + zeropoint
    
    fluxes, fluxes_err = mag2flux(feature_zp, feature_error)
    
    for i in range(Nintegral):
        print("%d out of %d"%(i, Nintegral))
        fluxes_real = np.random.normal(fluxes, fluxes_err)
        
        fluxes_real = np.where(
            fluxes_real > 0.0, 
            fluxes_real, 
            fluxes * 10**(-0.4 * 1)
        )
        f_real = flux2mag(fluxes_real, fluxes_err)[0]
        
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
    
    return pz


def get_normal_qmc_samples(means, sigmas, Nsamples):
    ngal, Ndim = means.shape
    sampler = qmc.LatinHypercube(d=Ndim)
    sample = sampler.random(n=Nsamples)
    
    sample_final = np.zeros((Nsamples, ngal, Ndim))
    for ng in range(ngal):
        for i in range(Ndim):
            sample_final[:, ng, i] = norm.ppf(sample[:,i], loc=means[ng, i], scale=sigmas[ng, i])
    return sample_final


def pz_error_batch_flux_QMC(
    preproc, 
    preproc_y, 
    model_train, 
    feature, 
    feature_error, 
    Nintegral,
    zgrid,
    zeropoint=None,
):
    """Function to calculate the p(z) of a galaxy marginalizing over the feature errors.
    Features are assumed to be: (u-g, g-r, r-i, i-z, i) magnitudes.
    The errors are sampled from 5 independent Gaussians in (u, g, r, i, z) space 
    and then converted to (u-g, g-r, r-i, i-z, i) space.
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
        
    feature : ndarray of shape (n_galaxies, n_features)
        Array with the input features.
        
    feature_error : ndarray of shape (n_galaxies, n_features)
        Array with the error of the input features.    
    
    Nintegral : int
        Number of samples to integrate the photometric noise. 
        
    zgrid : ndarray of shape (n_grid).
        Redshift grid to perform the exact likelihood computation. 
        
    zeropoint : (n_features, )
        Zero point shifts that correct the leading order bias between models and data. 
        
    Returns
    -------
    loglike : ndarray of shape (n_galaxies, )
        The zero point loglikelihood value for each galaxy   
    """
    
    pz = np.zeros((Nintegral, len(feature), len(zgrid)))
    
    if zeropoint is None:
        zeropoint = np.zeros(feature.shape[1])
    feature_zp = feature + zeropoint
    
    fluxes, fluxes_err = mag2flux(feature_zp, feature_error)
    
    samples = get_normal_qmc_samples(fluxes, fluxes_err, Nintegral)
    
    for i in range(Nintegral):
        print("%d out of %d"%(i, Nintegral))
        fluxes_real = samples[i]
        fluxes_real = np.where(
            fluxes_real > 0.0, 
            fluxes_real, 
            fluxes * 10**(-0.4 * 1)
        )
        
        f_real = flux2mag(fluxes_real, fluxes_err)[0]
        
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
    
    return pz