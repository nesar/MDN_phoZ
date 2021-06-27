import numpy as np
import copy
import matplotlib.pylab as plt
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
import SetPub
from scipy import stats
SetPub.set_pub()
np.random.seed(12211) 

##############################
# Perturb bands (one or all) #
##############################

def perturb(X_test, band_n = None, delta_c = 1.1, approach = 'a', X_err = None): # X_err needed for approaches 'c' and 'd'
    
    prtb_X_test = copy.deepcopy(X_test)

    np.random.seed(0)
    if approach == 'a':    # percent change
        if type(delta_c) == list: # this way we can handle multiple delta_c's
            prtb_X_test_lst = []
            for this_delta_c in delta_c:
                prtb_X_test = copy.deepcopy(X_test)
                prtb_X_test[:,band_n] = prtb_X_test[:,band_n]*this_delta_c
                prtb_X_test_lst.append(prtb_X_test)
            return prtb_X_test_lst
        prtb_X_test[:,band_n] = prtb_X_test[:,band_n]*delta_c
    elif approach == 'b': # random value
        np.random.shuffle(prtb_X_test[:, band_n])
    elif approach == 'c': # draw from Gaussian (indiv band)
        if X_err is None:
            print("Please include an X_err")
            return 0
        for i in range(len(X_test)): # for each galaxy
            mu, sigma = X_test[i,band_n], X_err[i,band_n] # mean and standard deviation
            prtb_X_test[i,band_n] = np.random.normal(mu, sigma, 1)
        #print("mu:\n", mu, "\nmu shape: ", mu.shape, "\nsigma:\n", sigma, "\nsigma shape: ", sigma.shape)
    elif approach == 'd': # draw from Gaussian (all bands)
        if X_err is None:
            print("Please include an X_err")
            return 0
        mu = X_test
        for i in range(len(X_test)): # for each galaxy
            cov = np.identity(len(X_err[0])) # Second dimension is num of colors (better way?)
            for j in range(len(X_test[0])): # for each color
                cov[j,j] = X_err[i][j]
            prtb_X_test[i] = np.random.multivariate_normal(mu[i], cov, 1)
    elif approach == 'e':
        prtb_X_test = np.delete(prtb_X_test, band_n, axis = 1)
        
    # Quantify the results (i.e. mean squared error) -- Nesar will send some things
    return prtb_X_test

#####################
# Predict photo-z's #
#####################

def prediction_zp(xdata, xzero, model_train, preproc, preproc_y, galID = None):
    f_real = xdata.copy()
    f_real[:,:4] = f_real[:,:4] + xzero # So fifth band (magnitude) doesn't get changed, but it should still be there... # This is not vectorized addition
    if galID is None:
        f_real = preproc.transform(f_real)
        y_pred = np.array(model_train(f_real)) # y_pred is straight up the outputs of training the model on this data
    else:
        f_real = preproc.transform(f_real[galID][None, :])
        y_pred = np.array(model_train(f_real))
    y_pred_arg = np.argmax(y_pred[2, :, :], axis = 1)
    y_pred_mean = y_pred[0, :, :][:, y_pred_arg][:, 0]
    y_pred_std = np.sqrt(np.log(y_pred[1, :, :][:, y_pred_arg][:, 0]))
    photoz = preproc_y.inverse_transform(y_pred_mean.reshape(-1, 1))[:, 0] # this is the part I don't get... is this finding photo_z's that would lead to the same predictions?
    photoz_err = preproc_y.inverse_transform(y_pred_std.reshape(-1, 1) )[:, 0]
    return photoz, photoz_err#, y_pred # mean, stddev

def prediction_zp_all(X_test, label_test, save_mod, surveystring, model_train, preproc, preproc_y): # combine the surveys, then predict the squished total
    xzero = np.zeros_like(X_test[:,:4]) # everything except the magnitude bin
    xzero[label_test == 0] = np.loadtxt(save_mod + '_xzero_' + surveystring[0])
    xzero[label_test == 1] = np.loadtxt(save_mod + '_xzero_' + surveystring[1])
    xzero[label_test == 2] = np.loadtxt(save_mod + '_xzero_' + surveystring[2])
    photoz, photoz_err = prediction_zp(X_test, xzero, model_train, preproc, preproc_y)
    return photoz, photoz_err, xzero

def Nesar_prediction_zp_all(X_test, label_test, save_mod, surveystring, model_train, preproc, preproc_y):
    xzero = np.zeros_like(X_test[:,:4])
    xzero[label_test == 0] = np.loadtxt(save_mod + '_xzero_' + surveystring[0])
    xzero[label_test == 1] = np.loadtxt(save_mod + '_xzero_' + surveystring[1])
    xzero[label_test == 2] = np.loadtxt(save_mod + '_xzero_' + surveystring[2])  
    f_real = X_test.copy()
    f_real[:,:4] = f_real[:,:4] + xzero
    f_real = preproc.transform(f_real)
    y_pred = np.array(model_train(f_real))
    y_pred_3means = preproc_y.inverse_transform(y_pred[0, :, :])
    y_pred_3std = preproc_y.inverse_transform( np.sqrt(np.log(y_pred[1, :, :])  ))
    y_pred_3weights = y_pred[2, :, :]
    predstdweights = np.array([y_pred_3means, y_pred_3std, y_pred_3weights])
    return y_pred_3means, y_pred_3std, y_pred_3weights, xzero

def old_predict(X_test, preproc, model_train):
    X_test = preproc.transform(X_test)
    y_pred = np.array(model_train(X_test))
    y_pred_arg = np.argmax(y_pred[2, :, :], axis = 1)
    y_pred_mean = y_pred[0, :, :][:, y_pred_arg][:, 0]
    y_pred_std = np.sqrt(np.log(y_pred[1, :, :][:, y_pred_arg][:, 0]))
    return y_pred_mean, y_pred_std

##############################################
# Perturb all bands, using a single approach #
##############################################

def squish_perturb_mult(X_test, X_err, label_test, save_mod, surveystring, model_train, preproc, preproc_y, approach = 'a', delta_c = 1.1): # quickly perturb multiple bins
    
    # New version
    og_y_pred_mean, og_y_pred_std  = prediction_zp_all(X_test, label_test, save_mod, surveystring, model_train, preproc, preproc_y)

    prtb_X_test_lst = []
    prtb_y_pred_mean_lst = []
    prtb_y_pred_std_lst = []
    for band_n in [0, 1, 2, 3, 4]: # all the way through 4?
        prtb_X_test = perturb(X_test, band_n, delta_c = delta_c, approach = approach, X_err = X_err)
        prtb_X_test_lst.append(prtb_X_test)
        if approach == 'a' and type(delta_c) == list:
            prtb_y_pred_mean = []
            prtb_y_pred_std = []
            for this_prtb_X_test in prtb_X_test:
                this_prtb_y_pred_mean, this_prtb_y_pred_std  = prediction_zp_all(prtb_X_test, label_test, save_mod, surveystring, model_train, preproc, preproc_y)
                prtb_y_pred_mean.append(this_prtb_y_pred_mean)
                prtb_y_pred_std.append(this_prtb_y_pred_std)
        else:
            prtb_y_pred_mean, prtb_y_pred_std = prediction_zp_all(prtb_X_test, label_test, save_mod, surveystring, model_train, preproc, preproc_y)
        prtb_y_pred_mean_lst.append(prtb_y_pred_mean)
        prtb_y_pred_std_lst.append(prtb_y_pred_std)
    
    return prtb_X_test_lst, prtb_y_pred_mean_lst, prtb_y_pred_std, og_y_pred_mean, og_y_pred_std

def perturb_mult(X_test, X_err, sel_ind, sel, xzero_correction, model_train, preproc, preproc_y, approach = 'a', delta_c = 1.1): # quickly perturb multiple bins
    
    # New version
    og_y_pred_mean, og_y_pred_std  = prediction_zp(X_test[sel], xzero_correction,  model_train, preproc, preproc_y)

    prtb_X_test_lst = []
    prtb_y_pred_mean_lst = []
    prtb_y_pred_std_lst = []
    for band_n in [0, 1, 2, 3, 4]:
        prtb_X_test = perturb(X_test, band_n, delta_c = delta_c, approach = approach, X_err = X_err)
        prtb_X_test_lst.append(prtb_X_test)
        if approach == 'a' and type(delta_c) == list:
            prtb_y_pred_mean = [] 
            prtb_y_pred_std = []
            for this_prtb_X_test in prtb_X_test:
                this_prtb_y_pred_mean, this_prtb_y_pred_std  = prediction_zp(this_prtb_X_test[sel], xzero_correction, model_train, preproc, preproc_y)
                prtb_y_pred_mean.append(this_prtb_y_pred_mean)
                prtb_y_pred_std.append(this_prtb_y_pred_std)
        else:
            prtb_y_pred_mean, prtb_y_pred_std = prediction_zp(prtb_X_test[sel], xzero_correction, model_train, preproc, preproc_y)
        prtb_y_pred_mean_lst.append(prtb_y_pred_mean)
        prtb_y_pred_std_lst.append(prtb_y_pred_std)
    
    return prtb_X_test_lst, prtb_y_pred_mean_lst, prtb_y_pred_std, og_y_pred_mean, og_y_pred_std


def old_perturb_mult(X_test, X_err, preproc, model_train, approach = 'a', delta_c = 1.1): # quickly perturb multiple bins

    og_y_pred_mean, og_y_pred_std = predict(X_test, preproc, model_train)
    
    prtb_X_test_lst = []
    prtb_y_pred_mean_lst = []
    prtb_y_pred_std_lst = []
    for band_n in [0, 1, 2, 3, 4]:
        prtb_X_test = perturb(X_test, band_n, delta_c = delta_c, approach = approach, X_err = X_err)
        prtb_X_test_lst.append(prtb_X_test)
        if approach == 'a' and type(delta_c) == list:
            prtb_y_pred_mean = [] 
            prtb_y_pred_std = []
            for this_prtb_X_test in prtb_X_test:
                this_prtb_y_pred_mean, this_prtb_y_pred_std = predict(this_prtb_X_test, preproc, model_train)
                prtb_y_pred_mean.append(this_prtb_y_pred_mean)
                prtb_y_pred_std.append(this_prtb_y_pred_std)
        else:
            prtb_y_pred_mean, prtb_y_pred_std = predict(prtb_X_test, preproc, model_train)
        prtb_y_pred_mean_lst.append(prtb_y_pred_mean)
        prtb_y_pred_std_lst.append(prtb_y_pred_std)
    
    return prtb_X_test_lst, prtb_y_pred_mean_lst, prtb_y_pred_std, og_y_pred_mean, og_y_pred_std

##################################
# Validate with sigma or outFrac #
##################################

def sigmaNMAD(z_spec, z_pho):
    return 1.48*np.median( np.abs( z_pho - z_spec)/(1 + z_spec))
    # else: return 1.48*np.median( np.abs( z_pho - z_spec)/(1 + z_spec),)
def outlierFrac(z_spec, z_pho, threshold = 0.15):
    outliers = z_pho[ (np.abs(z_spec - z_pho)) >= threshold*z_spec ]
    return 100.0*len(outliers)/np.shape(z_pho)[0]

def validate(z_spec, z_pho): # choose whether to offer a perturbed or original dataset

    bins = np.linspace(0, 1.5, 15) # what's with these bins? And why are we assuming z_spec and z_pho have multiple dimensions?
    bincenter = (bins[1:] + bins[:-1]) / 2.
    z_spec_digitize = np.digitize(z_spec, bins) # Return the indices of the bins to which each value in input array belongs. 
    sigmaNMAD_array = np.zeros(shape=bins.shape[0])
    outFr_array = np.zeros(shape=bins.shape[0])
    
    for ind in range(bins.shape[0] - 1): # index: where do we fit in the digitized space?
        z_spec_bin_z =  z_spec[ z_spec_digitize  == ind + 1] # Why is it + 1?
        z_pho_bin_z =  z_pho[ z_spec_digitize  == ind + 1]
        if len(z_spec_digitize[z_spec_digitize == ind + 1]) > 0:
            sigmaNMAD_array[ind] =  sigmaNMAD(z_spec_bin_z, z_pho_bin_z)
            outFr_array[ind] = outlierFrac(z_spec_bin_z, z_pho_bin_z, 0.15)
        else:
            print("length zero for bin ", ind + 1)
        
    return sigmaNMAD_array, outFr_array, bins

def squish_validate(y_test, y_pred_mean): # choose whether to offer a perturbed or original dataset
    z_spec = y_test
    z_pho = y_pred_mean
    
    bins = np.linspace(0, 1, 20)
    bincenter = (bins[1:] + bins[:-1]) / 2.
    z_spec_digitize = np.digitize(z_spec, bins) # Return the indices of the bins to which each value in input array belongs. 
    sigmaNMAD_array = np.zeros(shape=bins.shape[0])
    outFr_array = np.zeros(shape=bins.shape[0])
    
    for ind in range(bins.shape[0] - 1): # index?
        z_spec_bin_z =  z_spec[ z_spec_digitize  == ind + 1]
        z_pho_bin_z =  z_pho[ z_spec_digitize  == ind + 1]
        sigmaNMAD_array[ind] =  sigmaNMAD(z_spec_bin_z, z_pho_bin_z)
        outFr_array[ind] = outlierFrac(z_spec_bin_z, z_pho_bin_z, 0.15)
        
    return sigmaNMAD_array, outFr_array, bins

def validate_sel(y_test, sel, new_y_pred_mean): # choose whether to offer a perturbed or original dataset
    z_spec = y_test[sel]
    z_pho = new_y_pred_mean
    
    bins = np.linspace(0, 1, 20)
    bincenter = (bins[1:] + bins[:-1]) / 2.
    z_spec_digitize = np.digitize(z_spec, bins) # Return the indices of the bins to which each value in input array belongs. 
    sigmaNMAD_array = np.zeros(shape=bins.shape[0])
    outFr_array = np.zeros(shape=bins.shape[0])
    
    for ind in range(bins.shape[0] - 1): # index?
        z_spec_bin_z =  z_spec[ z_spec_digitize  == ind + 1]
        z_pho_bin_z =  z_pho[ z_spec_digitize  == ind + 1]
        sigmaNMAD_array[ind] =  sigmaNMAD(z_spec_bin_z, z_pho_bin_z)
        outFr_array[ind] = outlierFrac(z_spec_bin_z, z_pho_bin_z, 0.15)
        
    return sigmaNMAD_array, outFr_array, bins

def old_validate(y_test, y_pred_mean, preproc_y): # choose whether to offer a perturbed or original dataset
    print(type(y_pred_mean))
    print(y_pred_mean)
    z_spec = y_test
    z_pho = preproc_y.inverse_transform(y_pred_mean.reshape(-1, 1))[:, 0] # prtb or og
    
    bins = np.linspace(0, 1, 20)
    bincenter = (bins[1:] + bins[:-1]) / 2.
    z_spec_digitize = np.digitize(z_spec, bins) # Return the indices of the bins to which each value in input array belongs. 
    sigmaNMAD_array = np.zeros(shape=bins.shape[0])
    outFr_array = np.zeros(shape=bins.shape[0])
    
    for ind in range(bins.shape[0] - 1): # index?
        z_spec_bin_z =  z_spec[ z_spec_digitize  == ind + 1]
        z_pho_bin_z =  z_pho[ z_spec_digitize  == ind + 1]
        sigmaNMAD_array[ind] =  sigmaNMAD(z_spec_bin_z, z_pho_bin_z)
        outFr_array[ind] = outlierFrac(z_spec_bin_z, z_pho_bin_z, 0.15)
        
    return sigmaNMAD_array, outFr_array, bins

#############
# Plot PDFs #
#############

def plot_normal_mix(pis, mus, sigmas, ax, label='', color = '', comp=True):
    """Plots the mixture of Normal models to axis=ax comp=True plots all
    components of mixture model
    """
    x = np.linspace(-0.1, 1.1, 250)
    final = np.zeros_like(x)
    for i, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
        temp = stats.norm.pdf(x, mu_mix, sigma_mix) * weight_mix
        final = final + temp
        if comp:
            ax.plot(x, temp, 'k--', alpha =0.9)
        ax.plot(x, final,label=label, color = color)
        ax.legend(fontsize=13)
    return final

def plot_pdfs(pred_means,pred_weights,pred_std, y, num_train = 200000, num_test = 20000, num=4, label = '', color = '', train=False, comp = False):
    np.random.seed(12)
    nrows = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex = True, figsize=(8, nrows*3), num='pdfs')
    if train:
        obj = np.random.randint(0,num_train-1,num)
    else:
        obj = np.random.randint(0,num_test-1,num)
    obj = [42, 30, 80]
    allfs = []
    for i in range(len(obj)):
        print(i)
        if (i==0):
            fs = plot_normal_mix(pred_weights[obj][i], pred_means[obj][i], pred_std[obj][i], axes[i], label = label, color = color, comp=comp)
        else: fs = plot_normal_mix(pred_weights[obj][i], pred_means[obj][i], pred_std[obj][i], axes[i], label = '', color = color, comp=comp)
        axes[i].set_ylabel(r'${\rm PDF}$', fontsize = 22)
        allfs.append(fs)
        axes[i].axvline(x=y[obj][i], color='black', alpha=0.5)
    plt.xlabel(r'$z_{phot}$', fontsize = 26)
    return fig, axes

######################
# Plot side-by-sides #
######################

def old_plot_side_by_sides(y_test, preproc_y, y_pred_means, y_pred_stds, labels, approach = 'a', band_n = 0, delta_c = 1.1): # Assume the last two inputs are lists of equal length
    
    #plt.figure(figsize=(10, 10))
    fig, axs = plt.subplots(1,len(y_pred_means), figsize=(10,7.5))
    for i, ax in enumerate(axs):
        ax.errorbar(y_test, preproc_y.inverse_transform(y_pred_means[i].reshape(-1, 1))[:, 0], # Inverse transform KNOWS what you did to get preproc_y :P (this is the line where you need y_pred_means)
                 yerr= preproc_y.inverse_transform(y_pred_stds[i].reshape(-1, 1) )[:, 0], 
                 fmt='ro', ecolor='k', ms = 5, alpha = 0.1, label = labels[i])
        C = 0.05
        z_t = np.array([0, 1])
        z_tp = z_t + C*(1+z_t) # photo z's (predictions)
        z_tm = z_t - C*(1+z_t) # spectroscopic z's (trues)

        ax.plot(z_t, z_t, 'k')
        ax.plot(z_t, z_tp, 'k-.')
        ax.plot(z_t, z_tm, 'k-.')
        ax.set_ylabel(r'$z_{phot}$', fontsize=25)
        ax.set_xlabel(r'$z_{spec}$', fontsize=25)
        #plt.axes().set_aspect('equal')
        leg = ax.legend(fontsize = 'xx-large', markerscale=1., numpoints=2)
    plt.tight_layout()
    
    fig.subplots_adjust(top=0.94)
    if approach == 'a':
        fig.suptitle("Band " + str(band_n) + " perturbed with " + str(np.around((delta_c - 1)*100)) + " percent increase", fontsize = 16)
    elif approach == 'b':
        fig.suptitle("Perturbed with random numbers", fontsize = 16)
    elif approach == 'c':
        fig.suptitle("Perturbed with gaussian sampling (band " + str(band_n) + ")", fontsize = 16)
    elif approach == 'd':
        fig.suptitle("Perturbed with gaussian sampling (all bands)", fontsize = 16)
    return fig, axs

def squish_plot_side_by_sides(y_test, y_pred_means, y_pred_stds, labels, approach = 'a', band_n = 0, delta_c = 1.1): # Assume the last two inputs are lists of equal length
    
    #plt.figure(figsize=(10, 10))
    ylims = np.zeros((2,2))
    fig, axs = plt.subplots(1,len(y_pred_means), figsize=(10,7.5))
    for i, ax in enumerate(axs):
        ax.errorbar(y_test, y_pred_means[i], yerr= y_pred_stds[i], fmt='ro', ecolor='k', ms = 5, alpha = 0.1, label = labels[i])
        C = 0.05
        z_t = np.array([0, 1])
        z_tp = z_t + C*(1+z_t) # photo z's (predictions)
        z_tm = z_t - C*(1+z_t) # spectroscopic z's (trues)

        ax.plot(z_t, z_t, 'k')
        ax.plot(z_t, z_tp, 'k-.')
        ax.plot(z_t, z_tm, 'k-.')
        ax.set_ylabel(r'$z_{phot}$', fontsize=25)
        ax.set_xlabel(r'$z_{spec}$', fontsize=25)
        #plt.axes().set_aspect('equal')
        leg = ax.legend(fontsize = 'xx-large', markerscale=1., numpoints=2)
        ylims[i] = ax.get_ylim()
    plt.tight_layout()
    
    # Adjust y axes
    ytop = np.min(ylims[:,0])
    ybottom = np.max(ylims[:,1])
    axs[0].set_ylim(ytop, ybottom)
    axs[1].set_ylim(ytop, ybottom)
    
    fig.subplots_adjust(top=0.94)
    band_names = ["u-g", "g-r", "r-i", "i-z", "mag(i)"]
    if approach == 'a':
        fig.suptitle("Band " + band_names[band_n] + " perturbed with " + str(np.around((delta_c - 1)*100)) + " percent increase", fontsize = 16)
    elif approach == 'b':
        fig.suptitle("Perturbed with random numbers (band " + band_names[band_n] + ")", fontsize = 16)
    elif approach == 'c':
        fig.suptitle("Perturbed with gaussian sampling (band " + band_names[band_n] + ")", fontsize = 16)
    elif approach == 'd':
        fig.suptitle("Perturbed with gaussian sampling (all bands)", fontsize = 16)
    return fig, axs

def plot_side_by_sides(y_test, sel, sel_ind, new_y_pred_means, new_y_pred_stds, labels, approach = 'a', band_n = 0, delta_c = 1.1): # Assume the last two inputs are lists of equal length
    
    #plt.figure(figsize=(10, 10))
    ylims = np.zeros((2,2))
    fig, axs = plt.subplots(1,len(new_y_pred_means), figsize=(10,7.5))
    for i, ax in enumerate(axs):
        ax.errorbar(y_test[sel], new_y_pred_means[i], yerr= new_y_pred_stds[i], fmt='ro', ecolor='k', ms = 5, alpha = 0.1, label = labels[i])
        C = 0.05
        z_t = np.array([0, 1])
        z_tp = z_t + C*(1+z_t) # photo z's (predictions)
        z_tm = z_t - C*(1+z_t) # spectroscopic z's (trues)

        ax.plot(z_t, z_t, 'k')
        ax.plot(z_t, z_tp, 'k-.')
        ax.plot(z_t, z_tm, 'k-.')
        ax.set_ylabel(r'$z_{phot}$', fontsize=25)
        ax.set_xlabel(r'$z_{spec}$', fontsize=25)
        #plt.axes().set_aspect('equal')
        leg = ax.legend(fontsize = 'xx-large', markerscale=1., numpoints=2)
        ylims[i] = ax.get_ylim()
    plt.tight_layout()
    
    # Adjust y axes
    ytop = np.min(ylims[:,0])
    ybottom = np.max(ylims[:,1])
    axs[0].set_ylim(ytop, ybottom)
    axs[1].set_ylim(ytop, ybottom)
    
    fig.subplots_adjust(top=0.94)
    if approach == 'a':
        fig.suptitle("Survey " + str(sel_ind) + ": Band " + str(band_n) + " perturbed with " + str(np.around((delta_c - 1)*100)) + " percent increase", fontsize = 16)
    elif approach == 'b':
        fig.suptitle("Survey " + str(sel_ind) + ": Perturbed with random numbers (band " + band_names[band_n] + ")", fontsize = 16)
    elif approach == 'c':
        fig.suptitle("Survey " + str(sel_ind) + ": Perturbed with gaussian sampling (band " + band_names[band_n] + ")", fontsize = 16)
    elif approach == 'd':
        fig.suptitle("Survey " + str(sel_ind) + ": Perturbed with gaussian sampling (all bands)", fontsize = 16)
    return fig, axs

####################################
# Plot sigma or outFrac histograms #
####################################

def plot_metric(metric, metric_name, approach = None, band_n = None, bins = np.linspace(0, 1.5, 15), save = False, fig = None, ax = None, **kwargs):
    band_names = ["u-g", "g-r", "r-i", "i-z", "mag(i)"]
    approaches_dict = {'a': "percent increase", 'b': "random numbers (one band)", 'c': "gaussian sampling (one band)", 'd': "gaussian sampling (all bands)"}
    
    if fig == None:
        fig, ax = plt.subplots()
    ax.plot(bins, metric, **kwargs)
    ax.set_xlabel("z bins")
    ax.set_ylabel(metric_name)
    
    if approach is not None:
        if band_n is None:
            ax.set_title("Perturbation approach: " + approaches_dict.get(approach))
        else:
            ax.set_title("Perturbation approach: " + approaches_dict.get(approach) + ", band " + band_names[band_n])
    if save:
        fig.savefig(metric_name + "_bins_" + approach + ".png")
    return fig, ax

#####################
# Remove Bad Values #
#####################

def rm_bad_vals(X_test, X_err, y_test, label_test):
    mask =  np.where( 
            (X_err[:, 0] > 0 ) &
            (X_err[:, 1] > 0 ) &
            (X_err[:, 2] > 0 ) &
            (X_err[:, 3] > 0 ) &
            (X_err[:, 4] > 0 ) )
    X_test_new = X_test[mask]
    X_err_new = X_err[mask]
    y_test_new = y_test[mask]
    label_test_new = label_test[mask]
    return X_test_new, X_err_new, y_test_new, label_test_new