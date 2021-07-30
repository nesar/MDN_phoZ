import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.models import load_model, Sequential, Model

# Activate TF2 behavior:
from tensorflow.python import tf2
if not tf2.enabled():
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    assert tf2.enabled()

np.random.seed(12211)  

#############
# Trim data #
#############

def minmax_cuts(X, y, mins_and_maxs):
    
    min_col, max_col, min_mag, max_mag, min_z, max_z = mins_and_maxs
    print(min_col, max_col)
    mask_cond =  np.where( 
        (X[:, 0] < max_col[0]) & (X[:, 0] > min_col[0]) &
        (X[:, 1] < max_col[1]) & (X[:, 1] > min_col[1]) &
        (X[:, 2] < max_col[2]) & (X[:, 2] > min_col[2]) &
        (X[:, 3] < max_col[3]) & (X[:, 3] > min_col[3]) &
        (X[:, 4] < max_mag) & (X[:, 4] > min_mag) &
        (y < max_z) & (y > min_z) )
    # Why was it this way before?
        #(X[:, 0] < max_col) & (X[:, 0] > min_col) &
        #(X[:, 1] < max_col) & (X[:, 1] > min_col) &
        #(X[:, 2] < max_col) & (X[:, 2] > min_col) &
        #(X[:, 3] < max_col) & (X[:, 3] > min_col) &
        #(X[:, 4] < max_mag) & (X[:, 4] > min_mag) &
        #(y < max_z) & (y > min_z) )
    
    X_new = X[mask_cond]
    y_new = y[mask_cond]
    return X_new, y_new, mask_cond

def minmax_cutsOBSarr(X, X_err, y, l, mins_and_maxs): # Same, with labels
    
    min_col, max_col, min_mag, max_mag, min_z, max_z = mins_and_maxs
    print(min_col)
    print(min_col[0])
    print(X.shape)
    print(X[:,0].shape)
    print((X[:, 0] < max_col[0]).shape)
    mask_cond =  np.where( 
        (X[:, 0] < max_col[0]) & (X[:, 0] > min_col[0]) &
        (X[:, 1] < max_col[1]) & (X[:, 1] > min_col[1]) &
        (X[:, 2] < max_col[2]) & (X[:, 2] > min_col[2]) &
        (X[:, 3] < max_col[3]) & (X[:, 3] > min_col[3]) &
        (X[:, 4] < max_mag) & (X[:, 4] > min_mag) &
        (y < max_z) & (y > min_z) )
    

    X_new = X[mask_cond]
    y_new = y[mask_cond]
    l_new = l[mask_cond]
    X_err_new = X_err[mask_cond]
    return X_new, y_new, l_new, mask_cond, X_err_new


def minmax_cuts_general(X, y, mins_and_maxs):
    min_col, max_col, min_z, max_z = mins_and_maxs
    print(min_col, max_col)
    mask_cond =  np.where(
        (X[:, 0] < max_col[0]) & (X[:, 0] > min_col[0]) &
        (X[:, 1] < max_col[1]) & (X[:, 1] > min_col[1]) &
        (X[:, 2] < max_col[2]) & (X[:, 2] > min_col[2]) &
        (X[:, 3] < max_col[3]) & (X[:, 3] > min_col[3]) &
        (X[:, 4] < max_col[4]) & (X[:, 4] > min_col[4]) &
        (X[:, 5] < max_col[5]) & (X[:, 5] > min_col[5]) &
        (y < max_z) & (y > min_z) )
    X_new = X[mask_cond]
    y_new = y[mask_cond]
    return X_new, y_new, mask_cond

################
# Print limits #
################

def print_limits(X, y):
    print(10*'-')
    print('number of datapoints: ', str(y.shape[0]))
    print('z-minmax: ', y.min(), y.max())
    print('ColMag-min: ', np.min(X, axis=0))
    print('ColMag-max: ', np.max(X, axis=0))
    print(10*'-')
    
################
# Shuffle Data #
################

def shuffle(X, y):
    shuffleOrder = np.arange(X.shape[0])
    np.random.shuffle(shuffleOrder)
    X = X[shuffleOrder]
    y = y[shuffleOrder]
    return X, y, shuffleOrder

def shuffleOBS(X, y, l): # Same, with labels
    shuffleOrder = np.arange(X.shape[0])
    np.random.shuffle(shuffleOrder)
    X = X[shuffleOrder]
    y = y[shuffleOrder]
    l = l[shuffleOrder]
    return X, y, l, shuffleOrder

#################
# Resample Data #
#################

# Training data is biased towards lower redshift values -- resampling fixes that issue

def resample(X, y, n_bins = 200, select_per_bin = 500): # 50 #np.int(num_train/n_bins) #100 # This was an alternative way to control num_train. Without num train requisites, it will just go through the whole training set

    bins = np.linspace(y.min(), y.max(), n_bins)
    # bins = np.logspace(np.log10(y_test.min()+1e-2), np.log10(y_test.max()+1e-2), n_bins)
    inds = np.digitize(y, bins) # This is a way to sort data (y) into bins (right?)

    resampled_ind = [] # index? Like a mask?

    for ind_i in range(n_bins):
        ind_bin = np.where(inds==ind_i)
        random_choices = np.min( [select_per_bin, np.size(ind_bin) ])
        index = np.random.choice(ind_bin[0], random_choices, replace=False)
        resampled_ind = np.append(resampled_ind, index)

    resampled_ind = resampled_ind.astype('int')
    all_ind = np.arange(y.shape[0])
    # resampled_ind_not = ~np.in1d(np.arange(all_ind.shape[-1]), resampled_ind)

    plt.figure(23)
    print(y)
    plt.hist(y, density=True, bins = n_bins, histtype='step', label='original')
    y_resampled = y[resampled_ind]
    X_resampled = X[resampled_ind]

    plt.hist(y_resampled, density=True, bins = n_bins, histtype='step', label='resampled')
    #plt.hist(y_train, density=True, bins = n_bins, histtype='step', label='rest') # For us, "y_train" and "y" are the same
    
    plt.legend()
    plt.savefig("training_plots/resampled.png")
    plt.show()

    print(y_resampled.shape)
    
    return X_resampled, y_resampled, resampled_ind

##################################
# Load Training and Testing Data #
##################################

def loadTrainTest_july(dirIn = '../../Data/fromGalaxev/photozs/datasets/data_july_2020/'):

    # train_data = np.load(dirIn + 'july13_100k.npy')
    train_data = np.load(dirIn + 'july14_200k.npy') # load in the training data
    test_data = np.load(dirIn + 'july13_10k.npy')   # load in the test data
    #print(train_data)
    #print(test_data)
    
    X_train = train_data['color']#[train_data['redshift_flags'] == 0] # Specifically, draw out the colors (inputs) and redshifts (outputs) of the training data
    y_train = train_data['redshift']#[train_data['redshift_flags'] == 0]

    # data['colors'] #  colors (ngal, ncols)
    # data['redshifts'] # redshifts

    X_test = test_data['color']#[test_data['redshift_flags'] == 0] # Same for the test data
    y_test = test_data['redshift']#[test_data['redshift_flags'] == 0]

    print_limits(X_train, y_train)
    print_limits(X_test, y_test)

    return X_train, y_train, X_test, y_test

def loadTrainTest_custom_randz(Testset, sim = 'des', train_dirIn = '/data/a/cpac/nramachandra/Projects/phoZ/Synthetic_Data/fsps_wrapper/notebooks/out/rand_z/', nbands = 5, frac_train = 0.9):

    print("Training on sim: ", sim)
    
    # Load in (raw) training data, reshape
    '''
    col_mag = np.load(train_dirIn + 'col_mag_' + sim + '_0.npy')
    nprop, ngal, nz, ncol = col_mag.shape
    fsps_reshaped = col_mag.reshape(-1, nz, ncol) # Keep the last two dimensions, crush the others
    fsps_reshaped = fsps_reshaped.reshape(-1, ncol) # And then crush the remaining one?
    
    zz_all = np.load(train_dirIn + 'zz_'+ sim +'.npy')
    ngal, nprop, nz = zz_all.shape
    zz_all_reshaped = zz_all.reshape(-1, nz)
    zz_all_reshaped = zz_all_reshaped.reshape(-1)
    '''
    if sim == "irac":
        D = 4
        
    elif sim == "wise":
        D = 4
    elif sim == "des" or sim == "sdss":
        D = 5
    elif sim == "lsst":
        D = 6
    elif sim == "jwst":
        D = 8
    elif sim == "pau":
        D = 46 # Should this be 40?
    elif sim == "spherex":
        D = 102
        
    fsps_reshaped = np.zeros(shape=(1, nbands))
    zz_reshaped = np.zeros(shape=(1))

    for seed_id in range(100):

        col_mag = np.load(train_dirIn + 'col_mag_z_' + str(seed_id)+ sim + '_0.npy')
        # ngal, nprop, nz, ncol = col_mag.shape
        nprop, ngal, nz, ncol = col_mag.shape

        fsps_reshaped0 = col_mag.reshape(-1, nz, ncol)
        fsps_reshaped = np.append( fsps_reshaped, fsps_reshaped0.reshape(-1, ncol), axis=0)

        zz_all = np.load(train_dirIn + 'zz_z_'+ str(seed_id) + sim +'.npy')
        #print(zz_all.shape)

        ngal, nprop, nz = zz_all.shape
        zz_reshaped0 = zz_all.reshape(-1, nz)
        zz_reshaped = np.append( zz_reshaped, zz_reshaped0.reshape(-1) )

    fsps_reshaped = fsps_reshaped[1:, :]
    zz_all_reshaped = zz_reshaped[1:]
    
    # Randomly sample 90% and 10% for training and testing # Should we even bother with this if we're using Lindsey's test data?
    sample_size = fsps_reshaped.shape[0] # first dimension
    train_n = int(frac_train*sample_size) # note: rounds down
    np.random.seed(0)
    full_idx = np.arange(sample_size) # indices for all datapoints, ordered
    training_idx = np.random.choice(full_idx, train_n, replace = False) # pick a subset of those indices for training
    X_train = fsps_reshaped[training_idx]
    y_train = zz_all_reshaped[training_idx]
    X_test = fsps_reshaped[full_idx[~np.isin(full_idx, training_idx)]]  # the indices not part of the subset will be used for testing
    y_test = zz_all_reshaped[full_idx[~np.isin(full_idx, training_idx)]]
    
    print("X training data shape: ", X_train.shape)
    print("y training data shape: ", y_train.shape)
    #X_err = np.load(test_dirIn + 'test_' + Testset +'_err.npy')
    #test_labels = np.load(test_dirIn + 'test_' + Testset + '_label.npy')
    
    #print_limits(fsps_reshaped, zz_all_reshaped)
    #print_limits(X_test, y_test)

    return X_train, y_train, X_test, y_test#, X_err, test_labels


def loadTrainTest_custom(Testset, sim = 'des', train_dirIn = '/data/a/cpac/nramachandra/Projects/phoZ/Synthetic_Data/fsps_wrapper/notebooks/out/', frac_train = 0.9):

    print("Training on sim: ", sim)
    
    # Load in (raw) training data, reshape
    col_mag = np.load(train_dirIn + 'col_mag_' + sim + '_0.npy')
    nprop, ngal, nz, ncol = col_mag.shape
    fsps_reshaped = col_mag.reshape(-1, nz, ncol) # Keep the last two dimensions, crush the others
    fsps_reshaped = fsps_reshaped.reshape(-1, ncol) # And then crush the remaining one?
    
    zz_all = np.load(train_dirIn + 'zz_'+ sim +'.npy')
    ngal, nprop, nz = zz_all.shape
    zz_all_reshaped = zz_all.reshape(-1, nz)
    zz_all_reshaped = zz_all_reshaped.reshape(-1)
    
    print("X training data shape: ", fsps_reshaped.shape)
    print("y training data shape: ", zz_all_reshaped.shape)
    
    # Randomly sample 90% and 10% for training and testing
    sample_size = fsps_reshaped.shape[0] # first dimension
    train_n = int(frac_train*sample_size) # note: rounds down
    np.random.seed(0)
    full_idx = np.arange(sample_size)
    training_idx = np.random.choice(full_idx, train_n, replace = False)
    X_train = fsps_reshaped[training_idx]
    y_train = zz_all_reshaped[training_idx]
    X_test = fsps_reshaped[full_idx[~np.isin(full_idx, training_idx)]]
    y_test = zz_all_reshaped[full_idx[~np.isin(full_idx, training_idx)]]
    
    #X_err = np.load(test_dirIn + 'test_' + Testset +'_err.npy')
    #test_labels = np.load(test_dirIn + 'test_' + Testset + '_label.npy')
    
    #print_limits(fsps_reshaped, zz_all_reshaped)
    #print_limits(X_test, y_test)

    return X_train, y_train, X_test, y_test#, X_err, test_labels


def loadTrainTest_custom_future_ed(Testset, sim = 'des', train_dirIn = '/data/a/cpac/nramachandra/Projects/phoZ/Synthetic_Data/Data/', test_dirIn = '/data/a/cpac/aurora/MDN_phoZ/Data/fromGalaxev/photozs/datasets/data_feb_2021/'):

    # train_data = np.load(dirIn + 'july13_100k.npy')
    #X_train = np.load(train_dirIn + sim + '_col_mag.npy') # load in the training data
    #y_train = np.load(train_dirIn + sim + '_z.npy')
    X_train = np.load(train_dirIn + sim + '_col_mag.npy') # load in the training data
    y_train = np.load(train_dirIn + sim + '_z.npy')
    test_data = np.load(test_dirIn + 'test_' + Testset + '.npy')   # load in the test data
    
    # For now, use 10% of X data for testing
    test_data = np.load(test_dirIn + 'test_' + Testset + '.npy')   # load in the test data
    print("Training on sim: ", sim)
    print("X training data shape: ", X_train.shape)
    print("y training data shape: ", y_train.shape)

    X_test = test_data[: , :-1]
    y_test = test_data[: , -1]
    
    X_err = np.load(test_dirIn + 'test_' + Testset +'_err.npy')
    test_labels = np.load(test_dirIn + 'test_' + Testset + '_label.npy')
    
    print_limits(X_train, y_train)
    print_limits(X_test, y_test)

    return X_train, y_train, X_test, y_test, X_err, test_labels
    

# Not clear why these are separate?
def loadTest(Testset, dirIn = '../../Data/fromGalaxev/photozs/datasets/data_feb_2021/'):
    
    test_data = np.load(dirIn + 'test_' + Testset + '.npy') # didn't we just do this in the last function? (loadTrainTest_july(dirIn))

    X_test = test_data[: , :-1]
    y_test = test_data[: , -1]

    print_limits(X_test, y_test)

    X_err = np.load(dirIn + 'test_' + Testset +'_err.npy')
    test_labels = np.load(dirIn + 'test_' + Testset + '_label.npy')

    return X_test, y_test, X_err, test_labels # yeah, why not just combine these?

def loadTest_lindsey(Testset, dirIn = '/data/a/cpac/nramachandra/Projects/phoZ/SurveyTrain/TestingDataLindsey/selected_des_irac.npy'):
    
    test_data = np.load(dirIn + 'test_' + Testset + '.npy') # didn't we just do this in the last function? (loadTrainTest_july(dirIn))

    X_test = test_data[: , :-1]
    y_test = test_data[: , -1]

    #print_limits(X_test, y_test)

    #X_err = np.load(dirIn + 'test_' + Testset +'_err.npy')
    #test_labels = np.load(dirIn + 'test_' + Testset + '_label.npy')

    return X_test, y_test, X_err, test_labels # yeah, why not just combine these?

#########
# Decay #
#########

def decay(epoch, decay_rate, learning_rate): # Oh hey I vaguely remember this!
    if (epoch < 1):
        return learning_rate
    else:
        return learning_rate*(1.0/(1.0+decay_rate*(epoch)))
    
##############
# Prediction #
##############

def prediction(X_test, save_mod, surveystring, model_train, preproc, preproc_y):
    f_real = X_test.copy()
    f_real[:,:4] = f_real[:,:4]
    f_real = preproc.transform(f_real)
    y_pred = np.array(model_train(  [f_real, np.zeros(shape = f_real.shape[0]) ] ))
    y_pred_arg = np.argmax(y_pred[2, :, :], axis = 1)
    y_pred_mean = y_pred[0, :, :][:, y_pred_arg][:, 0]
    y_pred_std = np.sqrt(np.log(y_pred[1, :, :][:, y_pred_arg][:, 0]))
    y_pred_3means = preproc_y.inverse_transform(y_pred[0, :, :]) # pretty sure this is the same as y_pred_std.reshape(-1, 1)[:, 0]??
    y_pred_3std = preproc_y.inverse_transform( np.sqrt(np.log(y_pred[1, :, :])  ))
    y_pred_3weights = y_pred[2, :, :]
    predstdweights = np.array([y_pred_3means, y_pred_3std, y_pred_3weights]) # Should I be returning this?    
    return y_pred_3means, y_pred_3std, y_pred_3weights, y_pred_arg, y_pred_mean, y_pred_std # photo_z, photo_z_err
