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
    mask_cond =  np.where( 
        (X[:, 0] < max_col) & (X[:, 0] > min_col) &
        (X[:, 1] < max_col) & (X[:, 1] > min_col) &
        (X[:, 2] < max_col) & (X[:, 2] > min_col) &
        (X[:, 3] < max_col) & (X[:, 3] > min_col) &
        (X[:, 4] < max_mag) & (X[:, 4] > min_mag) &
        (y < max_z) & (y > min_z) )
    
    X_new = X[mask_cond]
    y_new = y[mask_cond]
    return X_new, y_new, mask_cond

def minmax_cutsOBSarr(X, X_err, y, l, mins_and_maxs): # Same, with labels
    
    min_col, max_col, min_mag, max_mag, min_z, max_z = mins_and_maxs
    mask_cond =  np.where( 
        (X[:, 0] < max_col[0]) & (X[:, 0] > min_col[0]) &
        (X[:, 1] < max_col[1]) & (X[:, 1] > min_col[1]) &
        (X[:, 2] < max_col[2]) & (X[:, 2] > min_col[2]) &
        (X[:, 3] < max_col[3]) & (X[:, 3] > min_col[3]) & 
        (X[:, 4] < max_mag) & (X[:, 4] > min_mag) &
        (y < max_z) & (y > min_z) )
    

    X_new = X[mask_cond]
    X_err_new = X_err[mask_cond]
    y_new = y[mask_cond]
    l_new = l[mask_cond]
    
    return X_new, X_err_new, y_new, l_new, mask_cond

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

def resample(X, y, n_bins = 200, select_per_bin = 500): # 50 #np.int(num_train/n_bins) #100 # This was an alternative way to control num_train. Without num train requisites, it will jsut go through the whole training set

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
    plt.hist(y, density=True, bins = n_bins, histtype='step', label='original')
    y_resampled = y[resampled_ind]
    X_resampled = X[resampled_ind]

    plt.hist(y_resampled, density=True, bins = n_bins, histtype='step', label='resampled')
    #plt.hist(y_train, density=True, bins = n_bins, histtype='step', label='rest') # For us, "y_train" and "y" are the same
    
    plt.legend()
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
    
    
    X_train = train_data['color']#[train_data['redshift_flags'] == 0] # Specifically, draw out the colors (inputs) and redshifts (outputs) of the training data
    y_train = train_data['redshift']#[train_data['redshift_flags'] == 0]

    # data['colors'] #  colors (ngal, ncols)
    # data['redshifts'] # redshifts

    X_test = test_data['color']#[test_data['redshift_flags'] == 0] # Same for the test data
    y_test = test_data['redshift']#[test_data['redshift_flags'] == 0]

    print_limits(X_train, y_train)
    print_limits(X_test, y_test)

    return X_train, y_train, X_test, y_test

def loadTrainTest_custom(train_data_dirIn = '/data/a/cpac/nramachandra/Projects/phoZ/Synthetic_Data/Data/lsst_col_mag.npy', test_data_dirIn = '/data/a/cpac/nramachandra/Projects/phoZ/Synthetic_Data/Data/lsst_z.npy'):

    # train_data = np.load(dirIn + 'july13_100k.npy')
    train_data = np.load(train_data_dirIn) # load in the training data
    test_data = np.load(test_data_dirIn)   # load in the test data
    
    
    X_train = train_data['color']#[train_data['redshift_flags'] == 0] # Specifically, draw out the colors (inputs) and redshifts (outputs) of the training data
    y_train = train_data['redshift']#[train_data['redshift_flags'] == 0]

    # data['colors'] #  colors (ngal, ncols)
    # data['redshifts'] # redshifts

    X_test = test_data['color']#[test_data['redshift_flags'] == 0] # Same for the test data
    y_test = test_data['redshift']#[test_data['redshift_flags'] == 0]

    print_limits(X_train, y_train)
    print_limits(X_test, y_test)

    return X_train, y_train, X_test, y_test

# Not clear why these are separate?
def loadTest(Testset, dirIn = '../../Data/fromGalaxev/photozs/datasets/data_feb_2021/'):
    
    test_data = np.load(dirIn + 'test_' + Testset + '.npy') # didn't we just do this in the last function? (loadTrainTest_july(dirIn))

    X_test = test_data[: , :-1]
    y_test = test_data[: , -1]

    print_limits(X_test, y_test)

    X_err = np.load(dirIn + 'test_' + Testset +'_err.npy')
    test_labels = np.load(dirIn + 'test_' + Testset + '_label.npy')

    return X_test, y_test, X_err, test_labels # yeah, why not just combine these?

#########
# Decay #
#########

def decay(epoch, decay_rate, learning_rate): # Oh hey I vaguely remember this!
    if (epoch < 1):
        return learning_rate
    else:
        return learning_rate*(1.0/(1.0+decay_rate*(epoch)))