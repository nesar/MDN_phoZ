# Inject noise

# will this work here?
import help_train
import help_funcs
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import tensorflow.keras as keras
import tensorflow_probability as tfp
tfd = tfp.distributions
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.python import tf2 # Activate TF2 behavior:
if not tf2.enabled():
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    assert tf2.enabled()

np.random.seed(12211)
train_mode = True

# Might not need this?
#import SetPub
#SetPub.set_pub()

########################
# Load hyperparameters #
########################

import argparse

# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument("-sim", dest='sim', default = 'sdss', type = str, required = False)
parser.add_argument("-ntrain", dest='num_train', default = 200000, type = int, required = False)
parser.add_argument("-ntest", dest='num_test', default = 20000, type = int, required = False)
parser.add_argument("-frac_train", dest='frac_train', default = 0.9, type = float, required = False)
parser.add_argument("-nepochs", dest='n_epochs', default = 100, type = int, required = False)
#parser.add_argument("-D", dest='D', default = 5, help = "number of features", type = int, required = False)
parser.add_argument("-K", dest='K', default = 3, help = "number of mixture components", type = int, required = False)
parser.add_argument("-lr", dest='learning_rate', default = 1e-4, type = float, required = False)
parser.add_argument("-dr", dest='decay_rate', default = 1e-2, type = float, required = False)
parser.add_argument("-bs", dest='batch_size', default = 256, type = int, required = False)
# Resampling parameters
parser.add_argument("-re", dest = "resampleTrain", default = False, type = bool, required = False) # HARD CODE
parser.add_argument("-nbins", dest = "nbins", default = 200, type = int, required = False)
parser.add_argument("-spb", dest = "select_per_bin", default = 400, type = int, required = False)
parser.add_argument("-prtb", dest = "prtb", default = False, type = bool, required = False) # HARD CODE
parser.add_argument("-rm_band", dest = "rm_band", default = False, type = bool, required = False) # HARD CODE
parser.add_argument("-sim2", dest='sim2', default = None, type = str, required = False)
parser.add_argument("-D2", dest='D2', default = None, type = int, required = False)
parser.add_argument("-use_lindseys_test", dest = "use_lindseys_test", default = False, type = bool, required = False)
parser.add_argument("-ntrain_points", dest = "ntrain_points", default = 100000, type = int, required = False)
parser.add_argument("-std", dest = "std", default = 0.1, type = float, required = True)
parser.add_argument("-factor", dest = "factor", default = 4, type = int, required = False)
# Copy this: python train_ALL_the_all_the_things.py -sim 'jwst' -ntrain 200000 -ntest 20000 -nepochs 20 -K 3 -lr 1e-4 -dr 1e-2 -bs 256 -re False -nbins 200 -spb 400 -prtb False -rm_band False

# This is probably a pretty lazy solution...
args = parser.parse_args()
print(args)
sim = args.sim
num_train = args.num_train
num_test = args.num_test
frac_train = args.frac_train
n_epochs = args.n_epochs
#D = args.D # This now gets assigned later
K = args.K
learning_rate = args.learning_rate
decay_rate = args.decay_rate
batch_size = args.batch_size
resampleTrain = args.resampleTrain
n_bins = args.nbins
select_per_bin = args.select_per_bin
prtb = args.prtb
rm_band = args.rm_band
sim2 = args.sim2
D2 = args.D2
use_lindseys_test = args.use_lindseys_test
ntrain_points = args.ntrain_points
std = args.std
factor = args.factor

#######################
# Constant parameters #
#######################

Trainset = ['FSPSlin', 'FSPSlog', 'FSPSall', 'OBS', 'UM', 'BP', 'UMnew'][6] # Soon there will be more!
Testset = ['FSPSlin', 'FSPSlog', 'FSPSall', 'OBS', 'UM', 'BP', 'UMnew', 'OBSuq'][7] # Test on the same things we tested before (SDSS)
surveystring = ['SDSS', 'VIPERS', 'DEEP2', 'PRIMUS']

# How to do this with less hardcoding?
if sim == "irac" or sim == "wise":
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
    
# Make dictionaries
des_bands = {'g-r': 0, 'r-i': 1, 'i-z': 2, 'z-y': 3, 'mag(g)': 4} # or is it mag(i)?
irac_bands = {'I2-I1': 0, 'I3-I2': 1, 'I4-I3': 2, 'mag(I1)': 3}
wise_bands = {'w2-w1': 0, 'w3-w2': 1, 'w4-w3': 2, 'mag(w1)': 3}

#############################################
# Load in training/testing data, shuffle it #
#############################################

train_dirIn = '/data/a/cpac/nramachandra/Projects/phoZ/Synthetic_Data/fsps_wrapper/notebooks/out/rand_z/'
test_dirIn = '/data/a/cpac/aurora/MDN_phoZ/Data/fromGalaxev/photozs/datasets/data_feb_2021/'

if use_lindseys_test: # assumes 'des' and 'irac', in that order
    print("using lindsey's test")
    suffix = "_lindsey"
    # Load training data
    X_train1, y_train1, X_test1, y_test1 = help_train.loadTrainTest_custom_randz(Testset, sim, train_dirIn, nbands = D, frac_train = frac_train)
    X_train2, y_train2, X_test2, y_test2 = help_train.loadTrainTest_custom_randz(Testset, sim2, train_dirIn, nbands = D2, frac_train = frac_train)
    
    # Match training data with Lindsey's code
    if sim2 == 'irac': # COME BACK AND MAKE THIS LESS HARD CODED
        X_train1_subsample = np.delete(X_train1, des_bands['z-y'], axis = 1)
        X_train2_subsample = np.delete(X_train2, [irac_bands['I3-I2'], irac_bands['I4-I3']], axis = 1)
        del des_bands['z-y']
        del irac_bands['I3-I2'] # How are you going to update those values?
        del irac_bands['I4-I3']
        # New band: combine des_mag(g) and irac_mag(i1)
        irac_bands['mag(g)-mag(I1)'] = len(list(irac_bands)) # One more than the last one
        new_band = np.array([X_train1_subsample[:,-1] - X_train2_subsample[:,-1]]).T
        X_train2_subsample = np.concatenate((X_train2_subsample, new_band), axis = 1) # Add new band to the end
        D = len(list(des_bands))
        D2 = len(list(irac_bands))
        sim2_bands = irac_bands
        print("End of if")
    else:
        print("Please use des and irac with lindsey's data")
    # Oops, guess this isn't set up yet
    #elif sim2 == 'wise':
    #    X_train1_subsample = np.delete(X_train1, des_bands['z-y'], axis = 1)
    #    X_train2_subsample = np.delete(X_train2, [wise_bands['w3-w2'], wise_bands['w4-w3']], axis = 1)
    #    del des_bands['z-y']
    #    del wise_bands['w3-w2']
    #    del wise_bands['w4-w3']
    #    D = len(list(des_bands))
    #    D2 = len(list(wise_bands))
    #    sim2_bands = wise_bands
    
    X_train = np.concatenate((X_train1_subsample, X_train2_subsample), axis = 1)
    y_train = y_train1
    
    # Load testing data
    #X_test = np.load('/data/a/cpac/nramachandra/Projects/phoZ/SurveyTrain/TestingDataLindsey/selected_des_irac.npy')
    #y_test = np.load('/data/a/cpac/nramachandra/Projects/phoZ/SurveyTrain/TestingDataLindsey/selected_des_irac_z.npy')
    X_test = np.load('/data/a/cpac/nramachandra/Projects/phoZ/SurveyTrain/TestingDataLindsey/magi_selected_des_irac_combined.npy')
    y_test = np.load('/data/a/cpac/nramachandra/Projects/phoZ/SurveyTrain/TestingDataLindsey/magi_selected_des_irac_z_combined.npy')
    print("X_test is:\n", X_test)
    
    # Update D
    D = D + D2 # Kind of HARD CODE?
    print(D)
    
else:
    print("Not using Lindsey's test")
    suffix = ""
    if sim2 is None:
        X_train, y_train, X_test, y_test = help_train.loadTrainTest_custom_randz(Testset, sim, train_dirIn, nbands = D, frac_train = 0.9) # Need X_err and test_labels
    else:
        X_train1, y_train1, X_test1, y_test1 = help_train.loadTrainTest_custom_randz(Testset, sim, train_dirIn, nbands = D, frac_train = frac_train)
        X_train2, y_train2, X_test2, y_test2 = help_train.loadTrainTest_custom_randz(Testset, sim2, train_dirIn, nbands = D2, frac_train = frac_train)
        X_train = np.concatenate((X_train1, X_train2), axis = 1)
        y_train = y_train1
        X_test = np.concatenate((X_test1, X_test2), axis = 1)
        y_test = y_test1 # They should both be identical
        print(X_train)
        D = D + D2

########################################################
# Update Hyperparameters (if removing a band) and save #
########################################################

if rm_band:
    D = D - 1 # number of features (in input space)
    
###################################
# Inject noise into training data #
###################################

import copy
# Injecting noise
#print(X_train.shape)
#print(X_train)

#print(y_train.shape)
#print(y_train)

sigma = std
nsamples = factor - 1 #since there's originally one row per data point
ncolors = X_train.shape[1]

new_X_train = copy.deepcopy(X_train[:ntrain_points, :])
new_y_train = copy.deepcopy(y_train[:ntrain_points])
print("old training sets")
print(new_X_train.shape, new_y_train.shape)
print(new_X_train)
print(new_y_train)

# Make a gaussian for each color point
for i, this_row in enumerate(new_X_train):
    new_rows = np.empty((nsamples, ncolors))
    for j, this_color in enumerate(this_row):
        mu = this_color
        new_samples = np.random.normal(mu, sigma, nsamples)
        new_rows[:,j] = new_samples
    new_X_train = np.vstack((new_X_train, new_rows))
    new_y_train = np.concatenate((new_y_train, [new_y_train[i]]*nsamples))

print("new training sets")
print(new_X_train.shape, new_y_train.shape)
print(new_X_train)
print(new_y_train)

pickle.dump(new_X_train, open( b"new_X_train_" + str(std).encode('ascii') + b".obj","wb" ))
pickle.dump(new_y_train, open( b"new_y_train_" + str(std).encode('ascii') + b".obj","wb" ))
