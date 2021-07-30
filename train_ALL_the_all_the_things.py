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
print("past imports")

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
#parser.add_argument("-use_injected_noise", dest = "use_injected_noise", default = False, type = boolean_string, required = False)
parser.add_argument("-std", dest = "std", default = 0.1, type = float, required = False)
parser.add_argument("-use_injected_noise", dest = "use_injected_noise", action='store_true')
parser.set_defaults(use_injected_noise=False)

print("past arg defs")
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
use_injected_noise = args.use_injected_noise
std = args.std

print("past arg parsing")

print("Using injected noise?: ", use_injected_noise)

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
        if use_injected_noise:
            print("using injected noise")
            suffix = suffix + "_noisy_std_" + str(std)
            X_train = pickle.load( open( b"new_X_train_" + str(std).encode('ascii') + b".obj","rb" ))
            y_train = pickle.load( open( b"new_y_train_" + str(std).encode('ascii') + b".obj","rb" ))
            
        else: 
            print("not using injected noise")
            X_train1_subsample = np.delete(X_train1, des_bands['z-y'], axis = 1)
            X_train2_subsample = np.delete(X_train2, [irac_bands['I3-I2'], irac_bands['I4-I3']], axis = 1)
            
            # New band
            new_band = np.array([X_train1_subsample[:,-1] - X_train2_subsample[:,-1]]).T
            X_train2_subsample = np.concatenate((X_train2_subsample, new_band), axis = 1) # Add new band to the end

            X_train = np.concatenate((X_train1_subsample, X_train2_subsample), axis = 1)
            y_train = y_train1
            
        # Fix up the dictionaries
        my_bands = {**des_bands, **irac_bands}
        del my_bands['z-y']
        del my_bands['I3-I2'] # How are you going to update those values?
        del my_bands['I4-I3']
        my_bands['mag(g)'] -= 1
        my_bands['I2-I1'] = my_bands['mag(g)'] + 1
        my_bands['mag(I1)'] = my_bands['I2-I1'] + 1
        my_bands['mag(g)-mag(I1)'] = len(list(my_bands)) # One more than the last one
        print("dictionary of bands\n", my_bands)

        # Load testing data
        X_test = np.load('/data/a/cpac/nramachandra/Projects/phoZ/SurveyTrain/TestingDataLindsey/magi_selected_des_irac_combined.npy')
        y_test = np.load('/data/a/cpac/nramachandra/Projects/phoZ/SurveyTrain/TestingDataLindsey/magi_selected_des_irac_z_combined.npy')
        print("X_test is:\n", X_test)

        # New band for testing too # Not necessary for this file I guess?
        #new_test_band = np.array([X_test[:, -1] - X_test[:, -3]]).T
        #X_test = np.concatenate((X_test, new_test_band), axis = 1)

        # Update D
        #D = len(list(des_bands))
        #D2 = len(list(irac_bands))
        #D = D + D2 # Kind of HARD CODE?
        D = len(list(my_bands))
        print(D)
        sim2_bands = irac_bands
        
    else:
        print("Please use des and irac with lindsey's data")
    
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
    
#D = X_train.shape[1] # Hoping to bring this back
# No shuffling currently I guess? Should we have that again? Here it is...

################
# Shuffle Data #
################

X_train, y_train, X_trainShuffleOrder = help_train.shuffle(X_train, y_train) # literally just shuffle the data

########################################################
# Update Hyperparameters (if removing a band) and save #
########################################################

if rm_band:
    D = D - 1 # number of features (in input space)

# Should I add a suffix for "sim" here?
save_mod = '/data/a/cpac/aurora/MDN_phoZ/saved_hubs/tf2models/'+'Train_'+Trainset+'_lr_'+str(learning_rate)+'_dr'+str(decay_rate)+'_ne'+str(n_epochs)+'_k'+str(K)+'_nt'+str(num_train) + suffix

#################
# Trim the data #
#################
# Old
#minmax = False # HARD CODE
#if minmax is True:
#    min_col = [-0.09145837, -0.05327791, -0.02479261, -0.10519464]
#    max_col = [ 3.825315,   2.8303378,  1.6937237,  1.5019817]
#    min_mag = 12
#    max_mag = 23
#    min_z = 0.0 #np.min(y_train)
#    max_z = 1.1 #np.max(y_train)
#    mins_and_maxs = [min_col, max_col, min_mag, max_mag, min_z, max_z]
#    X_test, y_test, label_test, mask_cond = help_train.minmax_cutsOBSarr(X_test, y_test, label_test, mins_and_maxs)
    
# Not exactly sure the difference here?
minmax = False # HARD CODE # False atm because I don't know how to make this work for all training sets (multiple dimensions?)
if minmax is True:
    min_col = np.min(X_train, axis=0)
    max_col = np.max(X_train, axis=0)
    min_z = 0.0 #np.min(y_train)
    max_z = 3.0 #np.max(y_train)
    mins_and_maxs = [min_col, max_col, min_z, max_z]
    X_test, y_test, mask_cond = help_train.minmax_cuts_general(X_test, y_test, mins_and_maxs)
print('=== After trimming ==')
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))

##################################
# Resample to account for z bias #
##################################

# Parameters for this section were parsed above

if resampleTrain:
    X_train, y_train, resampled_ind = help_train.resample(X_train, y_train, n_bins, select_per_bin) # What is "rest"? And where did the original data go?
    
print("Size of features in training data: {}".format(X_train.shape)) # what do we mean features?
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))

###############################
# Rescale (min-max rescaling) #
###############################

preproc = Pipeline([('stdscaler', StandardScaler())]) # This preproc tool is what allows us to transform inputs into min-max space (and therefore to feed it to the trained model)
X_train = preproc.fit_transform(X_train)
scalerfile = save_mod + '_scaling_X'
pickle.dump(preproc, open(scalerfile, 'wb'))
preproc = pickle.load(open(scalerfile, 'rb'))
X_test = preproc.transform(X_test) # Transform: put our input into min-max space
preproc_y = Pipeline([('stdscaler', MinMaxScaler())])
y_train = preproc_y.fit_transform(y_train.reshape(-1, 1))
scalerfile = save_mod + '_scaling_y'
pickle.dump(preproc_y, open(scalerfile, 'wb'))
preproc_y = pickle.load(open(scalerfile, 'rb'))
y_test = preproc_y.transform(y_test.reshape(-1, 1))

########################
# For all future plots #
########################

if sim2 is None:
    param_labels = ["sim: " + sim, "num train: " + str(num_train), "num test: " + str(num_test), "n epochs: " + str(n_epochs), "D: " + str(D), "K: " + str(K), "learning rate: " + str(learning_rate), "decay rate: " + str(decay_rate), "batch size: " + str(batch_size), "n bins: " + str(n_bins), "select per bin: " + str(select_per_bin)]

else:
    if use_lindseys_test:
        print("Using lindseys test, two sims")
        param_labels = ["sim1: " + sim, "sim2: " + sim2, "bands: " + str(list(des_bands)) + " " + str(list(sim2_bands)), "num train: " + str(num_train), "num test: " + str(num_test), "n epochs: " + str(n_epochs), "D: " + str(D), "K: " + str(K), "learning rate: " + str(learning_rate), "decay rate: " + str(decay_rate), "batch size: " + str(batch_size), "n bins: " + str(n_bins), "select per bin: " + str(select_per_bin), "noise: " + str(use_injected_noise)]
    else:
        param_labels = ["sim1: " + sim, "sim2: " + sim2, "num train: " + str(num_train), "num test: " + str(num_test), "n epochs: " + str(n_epochs), "D: " + str(D), "K: " + str(K), "learning rate: " + str(learning_rate), "decay rate: " + str(decay_rate), "batch size: " + str(batch_size), "n bins: " + str(n_bins), "select per bin: " + str(select_per_bin)]

#####################################
# Plot histograms of train and test #
#####################################

help_train.print_limits(X_train, y_train)
help_train.print_limits(X_test, y_test)

fig, ax = plt.subplots(figsize=(10, 10,)) # plt.figure(23)
ax.hist(y_train, density=True, bins = 250, histtype='step', label='train')
ax.hist(y_test, density=True, bins = 250, histtype='step', label='test')

leg1 = ax.legend(fontsize = 'xx-large', markerscale=1., numpoints=2)
fake_lines = [ax.plot([], [], c = "black")[0] for i in range(0,len(param_labels))]

ax.legend(handles = fake_lines, labels = param_labels, loc = "upper right")
ax.add_artist(leg1)
ax.set_title(sim)

if sim2 is None:
    print("sim2 is none")
    plt.savefig("training_plots/" + sim + "/" + sim + "_precision_ntrain" + str(num_train) + "_ntest" + str(num_test) + "_nepochs" + str(n_epochs) + "_D" + str(D) + "_K" + str(K) + "_lr" + str(learning_rate) + "_dr" + str(decay_rate) + "_bs" + str(batch_size) + "_re" + str(resampleTrain) + "_nbins" + str(n_bins) + "_spb" + str(select_per_bin) + ".png")
else:
    print("sim2 is not None")
    plt.savefig("training_plots/" + sim2 + "/" + sim + "_" + sim2 + suffix + "_precision_ntrain" + str(num_train) + "_ntest" + str(num_test) + "_nepochs" + str(n_epochs) + "_D" + str(D) + "_K" + str(K) + "_lr" + str(learning_rate) + "_dr" + str(decay_rate) + "_bs" + str(batch_size) + "_re" + str(resampleTrain) + "_nbins" + str(n_bins) + "_spb" + str(select_per_bin) + ".png")

################################################
# Some stuff to get ready for actual training? #
################################################

#from help_train import decay as decay

def decay(epoch): # Why do we need to supply epoch? Is that what's coming from LearningRateScheduler?
    if (epoch < 1):
        return learning_rate
    else:
        return learning_rate*(1.0/(1.0+decay_rate*(epoch)))
    
class PrintLR(tf.keras.callbacks.Callback): # Print learning rate at every epoch
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model_train.optimizer.lr.numpy()))
        
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

###############################
# Create network architecture #
###############################

# x = tf.keras.layers.InputLayer(input_shape=(D,)),
non_lin_act = tf.nn.relu #tf.nn.tanh
y_true = tf.keras.Input(shape=(1,))
inputs = tf.keras.Input(shape=(D,))
layer_1 = tf.keras.layers.Dense(units=512, activation=non_lin_act)(inputs)
layer_1a = tf.keras.layers.Dense(units=1024, activation=non_lin_act)(layer_1)
layer_1b = tf.keras.layers.Dense(units=2048, activation=non_lin_act)(layer_1a)
layer_1c = tf.keras.layers.Dense(units=1024, activation=non_lin_act)(layer_1b)
layer_2 = tf.keras.layers.Dense(units=512, activation=non_lin_act)(layer_1c)
layer_3 = tf.keras.layers.Dense(units=256, activation=non_lin_act)(layer_2)
layer_4 = tf.keras.layers.Dense(units=128, activation=non_lin_act)(layer_3)
layer_5 = tf.keras.layers.Dense(units=64, activation=non_lin_act)(layer_4)
layer_6 = tf.keras.layers.Dense(units=32, activation=non_lin_act)(layer_5)
mu = tf.keras.layers.Dense(units=K, activation=None, name="mu")(layer_6)
var = tf.keras.backend.exp(tf.keras.layers.Dense(units=K, activation=tf.nn.softplus, name="sigma")(layer_6))
pi = tf.keras.layers.Dense(units=K, activation=tf.nn.softmax, name="mixing")(layer_6)

model_train = Model([inputs, y_true], [mu, var, pi], name='mdn') # Previously, this was the mixed density model (is that the same as a Gaussian mixture model?)

#################
# Loss function #
#################

def custom_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, mu, var, pi):
        mixture_distribution = tfp.distributions.Categorical(probs=pi)
        distribution = tfp.distributions.Normal(loc=mu, scale=var)
        likelihood = tfp.distributions.MixtureSameFamily(mixture_distribution=mixture_distribution,components_distribution=distribution)

        log_likelihood = -1.0*likelihood.log_prob(tf.transpose(y_true)) # A little confusing (talk later)
        mean_loss = tf.reduce_mean(log_likelihood)

        return mean_loss
    return loss
    
model_train.add_loss(custom_loss(inputs)(y_true, mu, var, pi))
model_train.compile(optimizer='Nadam')
model_train.summary()

########################################
# Perturb training data if you want to #
# (not the same as injecting noise)    #
########################################

if rm_band:
    band_n = 0 # remove the u band
    approach = 'e'
    prtb_X_train = help_funcs.perturb(X_train, band_n, approach = approach)
    prtb_X_test = help_funcs.perturb(X_test, band_n, approach = approach)
    prtb_suffix = '_perturbed'
    X_train = prtb_X_train
    X_test = prtb_X_test
else:
    prtb_suffix = ""

##########
# Train! #
##########

if train_mode:

    history = model_train.fit([X_train, y_train], validation_data=(X_test, y_test), validation_split = 0.1, epochs=n_epochs, batch_size = batch_size, callbacks=callbacks)
    #history = model_train.fit([X_train, y_train], validation_split = 0.1, epochs=n_epochs, batch_size = batch_size, callbacks=callbacks)
    model_train.save_weights(save_mod + '.h5')
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epochs', fontsize = 28)
    plt.ylabel('Loss', fontsize = 28)
    plt.title(sim)
    
    # Fancy legend
    fake_lines = [plt.plot([], [], c = "black")[0] for i in range(0,len(param_labels))]
    plt.legend(handles = fake_lines, labels = param_labels, loc = "upper right")

    if sim2 is None:
        plt.savefig("training_plots/" + sim + "/" + sim + "_loss_epochs_ntrain" + str(num_train) + "_ntest" + str(num_test) + "_nepochs" + str(n_epochs) + "_D" + str(D) + "_K" + str(K) + "_lr" + str(learning_rate) + "_dr" + str(decay_rate) + "_bs" + str(batch_size) + "_re" + str(resampleTrain) + "_nbins" + str(n_bins) + "_spb" + str(select_per_bin) + ".png")
    else:
        print("sim2 is not none")
        plt.savefig("training_plots/" + sim2 + "/" + sim + "_" + sim2 + suffix + "_loss_epochs_ntrain" + str(num_train) + "_ntest" + str(num_test) + "_nepochs" + str(n_epochs) + "_D" + str(D) + "_K" + str(K) + "_lr" + str(learning_rate) + "_dr" + str(decay_rate) + "_bs" + str(batch_size) + "_re" + str(resampleTrain) + "_nbins" + str(n_bins) + "_spb" + str(select_per_bin) + ".png")

#########################
# Save Training Outputs #
#########################

model_train.load_weights(save_mod + '.h5') # Previously, this exact line was loading the weights of (from?) the mixed density model

# Is this all correct? (For no zp corrections of course)
# Old stuff now...
#y_pred = np.array(model_train(  [X_test, np.zeros(shape = X_test.shape[0]) ] ))
#y_pred_arg = np.argmax(y_pred[2, :, :], axis = 1)
#y_pred_mean = y_pred[0, :, :][:, y_pred_arg][:, 0]
#y_pred_std = np.sqrt(np.log(y_pred[1, :, :][:, y_pred_arg][:, 0]))

idx = np.arange(len(X_test))
np.random.seed(0)
random_idx = np.random.choice(idx, 1000, replace = False)
y_pred = np.array(model_train(  [X_test[random_idx], np.zeros(shape = X_test[random_idx].shape[0]) ] ))
y_test = y_test[random_idx]
X_test = X_test[random_idx]

y_pred_arg = np.argmax(y_pred[2, :, :], axis = 1)
y_pred_mean = y_pred[0, :, :][:, y_pred_arg][:, 0]
y_pred_std = np.sqrt(np.log(y_pred[1, :, :][:, y_pred_arg][:, 0]))

####################################
# Transform back to unscaled space #
####################################

# Weird because I'm pretty sure none of these get plotted??

y_pred_3means = preproc_y.inverse_transform(y_pred[0, :, :])
y_pred_3std = preproc_y.inverse_transform( np.sqrt(np.log(y_pred[1, :, :])  ))
y_pred_3weights = y_pred[2, :, :]
y_test_all = preproc_y.inverse_transform(y_test)

#####################
# Save outcomes (?) #
#####################

# We don't always have `label_test` anymore with this new data, I guess?

predstdweights = np.array([y_pred_3means, y_pred_3std, y_pred_3weights])
#truelabel = np.array([y_test_all[:, 0], label_test])

np.save(save_mod+'test_true', predstdweights )
#np.save(save_mod+'test_pred', truelabel )

########
# Plot #
########

ifPlotWeighted = True
y_pred_mean_best = y_pred_mean
y_pred_std_best = y_pred_std

fig, ax = plt.subplots(figsize=(10, 10,)) # How does this work (or conflict) with plt.figure() down below?
if ifPlotWeighted:

    colorstring = ['b', 'r', 'g', 'k', 'orange']
    surveystring = ['SDSS', 'VIPERS', 'DEEP2']

    plt.figure(22, figsize=(10, 10,))

    C = 0.05
    z_t = np.array([0, 3]) # Used to be 1
    z_tp = z_t + C*(1+z_t)
    z_tm = z_t - C*(1+z_t)

    ax.plot(z_t, z_t, 'k')
    ax.plot(z_t, z_tp, 'k-.')
    ax.plot(z_t, z_tm, 'k-.')

    # This section used to be in a for loop, for when we had label_test
    offset = 0.0
    
    print(preproc_y.inverse_transform(y_test)[:, 0].shape)
    print(preproc_y.inverse_transform(y_pred_mean_best.reshape(-1, 1))[:, 0].shape)
    
    ax.errorbar(preproc_y.inverse_transform(y_test)[:, 0], offset + preproc_y.inverse_transform(y_pred_mean_best.reshape(-1, 1))[:, 0], yerr= preproc_y.inverse_transform(y_pred_std_best.reshape(-1, 1))[:, 0], fmt = 'o', marker=None, ms = 4, alpha = 0.3, label = 'Training: Synthetic, Testing: '+sim, c = 'k')
# Cosmetics
ax.set_ylabel(r'$z_{phot}$', fontsize=25)
ax.set_xlabel(r'$z_{spec}$', fontsize=25)
ax.set_xlim(0.0, 3)
ax.set_ylim(0.0, 3)
plt.tight_layout()

fake_lines = [ax.plot([], [], c = "black", linestyle = '-')[0] for i in range(0,len(param_labels))]
ax.legend(handles = fake_lines, labels = param_labels, loc = "upper left")
ax.set_title(sim)

if prtb:
    ax.set_title("Perturbed")

if sim2 is None:
    fig.savefig("training_plots/" + sim + "/" + sim + "_phoz_ntrain" + str(num_train) + "_ntest" + str(num_test) + "_nepochs" + str(n_epochs) + "_D" + str(D) + "_K" + str(K) + "_lr" + str(learning_rate) + "_dr" + str(decay_rate) + "_bs" + str(batch_size) + "_re" + str(resampleTrain) + "_nbins" + str(n_bins) + "_spb" + str(select_per_bin) + ".png")
else:
    print("sim2 is not None")
    fig.savefig("training_plots/" + sim2 + "/" + sim + "_" + sim2 + suffix + "_phoz_ntrain" + str(num_train) + "_ntest" + str(num_test) + "_nepochs" + str(n_epochs) + "_D" + str(D) + "_K" + str(K) + "_lr" + str(learning_rate) + "_dr" + str(decay_rate) + "_bs" + str(batch_size) + "_re" + str(resampleTrain) + "_nbins" + str(n_bins) + "_spb" + str(select_per_bin) + ".png")

###########################
# Plot validation metrics #
###########################

#ncol = 3 # Used to need this for multiple perturbations
sigmaNMAD_array, outFr_array, bins = help_funcs.validate(y_test.T[0], y_pred_mean)

for metric, metric_name in zip([sigmaNMAD_array, outFr_array], ["sigma", "outFrac"]):

    fig, ax = help_funcs.plot_metric(metric, metric_name, approach = None, fig = None, ax = None, label = sim, color = "black", linestyle = '-')
    if sim2 is None:
        minimal_param_labels = ["sim: " + sim, "n epochs: " + str(n_epochs), "D: " + str(D), "K: " + str(K), "learning rate: " + str(learning_rate), "decay rate: " + str(decay_rate), "batch size: " + str(batch_size)]
    else:
        minimal_param_labels = ["sim1: " + sim, "sim2: " + sim2, "n epochs: " + str(n_epochs), "D: " + str(D), "K: " + str(K), "learning rate: " + str(learning_rate), "decay rate: " + str(decay_rate), "batch size: " + str(batch_size)]
        
    leg1 = ax.legend(loc = "upper right")
    fake_lines = [ax.plot([], [], c = "black", linestyle = '-')[0] for i in range(0,len(minimal_param_labels))]
    ax.legend(handles = fake_lines, labels = minimal_param_labels, loc = "upper left")
    ax.add_artist(leg1)
    ax.set_title(sim)

    if sim2 is None:
        print("sim2 is none")
        plt.savefig("training_plots/" + sim + "/" + sim + "_" + metric_name + "_ntrain" + str(num_train) + "_ntest" + str(num_test) + "_nepochs" + str(n_epochs) + "_D" + str(D) + "_K" + str(K) + "_lr" + str(learning_rate) + "_dr" + str(decay_rate) + "_bs" + str(batch_size) + "_re" + str(resampleTrain) + "_nbins" + str(n_bins) + "_spb" + str(select_per_bin) + ".png")
    else:
        print("sim2 is not None")
        plt.savefig("training_plots/" + sim2 + "/" + sim + "_" + sim2 + suffix + "_" + metric_name + "_ntrain" + str(num_train) + "_ntest" + str(num_test) + "_nepochs" + str(n_epochs) + "_D" + str(D) + "_K" + str(K) + "_lr" + str(learning_rate) + "_dr" + str(decay_rate) + "_bs" + str(batch_size) + "_re" + str(resampleTrain) + "_nbins" + str(n_bins) + "_spb" + str(select_per_bin) + ".png")
