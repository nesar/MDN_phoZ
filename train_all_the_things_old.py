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

########################
# Load hyperparameters #
########################

import argparse

# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument("-ntrain", dest='num_train', default = 200000, type = int, required = False)
parser.add_argument("-ntest", dest='num_test', default = 20000, type = int, required = False)
parser.add_argument("-nepochs", dest='n_epochs', default = 100, type = int, required = False)
parser.add_argument("-D", dest='D', default = 5, help = "number of features", type = int, required = False)
parser.add_argument("-K", dest='K', default = 3, help = "number of mixture components", type = int, required = False)
parser.add_argument("-lr", dest='learning_rate', default = 1e-4, type = float, required = False)
parser.add_argument("-dr", dest='decay_rate', default = 1e-3, type = float, required = False)
parser.add_argument("-bs", dest='batch_size', default = 1024, type = int, required = False)
# Resampling parameters
parser.add_argument("-re", dest = "resampleTrain", default = True, type = bool, required = False)
parser.add_argument("-nbins", dest = "nbins", default = 200, type = int, required = False)
parser.add_argument("-spb", dest = "select_per_bin", default = 400, type = int, required = False)
parser.add_argument("-prtb", dest = "prtb", default = False, type = bool, required = False)

# This is probably a pretty lazy solution...
args = parser.parse_args()
print(args)
num_train = args.num_train
num_test = args.num_test
n_epochs = args.n_epochs
D = args.D
K = args.K
learning_rate = args.learning_rate
decay_rate = args.decay_rate
batch_size = args.batch_size
resampleTrain = args.resampleTrain
n_bins = args.nbins
select_per_bin = args.select_per_bin
prtb = args.prtb

#######################
# Constant parameters #
#######################

Trainset = ['FSPSlin', 'FSPSlog', 'FSPSall', 'OBS', 'UM', 'BP', 'UMnew'][6] # Soon there will be more!
Testset = ['FSPSlin', 'FSPSlog', 'FSPSall', 'OBS', 'UM', 'BP', 'UMnew', 'OBSuq'][7] # Test on the same things we tested before (SDSS)

# This actually saves/defines the hyperparameters
save_mod = 'saved_hubs/tf2models/'+'Train_'+Trainset+'_lr_'+str(learning_rate)+'_dr'+str(decay_rate)+'_ne'+str(n_epochs)+'_k'+str(K)+'_nt'+str(num_train)

#############################################
# Load in training/testing data, shuffle it #
#############################################

X_train, y_train, _, _ = help_train.loadTrainTest_july(dirIn = '/data/a/cpac/nramachandra/Projects/phoZ/Data/fromGalaxev/photozs/datasets/data_july_2020/')
X_test, y_test, X_err, label_test = help_train.loadTest(Testset, dirIn = '/data/a/cpac/aurora/MDN_phoZ/Data/fromGalaxev/photozs/datasets/data_feb_2021/') # data_feb_2021

X_train, y_train, X_trainShuffleOrder = help_train.shuffle(X_train, y_train) # literally just shuffle the data

#################
# Trim the data #
#################

min_col = [-0.09145837, -0.05327791, -0.02479261, -0.10519464]
max_col = [ 3.825315,   2.8303378,  1.6937237,  1.5019817]
min_mag = 12
max_mag = 23
min_z = 0.0 #np.min(y_train)
max_z = 1.1 #np.max(y_train)
mins_and_maxs = [min_col, max_col, min_mag, max_mag, min_z, max_z]
X_test, y_test, label_test, mask_cond = help_train.minmax_cutsOBSarr(X_test, y_test, label_test, mins_and_maxs)

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

#####################################
# Plot histograms of train and test #
#####################################

help_train.print_limits(X_train, y_train)
help_train.print_limits(X_test, y_test)

fig, ax = plt.subplots(figsize=(10, 10,)) # plt.figure(23)
ax.hist(y_train, density=True, bins = 250, histtype='step', label='train')
ax.hist(y_test, density=True, bins = 250, histtype='step', label='test')

leg1 = ax.legend(fontsize = 'xx-large', markerscale=1., numpoints=2)
fake_lines = [ax.plot([], [], c = "black")[0] for i in range(0,10)]
param_labels = ["num_train: " + str(num_train), "num_test: " + str(num_test), "n_epochs: " + str(n_epochs), "D: " + str(D), "K: " + str(K), "learning_rate: " + str(learning_rate) + "decay_rate: " + str(decay_rate), "batch_size: " + str(batch_size), "n_bins: " + str(n_bins), "select_per_bin: " + str(select_per_bin)]
ax.legend(handles = fake_lines, labels = param_labels, loc = "upper right")
ax.add_artist(leg1)

plt.savefig("training_plots/precision_ntrain" + str(num_train) + "_ntest" + str(num_test) + "_nepochs" + str(n_epochs) + "_D" + str(D) + "_K" + str(K) + "_lr" + str(learning_rate) + "_dr" + str(decay_rate) + "_bs" + str(batch_size) + "_re" + str(resampleTrain) + "_nbins" + str(n_bins) + "_spb" + str(select_per_bin) + ".png")

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

##########################
# Perturb if you want to #
##########################

if prtb:
    band_n = 0 # remove the u band
    approach = 'e'
    prtb_X_train = help_funcs.perturb(X_train, band_n, approach = approach)
    prtb_X_test = help_funcs.perturb(X_test, band_n, approach = approach)
    suffix = '_perturbed'
    my_X_train = prtb_X_train
    my_X_test = prtb_X_test
else:
    suffix = ""
    my_X_train = X_train
    my_X_test = X_test

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
    
    # Fancy legend
    fake_lines = [plt.plot([], [], c = "black")[0] for i in range(0,10)]
    param_labels = ["num_train: " + str(num_train), "num_test: " + str(num_test), "n_epochs: " + str(n_epochs), "D: " + str(D), "K: " + str(K), "learning_rate: " + str(learning_rate), "decay_rate: " + str(decay_rate), "batch_size: " + str(batch_size), "n_bins: " + str(n_bins), "select_per_bin: " + str(select_per_bin)]
    plt.legend(handles = fake_lines, labels = param_labels, loc = "upper right")

    plt.savefig("training_plots/loss_epochs_ntrain" + str(num_train) + "_ntest" + str(num_test) + "_nepochs" + str(n_epochs) + "_D" + str(D) + "_K" + str(K) + "_lr" + str(learning_rate) + "_dr" + str(decay_rate) + "_bs" + str(batch_size) + "_re" + str(resampleTrain) + "_nbins" + str(n_bins) + "_spb" + str(select_per_bin) + ".png")

model_train.load_weights(save_mod + '.h5') # Previously, this exact line was loading the weights of (from?) the mixed density model

y_pred = np.array(model_train(  [X_test, np.zeros(shape = X_test.shape[0]) ] ))
y_pred_arg = np.argmax(y_pred[2, :, :], axis = 1)
y_pred_mean = y_pred[0, :, :][:, y_pred_arg][:, 0]
y_pred_std = np.sqrt(np.log(y_pred[1, :, :][:, y_pred_arg][:, 0]))

####################################
# Transform back to unscaled space #
####################################

y_pred_3means = preproc_y.inverse_transform(y_pred[0, :, :])
y_pred_3std = preproc_y.inverse_transform( np.sqrt(np.log(y_pred[1, :, :])  ))
y_pred_3weights = y_pred[2, :, :]
y_test_all = preproc_y.inverse_transform(y_test)

#####################
# Save outcomes (?) #
#####################

predstdweights = np.array([y_pred_3means, y_pred_3std, y_pred_3weights])
truelabel = np.array([y_test_all[:, 0], label_test])

np.save(save_mod+'test_true', predstdweights )
np.save(save_mod+'test_pred', truelabel )

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
    z_t = np.array([0, 1])
    z_tp = z_t + C*(1+z_t)
    z_tm = z_t - C*(1+z_t)

    ax.plot(z_t, z_t, 'k')
    ax.plot(z_t, z_tp, 'k-.')
    ax.plot(z_t, z_tm, 'k-.')

    for label_ind in [0, 1, 2]:
        surveyindx = np.where(label_test == label_ind)
        offset = 0.0
        
        ax.errorbar(preproc_y.inverse_transform(y_test)[surveyindx][:, 0], offset + preproc_y.inverse_transform(y_pred_mean_best.reshape(-1, 1))[surveyindx][:, 0], yerr= preproc_y.inverse_transform(y_pred_std_best.reshape(-1, 1))[surveyindx][:, 0], fmt = 'o', marker=None, ms = 4, alpha = 0.3, label = 'Training: Synthetic, Testing: '+surveystring[label_ind], c = colorstring[label_ind])

# Cosmetics
ax.set_ylabel(r'$z_{phot}$', fontsize=25)
ax.set_xlabel(r'$z_{spec}$', fontsize=25)
ax.set_xlim(0.0, 1)
ax.set_ylim(0.0, 1)
plt.tight_layout()
leg1 = ax.legend(fontsize = 'xx-large', markerscale=1., numpoints=2)
fake_lines = [ax.plot([], [], c = "black")[0] for i in range(0,10)]
param_labels = ["num_train: " + str(num_train), "num_test: " + str(num_test), "n_epochs: " + str(n_epochs), "D: " + str(D), "K: " + str(K), "learning_rate: " + str(learning_rate), "decay_rate: " + str(decay_rate), "batch_size: " + str(batch_size), "n_bins: " + str(n_bins), "select_per_bin: " + str(select_per_bin)]
ax.legend(handles = fake_lines, labels = param_labels, loc = "upper left")
ax.add_artist(leg1)
fig.savefig("training_plots/phoz_ntrain" + str(num_train) + "_ntest" + str(num_test) + "_nepochs" + str(n_epochs) + "_D" + str(D) + "_K" + str(K) + "_lr" + str(learning_rate) + "_dr" + str(decay_rate) + "_bs" + str(batch_size) + "_re" + str(resampleTrain) + "_nbins" + str(n_bins) + "_spb" + str(select_per_bin) + ".png")
fig.show()
