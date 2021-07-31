## Aurora's photo-z work

See additional details at https://www.overleaf.com/5899933454qzdcbprzxhfk

Tested on Python 3.7

Packages: 

1. tensorflow 2.3.1 (pip3 install tensorflow==2.3.1)
2. scikit-learn==0.21.3 (pip3 install scikit-learn==0.21.3)

## What's important in this repo?
- `training_plots/`: pngs from training our own neural net (a gaussian mixed density network) on synthetic data
- `testing_plots/`: pngs from testing a previously trained neural net on perturbed spectroscopic data
- `train_ALL_the_all_the_things.py`: a python script for training our own neural net. Can perform either in-set validation with purely synthetic data, or out-of-set validation on DES + IRAC using data from lindsey. To activate the latter option, run with `-use_lindseys_test True`. If you use Lindsey's data for validation, you can also choose whether to injuct random noise into the training data. Activate this with `-use_injected_noise`
- `run_training_ALL.sh`: a bash script to execute `train_ALL_the_all_the_things.py`. Instructions for executing the bash script from a terminal are written (commented) at the top of the file.
- `help_train.py`: a python script holding helper functions for `train_ALL_the_all_the_things.py`
- `help_funcs.py`: a python script holding helper functions used in perturbing test data ("Level 1" in the overleaf doc). It is called in most of the `ZPred` notebooks.

## What about all those jupyter notebooks?
These notebooks were mostly used for experimentation. For training our Gaussian MDN, all the relevant updates have been added to `train_ALL_the_all_the_things.py`. However, all of the `ZPred_` notebooks are stand-alone: they are not backed up in a python script. I believe the most up-to-date notebook is `Zpred_net_err-squish.ipynb`. Note that this depends on helper functions in `help_funcs.py`.
Meanwhile, the experimental notebooks are:
- `mdn_train_noise_wip.ipynb`: experiments for injecting noise
- `mdn_train_zeropt.ipynb`: experiments for using Lindsey's data for validation
- `galaxy_correspondence.ipynb`: experiments to ensure there was 1-1 correspondence between galaxies in DES and IRAC, and DES and WISE. This was necessary to use Lindsey's data.
- `deephyper_tut.ipynb`: experiments with using deephyper. Note that, to actually run deephyper, you need a terminal or a bash script. This notebook was just to check out some of the details of the python scripts in the tutorial.
