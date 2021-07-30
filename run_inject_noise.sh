#!/bin/bash

# To run: open terminal
# make sure you're in correct conda env
# then execute:
# chmod +x run_inject_noise.sh
# ./run_inject_noise.sh

echo "Hello World!"

python inject_noise.py -sim 'des' -sim2 'irac' -D2 4 -use_lindseys_test True -ntrain_points 200000 -std 0.05 -factor 10 | tee inject_noise_des_irac_100000_std_005.txt

# Go ahead and just do the training too

nepochs=100
this_sim='irac'

python train_ALL_the_all_the_things.py -sim 'des' -sim2 $this_sim -nepochs $nepochs -lr 1e-5 -D2 4 -use_lindseys_test True -std 0.05 -use_injected_noise | tee training_plots/$this_sim/output_combo_nepochs_100_noisy_std_005.txt

#nepochs=20
#this_sim='irac'

#python train_ALL_the_all_the_things.py -sim 'des' -sim2 $this_sim -nepochs $nepochs -lr 1e-5 -D2 4 -use_lindseys_test True -std 0.2 -use_injected_noise | tee training_plots/$this_sim/output_combo_nepochs_20_noisy_std_02.txt