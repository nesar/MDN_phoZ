#!/bin/bash 

# To run: open terminal 
# make sure you're in correct conda env
# then execute:
# chmod +x run_training_ALL.sh
# ./run_training_ALL.sh

echo "Hello World!"

# Each sim
# Copy this: python train_ALL_the_all_the_things.py -sim 'jwst' -ntrain 200000 -ntest 20000 -nepochs 20 -K 3 -lr 1e-4 -dr 1e-2 -bs 256 -re False -nbins 200 -spb 400 -prtb False -rm_band False
#python train_ALL_the_all_the_things.py -sim 'des' -sim2 'irac' | tee training_plots/irac/output_jwst.txt

nepochs=100
#this_sim='irac'

#python train_ALL_the_all_the_things.py -sim 'des' -sim2 $this_sim -nepochs $nepochs -lr 1e-5 -D2 4 -use_lindseys_test True -std 0.2 -use_injected_noise | tee training_plots/$this_sim/output_combo_nepochs_20_noisy_std_02.txt

for this_sim in 'jwst' 'lsst' 'sdss' 'spherex' 'pau'
do
    python train_ALL_the_all_the_things.py -sim $this_sim -nepochs $nepochs| tee training_plots/$this_sim/output_$this_sim_nepochs_$nepochs.txt
done
