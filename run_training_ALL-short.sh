#!/bin/bash 

# To run: open terminal 
# make sure you're in correct conda env
# then execute:
# chmod +x run_training_ALL.sh
# ./run_training_ALL.sh

# For each sim
# Copy this: python train_ALL_the_all_the_things.py -sim 'jwst' -ntrain 200000 -ntest 20000 -nepochs 20 -K 3 -lr 1e-4 -dr 1e-2 -bs 256 -re False -nbins 200 -spb 400 -prtb False -rm_band False

echo "Hello World!"

nepochs=5
#this_sim='irac'

#python train_ALL_the_all_the_things.py -sim 'des' -sim2 $this_sim -nepochs $nepochs -lr 1e-5 -D2 4 -use_lindseys_test True -std 0.2 -use_injected_noise | tee training_plots/$this_sim/output_combo_nepochs_5_noisy_std_02.txt

#python train_ALL_the_all_the_things.py -sim 'des' -sim2 $this_sim -nepochs $nepochs -lr 1e-5 -D2 4 -use_lindseys_test True -std 0.2 -use_injected_noise | tee training_plots/$this_sim/output_combo_nepochs_20_noisy_std_02.txt

#python train_ALL_the_all_the_things.py -sim 'des' -sim2 $this_sim -nepochs $nepochs -D2 4 -use_lindseys_test True | tee training_plots/$this_sim/output_combo_nepochs_20_noisy_False.txt

#for inject_noise in False True
#do
#    echo $inject_noise
#    python train_ALL_the_all_the_things.py -sim 'des' -sim2 $this_sim -nepochs $nepochs -D2 4 -use_lindseys_test true -use_injected_noise $inject_noise | tee training_plots/$this_sim/output_combo_nepochs_5_noisy_$inject_noise.txt
#done

for this_sim in 'des' #'irac' 'jwst' 'lsst' 'pau' 'sdss' 'spherex' 'wise' # Turns out I don't need a multivariable loop, but I still wish I knew how to do it
do
    python train_ALL_the_all_the_things.py -sim $this_sim -nepochs $nepochs | tee training_plots/$this_sim/output_nepochs_$nepochs_$this_sim.txt
done
