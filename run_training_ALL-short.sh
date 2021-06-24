#!/bin/bash 

# To run: open terminal 
# make sure you're in correct conda env
# then execute:
# chmod +x run_training_ALL.sh
# ./run_training_ALL.sh

echo "Hello World!"

# Each sim
# Copy this: python train_ALL_the_all_the_things.py -sim 'jwst' -ntrain 200000 -ntest 20000 -nepochs 20 -K 3 -lr 1e-4 -dr 1e-2 -bs 256 -re False -nbins 200 -spb 400 -prtb False -rm_band False

nepochs=5
#python train_ALL_the_all_the_things.py -sim 'des' -sim2 'irac' -D2 4 -use_lindseys_test True -nepochs $nepochs -lr 1e-7 | tee #training_plots/'irac'/output_combo_lindsey_nepochs_$nepochs.txt

for this_sim in 'irac' #'wise' # Turns out I don't need a multivariable loop, but I still wish I knew how to do it
do
    python train_ALL_the_all_the_things.py -sim 'des' -sim2 $this_sim -nepochs $nepochs -D2 4 -use_lindseys_test True | tee training_plots/$this_sim/output_combo_nepochs_$nepochs.txt
done

#this_sim='des'
#python train_ALL_the_all_the_things.py -sim $this_sim -nepochs $nepochs | tee training_plots/$this_sim/output_nepochs_$nepochs_$this_sim.txt

#for this_sim in 'des' 'jwst' 'lsst' 'pau' 'sdss' 'spherex' # Turns out I don't need a multivariable loop, but I still wish I knew how to do it
#do
#    python train_ALL_the_all_the_things.py -sim $this_sim -nepochs $nepochs | tee training_plots/$this_sim/output_nepochs_$nepochs_$this_sim.txt
#done
