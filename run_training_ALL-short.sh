#!/bin/bash 

# To run: open terminal 
# make sure you're in correct conda env
# then execute:
# chmod +x run_training_ALL.sh
# ./run_training_ALL.sh

echo "Hello World!"

# Each sim
# Copy this: python train_ALL_the_all_the_things.py -sim 'jwst' -ntrain 200000 -ntest 20000 -nepochs 20 -K 3 -lr 1e-4 -dr 1e-2 -bs 256 -re False -nbins 200 -spb 400 -prtb False -rm_band False
python train_ALL_the_all_the_things.py -sim 'jwst' | tee training_plots/output_jwst.txt

for this_sim in 'spherex' 'lsst' 'sdss' 'wise' 'pau'
do
    python train_ALL_the_all_the_things.py -sim $this_sim -nepochs 20 | tee training_plots/$this_sim/output_neposchs_20_$this_sim.txt
done