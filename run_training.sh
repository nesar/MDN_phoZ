#!/bin/bash 

echo "Hello World!"

# Start with lower epoch nums, decay of 0
# Change batch sizes
#python train_all_the_things.py -nepochs 50 -dr 0 -bs 32| tee training_plots/output_nepochs50_dr0_bs32.txt
#python train_all_the_things.py -nepochs 50 -dr 0 -bs 64| tee training_plots/output_nepochs50_dr0_bs64.txt
#python train_all_the_things.py -nepochs 50 -dr 0 -bs 128| tee training_plots/output_nepochs50_dr0_bs128.txt
#python train_all_the_things.py -nepochs 50 -dr 0 -bs 256| tee training_plots/output_nepochs50_dr0_bs256.txt
#python train_all_the_things.py -nepochs 50 -dr 0 -bs 512| tee training_plots/output_nepochs50_dr0_bs512.txt
#python train_all_the_things.py -nepochs 50 -dr 0 -bs 1024| tee training_plots/output_nepochs50_dr0_bs1024.txt
#python train_all_the_things.py -nepochs 50 -dr 0 -bs 2048| tee training_plots/output_nepochs50_dr0_bs2048.txt

# 5/11: batch size of 128 or 256 appears to be best
batch_size = 256
# Change learning rates
# How can I change the batch_size name in the output file too? Also, could I do this with a loop?
#python train_all_the_things.py -lr 1e-2 -nepochs 50 -dr 0 -bs 512 | tee training_plots/output_nepochs50_dr0_bs512_lr1e-2.txt
#python train_all_the_things.py -lr 1e-3 -nepochs 50 -dr 0 -bs 512 | tee training_plots/output_nepochs50_dr0_bs512_lr1e-3.txt
#python train_all_the_things.py -lr 1e-4 -nepochs 50 -dr 0 -bs 512 | tee training_plots/output_nepochs50_dr0_bs512_lr1e-4.txt
#python train_all_the_things.py -lr 1e-5 -nepochs 50 -dr 0 -bs 512 | tee training_plots/output_nepochs50_dr0_bs512_lr1e-5.txt
# 5/12: batch size of 256 seems to do a lot better than 512
# 5/12: higher learning rates seem to be better as well -- 1e-4 or 1e-5 at least

# Change decay rates
#python train_all_the_things.py -lr 1e-4 -nepochs 50 -dr 1e-1 -bs 256 | tee training_plots/output_nepochs50_dr1e-1_bs256_lr1e-4.txt
#python train_all_the_things.py -lr 1e-4 -nepochs 50 -dr 1e-2 -bs 256 | tee training_plots/output_nepochs50_dr1e-2_bs256_lr1e-4.txt
python train_all_the_things.py -lr 1e-4 -nepochs 50 -dr 1e-3 -bs 256 | tee training_plots/output_nepochs50_dr1e-3_bs256_lr1e-4.txt
#python train_all_the_things.py -lr 1e-4 -nepochs 50 -dr 1e-4 -bs 256 | tee training_plots/output_nepochs50_dr1e-4_bs256_lr1e-4.txt
#python train_all_the_things.py -lr 1e-4 -nepochs 50 -dr 1e-5 -bs 256 | tee training_plots/output_nepochs50_dr1e-5_bs256_lr1e-4.txt
# Change epochs
