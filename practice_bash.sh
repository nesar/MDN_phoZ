#!/bin/bash 

echo "Hello World!"

for this_sim in 'irac' 'des' 'irac' 'jwst' 'lsst' 'pau' 'sdss' 'wise'
do
    echo "$this_sim plus some more words" | tee training_plots/practice_$this_sim.txt
done