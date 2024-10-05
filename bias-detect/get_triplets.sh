#!/bin/bash

eps_array=(0.0003 0.0006 0.0009 0.002 0.004)
cutoff_array=(2.5 3.0 3.5 4.0 4.5 5.0)
output_directory=''

for eps in "${eps_array[@]}"
do
    for cutoff in "${cutoff_array[@]}"
    do 
    python get_triplets.py --epsilon $eps --proportion_cut_off $cutoff --output_dir $output_directory
    done
done