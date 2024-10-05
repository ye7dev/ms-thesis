#!/bin/bash

token_dir="/supplymentary/bias-detect/results/biased_token"
res_dir="/supplymentary/train/results"

for ((n=0; n<10; n++))
do      
        config="eps_cutoff${n}"

        # training 
        for file in "$token_dir"/*; do
                if [ -f "$file" ]; then
                filename=$(basename "$file")

                echo "Processing file: $filename"

                # {eps}_{cutoff}.pth
                cd /supplymentary/train
                python retrain.py \
                        --config $config \
                        --input_path $file 

                fi 
        done 
        echo "Training finished!"

done 