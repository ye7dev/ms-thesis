#!/bin/bash

file='supplementary/bias-detect/results/biased_token/biased_tokens_0.004_5.0.csv'
epochs=(1 2 3 4 5)
batch_size=(1)

for ((n=0; n<10; n++))
do
    config="filter_epoch_batch${n}"

    # training
    for epoch in ${epochs[@]}
    do
        for bsize in ${batch_size[@]}
        do
            echo "Current epoch: $epoch, batch size: $bsize"

            cd /supplymentary/train
            python retrain.py \
                --config $config \
                --input_path $file \
                --epochs $epoch \
                --batch_size $bsize
        done 

    done
        echo "Training finished!"

done