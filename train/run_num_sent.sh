#!/bin/bash

epochs=(1)
file='/supplementary/bias-detect/results/biased_token/biased_tokens_0.004_5.0.csv'

for ((n=0; n<10; n++))
do
    config="epoch_num_sent$n"
    for epoch in ${epochs[@]}
    do
                
        for ((ns=1; ns<6; ns++))
        do
                cd /supplementary/train
                python retrain.py \
                --config $config \
                --input_path $file \
                --epochs $epoch \
                --num_sent $ns
        done

    done
    echo "Training finished!"

done 