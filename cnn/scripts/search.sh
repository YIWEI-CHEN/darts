#!/usr/bin/env bash

DATE=`date +%m%d`

#LOSS="cce"
#EXP_PATH="exp/cifar10_${LOSS}_no_noise_gpu"
#GPU="1"
#python train_search.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --save ${EXP_PATH} --seed 1 --train_portion 0.9

#LOSS="rll"
#EXP_PATH="exp/cifar10_${LOSS}_uniform_07_gpu"
#GPU="2"
#python train_search.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --save ${EXP_PATH} --seed 1 --train_portion 0.9 \
#    --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS}

LOSS="rll"
EXP_PATH="exp/cifar10_${LOSS}_uniform_07_clean_valid_gpu"
GPU="2"
python train_search.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
    --save ${EXP_PATH} --seed 1 --train_portion 0.9 \
    --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} --clean_valid
