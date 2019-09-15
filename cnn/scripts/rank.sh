#!/usr/bin/env bash

#LOSS="rll"
#ARCH="RLL001_UNIFORM_06"
#GPU=2
#ALPHA=0.01
#EPOCHS=500
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank/A1_unif${ETA}_cifar10_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}


LOSS="cce"
ARCH="RLL001_UNIFORM_06"
GPU=2
EPOCHS=500
LAYERS=8
ETA=0.6
EXP_PATH="exp/MyDARTS/rank/A1_unif${ETA}_cifar10_${LOSS}_${ARCH}_gpu${GPU}"

python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --auxiliary --cutout \
    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS}