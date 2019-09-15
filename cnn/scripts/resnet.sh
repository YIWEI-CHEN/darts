#!/usr/bin/env bash

DATE=`date +%m%d`

#LOSS="cce"
#ARCH="resnet"
#GPU=0
#EXP_PATH="exp/cifar10_clean_${LOSS}_${ARCH}_gpu${GPU}"
#EPOCHS=600
#
#python resnet.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 \
#    --dataset cifar10 --gold_fraction 1.0 --loss_func ${LOSS}

#LOSS="rll"
#ARCH="resnet"
#GPU=0
#EXP_PATH="exp/resnet/cifar10_clean_${LOSS}_${ARCH}_gpu${GPU}"
#EPOCHS=600
#
#python resnet.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 \
#    --dataset cifar10 --gold_fraction 1.0 --loss_func ${LOSS}

#LOSS="cce"
#ARCH="resnet"
#GPU=1
#EXP_PATH="exp/resnet/cifar10_uniform_07_${LOSS}_${ARCH}_gpu${GPU}"
#EPOCHS=600
#
#python resnet.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 \
#    --dataset cifar10 --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS}

#LOSS="rll"
#ARCH="resnet"
#GPU=2
#EXP_PATH="exp/resnet/cifar10_uniform_07_${LOSS}_${ARCH}_gpu${GPU}"
#EPOCHS=600
#
#python resnet.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 \
#    --dataset cifar10 --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS}

#LOSS="forward_gold"
#ARCH="resnet"
#GPU=2
#EXP_PATH="exp/resnet/cifar10_uniform_07_${LOSS}_${ARCH}_gpu${GPU}"
#EPOCHS=600
#
#python resnet.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 \
#    --dataset cifar10 --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS}

LOSS="rll"
ARCH="resnet50"
GPU=3
ALPHA=0.01
EPOCHS=600
ETA=0.6
EXP_PATH="exp/resnet/cifar10_uniform${ETA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"

python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --cutout \
    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9