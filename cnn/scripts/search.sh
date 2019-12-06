#!/usr/bin/env bash

DATE=`date +%m%d`
GPU=1
SEED=10
EXP_PATH="exp/single_gpu/cifar10_seed${SEED}_gpu${GPU}"
python train_search.py --data /home/yiwei/cifar10 --batch_size 64 --gpu ${GPU} \
    --save ${EXP_PATH} --seed ${SEED} --train_portion 0.9 \
    --unrolled
