#!/usr/bin/env bash

GPU="3"
BATCH_SIZE=96
SEED=100
EXP_PATH="exp/single/batch_size${BATCH_SIZE}_gpu${GPU}"
EPOCH=400
ARCH="DARTS"

python train.py --data /home/yiwei/cifar10 --batch_size ${BATCH_SIZE} --gpu ${GPU} \
    --save ${EXP_PATH} --seed ${SEED} --cutout --auxiliary --exec_script scripts/single_eval.sh \
    --workers 0 --epochs ${EPOCH} --arch ${ARCH}