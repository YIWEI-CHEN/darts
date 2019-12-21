#!/usr/bin/env bash

GPU="0,1,2,3"
BATCH_SIZE=384
SEED=100
PORT=50017
EXP_PATH="exp/distributed/batch_size${BATCH_SIZE}_gpu${GPU}"
EPOCH=400
ARCH="DARTS"

python dist_train.py --data /home/yiwei/cifar10 --batch_size ${BATCH_SIZE} --gpu ${GPU} \
    --save ${EXP_PATH} --seed ${SEED} --cutout --auxiliary --exec_script scripts/eval.sh \
    --world-size 1 --rank 0 --workers 0 --epochs ${EPOCH} --arch ${ARCH} \
    --dist-url "tcp://datalab.cse.tamu.edu:${PORT}"