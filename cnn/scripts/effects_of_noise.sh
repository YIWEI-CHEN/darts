#!/usr/bin/env bash

DATE=`date +%m%d`

#LOSS="cce"
#SEED=2019
#ARCH="CCE_UNIFORM_06_CLEAN_VALID_${SEED}"
#GPU=2
#EPOCHS=600
#LAYERS=20
#ETA=0.6
#DATASET="cifar10"
#EXP_PATH="exp/MyDARTS/unif_${ETA}_${DATASET}_seed${SEED}_clean_valid_${LOSS}_${ARCH}_gpu${GPU}"
#
#python train.py --data "/home/yiwei/${DATASET}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --dataset ${DATASET} --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS} --clean_valid

LOSS="cce"
SEED=1989
ARCH="CCE_UNIFORM_06_CLEAN_TRAIN_${SEED}"
GPU=3
EPOCHS=600
LAYERS=20
ETA=0.6
DATASET="cifar10"
EXP_PATH="exp/MyDARTS/unif_${ETA}_${DATASET}_seed${SEED}_clean_train_${LOSS}_${ARCH}_gpu${GPU}"

python train.py --data "/home/yiwei/${DATASET}" --batch_size 64 --gpu ${GPU} \
    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
    --dataset ${DATASET} --corruption_prob ${ETA} --corruption_type unif --gold_fraction 1.0 --loss_func ${LOSS} \
    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS}

