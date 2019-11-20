#!/usr/bin/env bash

DATE=`date +%m%d`

#LOSS="cce"
#SEED=2019
#ARCH="CCE_UNIFORM_06_CLEAN_VALID_${SEED}"
#GPU=0
#EPOCHS=600
#LAYERS=20
#ETA=0.6
#DATASET="cifar10"
#EXP_PATH="exp/MyDARTS/noisy_retrain/${DATASET}_seed${SEED}_${LOSS}_${ARCH}_gpu${GPU}"
#
#python train.py --data "/home/yiwei/${DATASET}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --dataset ${DATASET} --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0.0 --loss_func ${LOSS} \
#    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS}

#LOSS="cce"
#SEED=1989
#ARCH="CCE_UNIFORM_06_CLEAN_TRAIN_${SEED}"
#GPU=3
#EPOCHS=600
#LAYERS=20
#ETA=0.6
#DATASET="cifar10"
#EXP_PATH="exp/MyDARTS/noisy_retrain/${DATASET}_seed${SEED}_${LOSS}_${ARCH}_gpu${GPU}"
#
#python train.py --data "/home/yiwei/${DATASET}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --dataset ${DATASET} --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0.0 --loss_func ${LOSS} \
#    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS}

#LOSS="cce"
#SEED=2019
#ARCH="CCE_ALL_NOISY_${SEED}"
#GPU=2
#EPOCHS=600
#LAYERS=20
#ETA=0.6
#DATASET="cifar10"
#EXP_PATH="exp/MyDARTS/clean_retrain/${DATASET}_seed${SEED}_${LOSS}_${ARCH}_gpu${GPU}"
#
#python train.py --data "/home/yiwei/${DATASET}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --dataset ${DATASET} --corruption_prob ${ETA} --corruption_type unif --gold_fraction 1.0 --loss_func ${LOSS} \
#    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS} --clean_valid


SEED=1  # change
ARCH="CCE_ALL_CLEAN_${SEED}" #change
GPU=2  # change
LOSS="rll"
DATA_SEED=1
EPOCHS=600
LAYERS=8
ETA=0.6
DATASET="cifar10"
ALPHA=0.01
EXP_PATH="exp/MyDARTS/noisy_retrain_rll0.01/${DATASET}_seed${SEED}_${ARCH}_gpu${GPU}"

python train.py --data "/home/yiwei/${DATASET}" --batch_size 64 --gpu ${GPU} \
    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
    --dataset ${DATASET} --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0.0 --loss_func ${LOSS} \
    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

