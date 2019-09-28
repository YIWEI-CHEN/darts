#!/usr/bin/env bash

SEED=1
GPU=2
DATA="cifar10"

#LOSS="rll"
#ARCH="DARTS_V2"
#ALPHA=0.01
#EPOCHS=350
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/F1_unif${ETA}_${DATA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

LOSS="rll"
ARCH="RLL001_UNIFORM_06"
ALPHA=0.01
EPOCHS=350
LAYERS=8
ETA=0.6
EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/F2_unif${ETA}_${DATA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"

python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}


#LOSS="rll"
#ARCH="CCE_UNIFORM_06"
#ALPHA=0.01
#EPOCHS=350
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/F3_unif${ETA}_${DATA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

#LOSS="cce"
#ARCH="DARTS_V2"
#EPOCHS=350
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/F1_unif${ETA}_${DATA}_${LOSS}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS}

#LOSS="cce"
#ARCH="RLL001_UNIFORM_06"
#EPOCHS=350
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/F2_unif${ETA}_${DATA}_${LOSS}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS}

#LOSS="cce"
#ARCH="CCE_UNIFORM_06"
#EPOCHS=350
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/F3_unif${ETA}_${DATA}_${LOSS}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS}


#LOSS="rll"
#ARCH="DARTS_V2"
#ALPHA=0.01
#EPOCHS=350
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/F1_clean_${DATA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --corruption_prob ${ETA} --corruption_type unif  --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS} --clean_train


#LOSS="rll"
#ARCH="RLL001_UNIFORM_06"
#ALPHA=0.01
#EPOCHS=350
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/F2_clean_${DATA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --corruption_prob ${ETA} --corruption_type unif  --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS} --clean_train

#LOSS="rll"
#ARCH="CCE_UNIFORM_06"
#ALPHA=0.01
#EPOCHS=350
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/F3_clean_${DATA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --corruption_prob ${ETA} --corruption_type unif  --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS} --clean_train