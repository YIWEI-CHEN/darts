#!/usr/bin/env bash

SEED=1
DATA="cifar10"

#LOSS="rll"
#ALPHA=0.01
#EPOCHS=1000
#LAYERS=8
#ETA=0.6
#GPU=0
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_arch/unif${ETA}_${DATA}_${LOSS}${ALPHA}_gpu${GPU}"
#
#python rank_loss_with_random_arch.py --data "/home/yiwei/${DATA}" --batch_size 1024 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

LOSS="cce"
EPOCHS=1000
LAYERS=8
ETA=0.6
GPU=1
EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_arch/unif${ETA}_${DATA}_${LOSS}${ALPHA}_gpu${GPU}"

python rank_loss_with_random_arch.py --data "/home/yiwei/${DATA}" --batch_size 1024 --gpu ${GPU} \
    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
    --train_portion 0.9 --layers ${LAYERS}