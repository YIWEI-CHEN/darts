#!/usr/bin/env bash

DATE=`date +%m%d`
SEED=1

#LOSS="cce"
#EXP_PATH="exp/cifar10_${LOSS}_no_noise_gpu"
#GPU="1"
#python train_search.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --save ${EXP_PATH} --seed ${SEED} --train_portion 0.9

#LOSS="rll"
#EXP_PATH="exp/cifar10_${LOSS}_uniform_07_gpu"
#GPU="2"
#python train_search.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --save ${EXP_PATH} --seed ${SEED} --train_portion 0.9 \
#    --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS}

#LOSS="rll"
#ALPHA=0.01
#ETA=0.4
#EXP_PATH="exp/cifar10_${LOSS}${ALPHA}_uniform_${ETA}_gpu"
#GPU="2"
#python train_search.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --save ${EXP_PATH} --seed ${SEED} --train_portion 0.9 \
#    --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --alpha ${ALPHA} --unrolled

#LOSS="cce"
#ETA=0.6
#EXP_PATH="exp/cifar10_seed${SEED}_${LOSS}_uniform_${ETA}_gpu"
#GPU="3"
#python train_search.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --save ${EXP_PATH} --seed ${SEED} --train_portion 0.9 \
#    --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --unrolled


#LOSS="rll"
#ALPHA=0.01
#ETA=0.6
#EXP_PATH="exp/unif${ETA}_cifar10_seed${SEED}_${LOSS}${ALPHA}_gpu"
#GPU="1"
#python train_search.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --save ${EXP_PATH} --seed ${SEED} --train_portion 0.9 \
#    --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --unrolled

#LOSS="rll"
#ALPHA=0.01
#ETA=0.4
#EXP_PATH="exp/hier${ETA}_cifar100_seed${SEED}_${LOSS}${ALPHA}_gpu"
#GPU="2"
#python train_search.py --data cifar100 --batch_size 64 --gpu ${GPU} \
#    --dataset cifar100 --save ${EXP_PATH} --seed ${SEED} --train_portion 0.9 \
#    --corruption_prob ${ETA} --corruption_type hierarchical --gold_fraction 0 --loss_func ${LOSS} \
#    --alpha ${ALPHA} --unrolled

LOSS="cce"
ETA=0.4
EXP_PATH="exp/hier${ETA}_cifar100_seed${SEED}_${LOSS}_gpu"
GPU="3"
python train_search.py --data cifar100 --batch_size 64 --gpu ${GPU} \
    --dataset cifar100 --save ${EXP_PATH} --seed ${SEED} --train_portion 0.9 \
    --corruption_prob ${ETA} --corruption_type hierarchical --gold_fraction 0 --loss_func ${LOSS} \
    --unrolled