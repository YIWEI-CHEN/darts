#!/usr/bin/env bash

DATE=`date +%m%d`
SEED=1

#LOSS="rll"
#ARCH="RLL_UNIFORM_07_EPOCH40"
#GPU=3
#EXP_PATH="exp/cifar10_${LOSS}_${ARCH}_gpu${GPU}"
#EPOCHS=600
#
#python train.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --drop_path_prob 0.3 --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH}

#LOSS="rll"
#ARCH="DARTS_V2"
#GPU=0
#EXP_PATH="exp/cifar10_uniform_07_${LOSS}_${ARCH}_gpu${GPU}"
#EPOCHS=600
#
#python train.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --drop_path_prob 0.3 --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH}

#LOSS="cce"
#ARCH="DARTS_V2"
#GPU=1
#EXP_PATH="exp/cifar10_uniform_07_${LOSS}_${ARCH}_gpu${GPU}"
#EPOCHS=600
#
#python train.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --drop_path_prob 0.3 --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH}

#LOSS="rll"
#ARCH="DARTS_V2"
#GPU=0
#EXP_PATH="exp/cifar10_clean_${LOSS}_${ARCH}_gpu${GPU}"
#EPOCHS=600
#
#python train.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --drop_path_prob 0.3 --auxiliary --cutout \
#    --dataset cifar10 --gold_fraction 1.0 --loss_func ${LOSS} \
#    --arch ${ARCH}

#LOSS="rll"
#ARCH="DARTS_V2"
#GPU=2
#ALPHA=0.000001
#EXP_PATH="exp/cifar10_clean_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#EPOCHS=600
#
#python train.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --drop_path_prob 0.3 --auxiliary --cutout \
#    --dataset cifar10 --gold_fraction 1.0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA}

#LOSS="cce"
#ARCH="DARTS_V2"
#GPU=3
#EXP_PATH="exp/cifar10_clean_${LOSS}_${ARCH}_gpu${GPU}"
#EPOCHS=600
#
#python train.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --drop_path_prob 0.3 --auxiliary --cutout \
#    --dataset cifar10 --gold_fraction 1.0 --loss_func ${LOSS} \
#    --arch ${ARCH}


#LOSS="rll"
#ARCH="DARTS_V2"
#GPU=0
#EXP_PATH="exp/cifar10_uniform_07_${LOSS}_${ARCH}_shallow_gpu${GPU}"
#EPOCHS=600
#
#python train.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --drop_path_prob 0.3 --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --layers 4 --init_channels 16

#LOSS="rll"
#ARCH="DARTS_V2"
#GPU=2
#ALPHA=0.1
#EPOCHS=600
#LAYERS=16
#EXP_PATH="exp/DARTS_V2/cifar10_uniform_07_${LOSS}${ALPHA}_${ARCH}-${LAYERS}_no_aux_cutout_gpu${GPU}"
#
#python train.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --drop_path_prob 0.3 \
#    --dataset cifar10 --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --layers ${LAYERS}

#LOSS="forward_gold"
#ARCH="DARTS_V2"
#GPU=3
#ALPHA=0.1
#EPOCHS=600
#LAYERS=20
#EXP_PATH="exp/DARTS_V2/cifar10_uniform_07_${LOSS}_${ARCH}-${LAYERS}_gpu${GPU}"
#
#python train.py --data cifar10 --batch_size 64 --learning_rate 0.1 --weight_decay 0.0005 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --drop_path_prob 0.3 --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob 0.7 --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --layers ${LAYERS}


#LOSS="rll"
#ARCH="CCE_UNIFORM_06"
#GPU=0
#ALPHA=0.01
#EPOCHS=600
#LAYERS=20
#ETA=0.6
#EXP_PATH="exp/MyDARTS/cifar10_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

#LOSS="cce"
#ARCH="N3"
#GPU=2
#EPOCHS=600
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/cifar10_${LOSS}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS}

#LOSS="rll"
#ARCH="CCE_UNIFORM_04"
#GPU=1
#ALPHA=0.01
#EPOCHS=600
#LAYERS=20
#ETA=0.4
#EXP_PATH="exp/MyDARTS/cifar10_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9

#LOSS="rll"
#ARCH="N3"
#GPU=3
#ALPHA=0.01
#EPOCHS=500
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/cifar10_clean_valid_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

#LOSS="cce"
#ARCH="N3"
#GPU=2
#EPOCHS=600
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/cifar10_${LOSS}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS}

#LOSS="rll"
#ARCH="CHECK_REDUCE_IMPACT"
#GPU=1
#ALPHA=0.01
#EPOCHS=600
#LAYERS=20
#ETA=0.6
#EXP_PATH="exp/handcraft/unif${ETA}_cifar10_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

#LOSS="rll"
##ARCH="CHECK_NORMAL_IMPACT"
##GPU=2
##ALPHA=0.01
##EPOCHS=600
##LAYERS=20
##ETA=0.6
##EXP_PATH="exp/handcraft/unif${ETA}_cifar10_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
##
##python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
##    --epochs ${EPOCHS} --save ${EXP_PATH} --seed 1 --auxiliary --cutout \
##    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
##    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

#LOSS="rll"
#ARCH="DARTS_V2"
#GPU=3
#ALPHA=0.01
#EPOCHS=600
#LAYERS=20
#ETA=0.6
#EXP_PATH="exp/DARTS/unif_${ETA}_cifar10_seed${SEED}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

#LOSS="rll"
#ARCH="RLL001_UNIFORM_06_1989"
#GPU=3
#ALPHA=0.01
#EPOCHS=600
#LAYERS=20
#ETA=0.6
#EXP_PATH="exp/MyDARTS/unif_${ETA}_cifar10_seed${SEED}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

#LOSS="rll"
#ARCH="CCE_UNIFORM_06_SEED_1989"
#GPU=0
#ALPHA=0.01
#EPOCHS=600
#LAYERS=20
#ETA=0.6
#EXP_PATH="exp/MyDARTS/unif_${ETA}_cifar10_seed${SEED}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

#LOSS="rll"
#ARCH="RLL001_UNIFORM_06_2019"
#GPU=1
#ALPHA=0.01
#EPOCHS=600
#LAYERS=20
#ETA=0.6
#EXP_PATH="exp/MyDARTS/unif_${ETA}_cifar10_seed${SEED}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar10 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --dataset cifar10 --corruption_prob ${ETA} --corruption_type unif --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

#LOSS="rll"
#ARCH="RLL001_HIER_06_1"
#GPU=0
#ALPHA=0.01
#EPOCHS=600
#LAYERS=20
#ETA=0.6
#EXP_PATH="exp/MyDARTS/hier_${ETA}_cifar100_seed${SEED}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar100 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --dataset cifar100 --corruption_prob ${ETA} --corruption_type hierarchical --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

#LOSS="rll"
#ARCH="RLL001_HIER_02_1"
#GPU=2
#ALPHA=0.01
#EPOCHS=600
#LAYERS=20
#ETA=0.2
#EXP_PATH="exp/MyDARTS/hier_${ETA}_cifar100_seed${SEED}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar100 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --dataset cifar100 --corruption_prob ${ETA} --corruption_type hierarchical --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

#LOSS="rll"
#ARCH="RLL001_HIER_04_1"
#GPU=0
#ALPHA=0.01
#EPOCHS=600
#LAYERS=20
#ETA=0.4
#EXP_PATH="exp/MyDARTS/hier_${ETA}_cifar100_seed${SEED}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar100 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --dataset cifar100 --corruption_prob ${ETA} --corruption_type hierarchical --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

#LOSS="rll"
#ARCH="CCE_HIER_06_1"
#GPU=1
#ALPHA=0.01
#EPOCHS=600
#LAYERS=20
#ETA=0.6
#EXP_PATH="exp/MyDARTS/hier_${ETA}_cifar100_seed${SEED}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar100 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --dataset cifar100 --corruption_prob ${ETA} --corruption_type hierarchical --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

#LOSS="rll"
#ARCH="CCE_HIER_04_1"
#GPU=1
#ALPHA=0.01
#EPOCHS=600
#LAYERS=20
#ETA=0.4
#EXP_PATH="exp/MyDARTS/hier_${ETA}_cifar100_seed${SEED}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python train.py --data cifar100 --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
#    --dataset cifar100 --corruption_prob ${ETA} --corruption_type hierarchical --gold_fraction 0 --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}

LOSS="rll"
ARCH="CCE_HIER_02_1"
GPU=3
ALPHA=0.01
EPOCHS=600
LAYERS=20
ETA=0.2
EXP_PATH="exp/MyDARTS/hier_${ETA}_cifar100_seed${SEED}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"

python train.py --data cifar100 --batch_size 64 --gpu ${GPU} \
    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --auxiliary --cutout \
    --dataset cifar100 --corruption_prob ${ETA} --corruption_type hierarchical --gold_fraction 0 --loss_func ${LOSS} \
    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}