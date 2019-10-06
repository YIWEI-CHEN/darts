#!/usr/bin/env bash

SEED=1
DATA="cifar10"

#LOSS="rll"
#ARCH="DARTS_V2"
#ALPHA=0.01
#EPOCHS=1000
#LAYERS=8
#ETA=0.6
#GPU=0
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_arch/F1_unif${ETA}_${DATA}_${LOSS}${ALPHA}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}  

LOSS="rll"
ALPHA=0.01
EPOCHS=1000
LAYERS=8
ETA=0.6
GPU=0
EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_arch/unif${ETA}_${DATA}_${LOSS}${ALPHA}_gpu${GPU}"

python rank_loss_with_random_arch.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
    --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}  


#LOSS="rll"
#ARCH="CCE_UNIFORM_06"
#ALPHA=0.01
#EPOCHS=1000
#LAYERS=8
#ETA=0.6
#GPU=2
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_arch/F3_unif${ETA}_${DATA}_${LOSS}${ALPHA}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}   


#LOSS="cce"
#ARCH="DARTS_V2"
#EPOCHS=1000
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_arch/F1_unif${ETA}_${DATA}_${LOSS}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --train_portion 0.9 --layers ${LAYERS}   

#LOSS="cce"
#ARCH="RLL001_UNIFORM_06"
#EPOCHS=1000
#LAYERS=8
#ETA=0.6
#GPU=1
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_arch/F2_unif${ETA}_${DATA}_${LOSS}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED}  --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --train_portion 0.9 --layers ${LAYERS}   

#LOSS="rll"
#ARCH="CCE_UNIFORM_06"
#ALPHA=0.01
#EPOCHS=1000
#LAYERS=8
#ETA=0.6
#GPU=0
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_arch/F3_unif${ETA}_${DATA}_${LOSS}${ALPHA}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}   
#
#LOSS="cce"
#ARCH="CCE_UNIFORM_06"
#EPOCHS=1000
#LAYERS=8
#ETA=0.6
#GPU=3
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_arch/F3_unif${ETA}_${DATA}_${LOSS}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --train_portion 0.9 --layers ${LAYERS}   


#LOSS="rll"
#ARCH="DARTS_V2"
#ALPHA=0.01
#EPOCHS=1000
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_arch/F1_clean_${DATA}_${LOSS}${ALPHA}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif  --loss_func ${LOSS} \
#    --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}  


#LOSS="rll"
#ARCH="RLL001_UNIFORM_06"
#ALPHA=0.01
#EPOCHS=1000
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_arch/F2_clean_${DATA}_${LOSS}${ALPHA}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif  --loss_func ${LOSS} \
#    --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}  

#LOSS="rll"
#ARCH="CCE_UNIFORM_06"
#ALPHA=0.01
#EPOCHS=1000
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_arch/F3_clean_${DATA}_${LOSS}${ALPHA}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif  --loss_func ${LOSS} \
#    --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}  