#!/usr/bin/env bash

SEED=1
DATA="cifar10"

#LOSS="rll"
#ARCH="DARTS_V2"
#ALPHA=0.01
#EPOCHS=3500
#LAYERS=8
#ETA=0.6
#GPU=0
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_weight/F1_unif${ETA}_${DATA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}  --random_weight

LOSS="rll"
ARCH="RLL001_UNIFORM_06"
ALPHA=0.01
EPOCHS=3500
LAYERS=8
ETA=0.6
GPU=0
EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_weight/F2_unif${ETA}_${DATA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"

python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}  --random_weight


#LOSS="rll"
#ARCH="CCE_UNIFORM_06"
#ALPHA=0.01
#EPOCHS=3500
#LAYERS=8
#ETA=0.6
#GPU=0
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_weight/no_aux_no_dropout/F3_unif${ETA}_${DATA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}


#LOSS="cce"
#ARCH="DARTS_V2"
#EPOCHS=3500
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_weight/F1_unif${ETA}_${DATA}_${LOSS}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS}

#LOSS="cce"
#ARCH="RLL001_UNIFORM_06"
#EPOCHS=3500
#LAYERS=8
#ETA=0.6
#GPU=1
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_weight/no_aux_no_dropout/F2_unif${ETA}_${DATA}_${LOSS}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED}  --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS}

#LOSS="rll"
#ARCH="CCE_UNIFORM_06"
#ALPHA=0.01
#EPOCHS=3500
#LAYERS=8
#ETA=0.6
#GPU=0
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_weight/no_aux_no_dropout/F3_unif${ETA}_${DATA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}
#
#LOSS="cce"
#ARCH="CCE_UNIFORM_06"
#EPOCHS=3500
#LAYERS=8
#ETA=0.6
#GPU=2
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_weight/no_aux_no_dropout/F3_unif${ETA}_${DATA}_${LOSS}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif --loss_func ${LOSS} \
#    --arch ${ARCH} --train_portion 0.9 --layers ${LAYERS}


#LOSS="rll"
#ARCH="DARTS_V2"
#ALPHA=0.01
#EPOCHS=3500
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_weight/F1_clean_${DATA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif  --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}  --random_weight


#LOSS="rll"
#ARCH="RLL001_UNIFORM_06"
#ALPHA=0.01
#EPOCHS=3500
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_weight/F2_clean_${DATA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif  --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}  --random_weight

#LOSS="rll"
#ARCH="CCE_UNIFORM_06"
#ALPHA=0.01
#EPOCHS=3500
#LAYERS=8
#ETA=0.6
#EXP_PATH="exp/MyDARTS/rank_loss_seed${SEED}/random_weight/F3_clean_${DATA}_${LOSS}${ALPHA}_${ARCH}_gpu${GPU}"
#
#python rank_loss.py --data "/home/yiwei/${DATA}" --batch_size 64 --gpu ${GPU} \
#    --epochs ${EPOCHS} --save ${EXP_PATH} --seed ${SEED} --cutout --drop_path_prob 0 \
#    --corruption_prob ${ETA} --corruption_type unif  --loss_func ${LOSS} \
#    --arch ${ARCH} --alpha ${ALPHA} --train_portion 0.9 --layers ${LAYERS}  --random_weight