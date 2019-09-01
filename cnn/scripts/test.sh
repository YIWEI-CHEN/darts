#!/usr/bin/env bash

DATE=`date +%m%d`

GPU=3
python test.py --data cifar10 --gpu ${GPU} --auxiliary --model_path cifar10_model.pt