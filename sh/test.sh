#!/bin/bash
NET=detnet59
DATASET=adas
BATCH_SIZE=1
CHECKPOINT=86000
vGPU=1

CHECKEPOCH=3
CHECKSESSION=1
SESSION=1
THRESH=0.5

EXP_NAME='car_cag_800'

echo "######### test ########"
CUDA_VISIBLE_DEVICES=${vGPU} /usr/bin/python test_net.py ${EXP_NAME} \
                --dataset ${DATASET} \
                --net ${NET} \
                --checksession ${CHECKSESSION} \
                --checkepoch ${CHECKEPOCH} --checkpoint ${CHECKPOINT} \
                --bs ${BATCH_SIZE} \
                --thresh ${THRESH} \
                --cag \
                --cuda
                --vis
