#!/bin/bash
NET=detnet59
DATASET=adas
BATCH_SIZE=1
CHECKPOINT=52000
DISP_INTERVAL=2000
NUM_WORKERS=1
LR_RATE=0.01
vGPU=3

CHECKEPOCH=2
CHECKSESSION=1
SESSION=1
#EXP_NAME="tired_tday_flip_720"
EXP_NAME="adas_mc_600"
#EXP_NAME="tired_tday_mooncake10000_flip_800"

echo "######### training ########"
CUDA_VISIBLE_DEVICES=${vGPU} /usr/bin/python train_net.py ${EXP_NAME} \
                --dataset ${DATASET} --net ${NET} \
                --session ${SESSION} --checksession ${CHECKSESSION} \
                --checkepoch ${CHECKEPOCH} --checkpoint ${CHECKPOINT} \
                --disp_interval ${DISP_INTERVAL} \
                --lr ${LR_RATE} \
                --bs ${BATCH_SIZE} \
                --r \
                --cag \
                --cuda
