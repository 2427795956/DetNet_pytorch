#!/bin/bash
NET=detnet59
DATASET=adas
BATCH_SIZE=1
CHECKPOINT=500
DISP_INTERVAL=2000
NUM_WORKERS=1
LR_RATE=0.01
vGPU=1

CHECKEPOCH=1
CHECKSESSION=1
SESSION=1
EXP_NAME="tired_tday_flip_600"

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
                --flip \
                --cuda

#echo "#########" ${i} "evaluate ########"
#CUDA_VISIBLE_DEVICES=${vGPU} /usr/bin/python test_net.py cag_tired_0907 --dataset ${DATASET} --net ${NET}  \
#                --checksession ${SESSION} \
#                --checkepoch ${CHECKEPOCH} --checkpoint ${CHECKPOINT} \
#                --bs ${BATCH_SIZE} \
#                --flip \
#                --cag \
#                --cuda
