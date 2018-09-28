#!/bin/bash
NET=detnet59
DATASET=adas
BATCH_SIZE=1
CHECKPOINT=82000
DISP_INTERVAL=2000
NUM_WORKERS=1
LR_RATE=0.01
vGPU=1

CHECKEPOCH=2
CHECKSESSION=1
SESSION=1
DECAY_EPOCH=5

echo "#########" ${i} "training ########"
CUDA_VISIBLE_DEVICES=${vGPU} /usr/bin/python train_net.py tired_0925_s600_cag_flip --dataset ${DATASET} --net ${NET} \
                --session ${SESSION} --checksession ${CHECKSESSION} \
                --checkepoch ${CHECKEPOCH} --checkpoint ${CHECKPOINT} \
                --disp_interval ${DISP_INTERVAL} \
                --lr ${LR_RATE} \
                --bs ${BATCH_SIZE} \
                --lr_decay_step ${DECAY_EPOCH} \
                --r \
                --cag \
                --flip \
                --cuda

#echo "#########" ${i} "evaluate ########"
#CUDA_VISIBLE_DEVICES=${vGPU} /usr/bin/python test_net.py cag_tired_0907 --dataset ${DATASET} --net ${NET}  \
#                --checksession ${SESSION} \
#                --checkepoch ${CHECKEPOCH} --checkpoint ${CHECKPOINT} \
#                --bs ${BATCH_SIZE} \
#                --cag \
#                --flip \
#                --cuda
