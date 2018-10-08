#!/bin/bash
NET=detnet59
DATASET=adas
CHECKPOINT=52000
vGPU=2

CHECKEPOCH=2
CHECKSESSION=1

#EXP_NAME='tired_tday_flip_720'
EXP_NAME='adas_mc_600'
VIDEO_FILE='/home/zuosi/Videos/0101.h264'

echo "######### demo ########"
CUDA_VISIBLE_DEVICES=${vGPU} /usr/bin/python demo_net.py ${EXP_NAME} \
		--video_file ${VIDEO_FILE} \
                --dataset ${DATASET} \
                --net ${NET} \
                --checksession ${CHECKSESSION} \
                --checkepoch ${CHECKEPOCH} --checkpoint ${CHECKPOINT} \
                --cag \
                --cuda 
