#!/bin/bash

while read line
do 
    OIFS=$IFS; IFS="|"; set -- $line; session=$1;checkepoch=$2;checkpoint=$3; IFS=$OIFS 
    CUDA_VISIBLE_DEVICES=0 /usr/bin/python test_net.py tired_0907_s800_cag_flip \
    --dataset adas \
    --net detnet59 \
    --checksession ${session} \
    --checkepoch ${checkepoch} \
    --checkpoint ${checkpoint} \
    --bs 1 \
    --cag \
    --cuda \
    --thresh 0.05
done < s1.log
