#!/bin/bash

# Check if the number of arguments is correct
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <CUDA_VISIBLE_DEVICES> <MASTER_IP_ADDRESS> <WORLD_SIZE> <MASTER_IP_PORT> <JOB_ID>"
    exit 1
fi

# Assign script arguments to variables
CUDA_VISIBLE_DEVICES=$1
MASTER_IP_ADDRESS=$2
WORLD_SIZE=$3
MASTER_IP_PORT=$4
JOB_ID=$5

# Run the script for each GPU
for (( RANK=0; RANK<$WORLD_SIZE; RANK++ ))
do
    CUDA_VISIBLE_DEVICES=$RANK python /scratch1/08503/rnjain/hal-blox-pal/blox-pal/blox_exp/models/pointnet_ddp.py \
        --master-ip-address=$MASTER_IP_ADDRESS \
        --world-size=$WORLD_SIZE \
        --rank=$RANK \
        --master-ip-port=$MASTER_IP_PORT \
        --job-id=$JOB_ID \
        --batch-size=32 &
done

# Wait for all background processes to finish
wait

