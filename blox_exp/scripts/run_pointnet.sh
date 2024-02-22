export CUDA_VISIBLE_DEVICES=$1
python $SCRATCH/Megatron-Resource/blox_exp/models/pointnet_ddp.py \
    --master-ip-address=$2 \
    --world-size=$3 \
    --rank=$4 \
    --master-ip-port=$5 \
    --job-id=$6 \
    --batch-size=32

