conda init bash
source ~/.bashrc
conda activate pollux

export CUDA_VISIBLE_DEVICES=$1

python /scratch1/08503/rnjain/blox-pal/blox_exp/profilers/pointnet_ddp.py \
    --master-ip-address=$2 \
    --world-size=$3 \
    --rank=$4 \
    --master-ip-port=$5 \
    --job-id=$7 \
    --batch-size=32

