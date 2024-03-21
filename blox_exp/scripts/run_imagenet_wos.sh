export CUDA_VISIBLE_DEVICES=$1

conda init bash
source ~/.bashrc
conda activate pollux

python3 /scratch1/08503/rnjain/blox-pal/blox_exp/models/imagenet_ddp.py \
    --master-ip-address=$2 \
    --world-size=$3 \
    --rank=$4 \
    --master-ip-port=$5 \
    --model-name=$6 \
    --job-id=$7 \
    --batch-size=32
