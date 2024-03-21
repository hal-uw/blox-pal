export CUDA_VISIBLE_DEVICES=$1

conda init bash
source ~/.bashrc
conda activate pollux

python /scratch1/08503/rnjain/blox-pal/blox_exp/models/dcgan_ddp.py \
    --master-ip-address=$2 \
    --world-size=$3 \
    --rank=$4 \
    --master-ip-port=$5 \
    --job-id=$6 \
    --batch-size=128

