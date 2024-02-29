module load tacc-apptainer

export CUDA_VISIBLE_DEVICES=$1

singularity pull docker://nvcr.io/nvidia/pytorch:22.06-py3
singularity exec --nv docker://nvcr.io/nvidia/pytorch:22.06-py3 python $SCRATCH/Megatron-Resource/blox_exp/models/dcgan_ddp.py \
    --master-ip-address=$2 \
    --world-size=$3 \
    --rank=$4 \
    --master-ip-port=$5 \
    --job-id=$6 \
    --batch-size=128

