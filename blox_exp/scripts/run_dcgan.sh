module load tacc-apptainer

export CUDA_VISIBLE_DEVICES=$1

singularity pull docker://nvcr.io/nvidia/pytorch:22.06-py3
singularity exec --nv --cleanenv -B /scratch1/08503/rnjain/anaconda3:/opt/conda docker://nvcr.io/nvidia/pytorch:22.06-py3 bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate pollux && python $SCRATCH/blox-pal/blox_exp/models/dcgan_ddp.py \
    --master-ip-address=$2 \
    --world-size=$3 \
    --rank=$4 \
    --master-ip-port=$5 \
    --job-id=$6 \
    --batch-size=128"

