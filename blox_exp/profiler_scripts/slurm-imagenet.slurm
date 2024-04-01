#!/bin/bash
#SBATCH -J single-job-profile              # Job name
#SBATCH -o debug/single.%j.out     # Name of stdout output file
#SBATCH -e debug/single.%j.err     # Name of stderr error file
#SBATCH -p rtx                  # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH --ntasks-per-node=4
#SBATCH -t 00:30:00              # Run time (hh:mm:ss)
#SBATCH --mail-user=rnjain@wisc.edu
#SBATCH --mail-type=all          # Send email at begin and end of job

# Initialise conda environment
conda init bash
conda --version
source ~/.bashrc
conda activate pollux

# Get the hostname of the allocated node
allocated_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)

# Resolve the hostname to get its IP address
host_ip=$(ip a show eno1 | awk '/inet / {print $2}' | cut -d'/' -f1)

echo $host_ip

# Define variables
CUDA_VISIBLE_DEVICES="0,1,2,3"
WORLD_SIZE=4
MASTER_IP_PORT="50051"
MODEL_NAME="resnet50"
JOB_ID="20"

# Run the script using srun for each GPU
for ((GPU_ID=0; GPU_ID<4; GPU_ID++)); do
    CUDA_VISIBLE_DEVICES=$GPU_ID srun -N 1 -n 1 bash -c "bash run_imagenet.sh '$CUDA_VISIBLE_DEVICES' '$host_ip' '$WORLD_SIZE' '$GPU_ID' '$MASTER_IP_PORT' '$MODEL_NAME' '$JOB_ID'"
done