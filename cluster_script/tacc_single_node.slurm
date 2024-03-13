#!/bin/bash
#SBATCH -J single-job-test              # Job name
#SBATCH -o debug/single.%j.out     # Name of stdout output file
#SBATCH -e debug/single.%j.err     # Name of stderr error file
#SBATCH -p rtx                  # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH --ntasks-per-node=4
#SBATCH -t 00:10:00              # Run time (hh:mm:ss)
#SBATCH --mail-user=rnjain@wisc.edu
#SBATCH --mail-type=all          # Send email at begin and end of job

echo "NODELIST="${SLURM_NODELIST}
master_addr_f=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR_F="$master_addr_f
export MASTER_ADDR=$master_addr_f
export JOB_ID=${SLURM_JOB_ID}

srun --nodes=1 -w $master_addr_f python single_submit.py &

wait


