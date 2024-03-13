#!/bin/bash
#SBATCH -J cluster-test              # Job name
#SBATCH -o debug/cluster.%j.out     # Name of stdout output file
#SBATCH -e debug/cluster.%j.err     # Name of stderr error file
#SBATCH --qos=debug                 #debug preempt
#SBATCH -p rtx                  # Queue (partition) name
#SBATCH -N 4                     # Total # of nodes (must be 1 for serial)
#SBATCH --ntasks-per-node=4
#SBATCH -t 00:25:00              # Run time (hh:mm:ss)
#SBATCH --mail-user=rnjain@wisc.edu
#SBATCH --mail-type=all          # Send email at begin and end of job

echo "NODELIST="${SLURM_NODELIST}
master_addr_f=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# master_addr_s=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 2 | tail -n 1)
master_addr_s=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n 1)
echo "MASTER_ADDR_F="$master_addr_f
echo "MASTER_ADDR_S="$master_addr_s
export MASTER_ADDR=$master_addr_f
export JOB_ID=${SLURM_JOB_ID}
export NUM_NODES=$SLURM_JOB_NUM_NODES

#srun python -u debug_submit.py

