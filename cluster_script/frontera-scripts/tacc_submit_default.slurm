#!/bin/bash
#SBATCH -J cluster-test              # Job name
#SBATCH -o debug/cluster.%j.out     # Name of stdout output file
#SBATCH -e debug/cluster.%j.err     # Name of stderr error file
#SBATCH -p rtx                 # Queue (partition) name
#SBATCH -N 2                    # Total # of nodes
#SBATCH --ntasks-per-node=16 
#SBATCH -t 06:00:00              # Run time (hh:mm:ss)
#SBATCH --mail-user=rnjain@wisc.edu
#SBATCH --mail-type=all          # Send email at begin and end of job

# Load necessary modules
module load tacc-apptainer
module load cuda

# Initialise conda environment
conda init bash
conda --version
source ~/.bashrc
conda activate pollux

# Get the IP address of the first node
node_ip=$(scontrol show hostnames | head -n 1)

# Export the IP address to an environment variable
export SCHEDULER_NODE_IP_ADDR=$node_ip

echo "IP Address of scheduler ${node_ip}"
echo "Launching run script on each node"

# Run submit script
python -u tacc_submit_default.py &

echo "Python script launched"

wait

wait

echo "All subprocesses have completed"
