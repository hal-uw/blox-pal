#!/bin/bash
#SBATCH -J cluster_experiment
#SBATCH -o cluster_experiment.%j.out
#SBATCH -e cluster_experiment.%j.err
#SBATCH -N 8
#SBATCH -t 24:00:00

# Load necessary modules
module load python
module load anaconda3

# Define your directory paths
REDIS_PATH="/scratch1/08503/rnjain/blox-pal/redis-7.2.4/src/redis-server"
SCHEDULER_SCRIPT="/scratch1/08503/rnjain/blox-pal/fifo_scheduler_cluster.py"
WORKER_SCRIPT="/scratch1/08503/rnjain/worker_node/blox-pal/node_manager.py"
SUBMIT_SCRIPT="/scratch1/08503/rnjain/worker_node/blox-pal/blox_exp/submit_modified.py"
CONDA_ENV="pollux"
WAIT_TIME=60

# Function to get IP address of a node
get_ip_address() {
    # Extract IP address of specified interface
    ip_addr=$(hostname -I | awk '{print $1}')
    echo $ip_addr
}

# Function to launch Redis server
launch_redis() {
    $REDIS_PATH &
}

# Function to activate conda environment
activate_conda() {
    source activate $CONDA_ENV
}

# Function to launch scheduler script
launch_scheduler_script() {
    python -u $SCHEDULER_SCRIPT --round-duration 30 --start-id-track 0 --stop-id-track 2 &
}

# Function to launch worker script
launch_worker_script() {
    ip_addr=$1
    worker_node_id=$2
    python -u $WORKER_SCRIPT --ipaddr $ip_addr --interface eno1 > /scratch1/08503/rnjain/worker_node/blox-pal/node_manager_stdout_${worker_node_id}.txt 2>&1 &
}

# Main function
main() {
    # Get scheduler node from SLURM_JOB_NODELIST
    read -a nodes <<< $(echo $SLURM_JOB_NODELIST | tr -d '[],')
    scheduler_node=${nodes[0]}

    # Activate conda environment
    activate_conda

    # Launch Redis server on all nodes
    srun -N $SLURM_JOB_NUM_NODES -n $SLURM_JOB_NUM_NODES -l bash -c "$REDIS_PATH &"

    # Launch scheduler script on scheduler node
    if [[ $(hostname) == *$scheduler_node* ]]; then
        launch_scheduler_script
        sleep $WAIT_TIME
    fi

    # Get IP address of scheduler node
    scheduler_ip=$(srun --nodes=1 --ntasks=1 -w $scheduler_node bash -c "$(typeset -f get_ip_address); get_ip_address")

    # Launch worker scripts on worker nodes
    for ((i=1; i<${#nodes[@]}; i++)); do
        worker_node=${nodes[$i]}
        worker_ip=$scheduler_ip
        if [[ $(hostname) != *$worker_node* ]]; then
            srun --nodes=1 --ntasks=1 -w $worker_node bash -c "$(typeset -f launch_worker_script); launch_worker_script $worker_ip $i"
        fi
    done

    # Launch submit script on scheduler node
    if [[ $(hostname) == *$scheduler_node* ]]; then
        python -u $SUBMIT_SCRIPT > /scratch1/08503/rnjain/worker_node/blox-pal/submit_stdout.txt 2>&1 &
    fi
}

# Execute main function
main

