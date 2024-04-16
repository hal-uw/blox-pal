#!/bin/bash

now=$(date +"%T")
echo "Starting Sia Simulation t = $now"

# Run
workloads=("workload-1" "workload-2" "workload-3" "workload-4" "workload-5" "workload-6" "workload-7" "workload-8")
placement_policies=("PAL" "PMFirst" "Default-Packed-S" "Default-Packed-NS" "Default-Random-S" "Default-Random-NS")
load_values=(2.0)

for workload in "${workloads[@]}"; do

    trace_path="workload-traces/philly/${workload}.csv"
    echo "Launching simulator for $workload"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator_sia.py \
    --cluster-job-log ./cluster_job_log \
    --sim-type trace-synthetic \
    --jobs-per-hour 2 \
    --exp-prefix sia \
    --start-job-track 0 \
    --end-job-track 159 \
    --trace $trace_path &
    pid1=$!

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python blox_new_flow_multi_run.py --start-id-track 0 --stop-id-track 159 --round-duration 360 --simulate &
    pid2=$!

    # Wait for resource_manager.py to finish
    wait $pid2

    for placement_policy in "${placement_policies[@]}"; do
        mv None_0_159_Fifo_AcceptAll_${placement_policy}_load_2.0_cluster_stats.json  None_0_159_Fifo_AcceptAll_${placement_policy}_${workload}_cluster_stats.json
        mv None_0_159_Fifo_AcceptAll_${placement_policy}_load_2.0_job_stats.json      None_0_159_Fifo_AcceptAll_${placement_policy}_${workload}_job_stats.json
        mv None_0_159_Fifo_AcceptAll_${placement_policy}_load_2.0_debug_stats.json    None_0_159_Fifo_AcceptAll_${placement_policy}_${workload}_debug_stats.json
        mv None_0_159_Fifo_AcceptAll_${placement_policy}_load_2.0_custom_stats.json   None_0_159_Fifo_AcceptAll_${placement_policy}_${workload}_custom_stats.json
        mv None_0_159_Fifo_AcceptAll_${placement_policy}_load_2.0_run_time_stats.json None_0_159_Fifo_AcceptAll_${placement_policy}_${workload}_run_time_stats.json
    
    done  
    
    echo "Resource manager finished, terminating simulator.py"
    if [ -n "$pid1" ]; then
        kill -9 $pid1
    else
        echo "simulator.py process does not exist"
    fi

    # Wait for 10 seconds before the next iteration to allow simulator.py to cleanup
    sleep 10

done