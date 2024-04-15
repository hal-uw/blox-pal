#!/bin/bash

now=$(date +"%T")
echo "Starting Synergy Simulation t = $now"

# Run

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator_synergy.py \
--cluster-job-log ./cluster_job_log \
--sim-type trace-synthetic \
--jobs-per-hour 2 \
--exp-prefix sia \
--start-job-track 2000 \
--end-job-track 3000 &

pid1=$!

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python blox_new_flow_multi_run.py --start-id-track 2000 --stop-id-track 3000 --round-duration 300 --simulate &
pid2=$!

# Wait for resource_manager.py to finish
wait $pid2

echo "Resource manager finished, terminating simulator.py"
if [ -n "$pid1" ]; then
    kill -9 $pid1
else
    echo "simulator.py process does not exist"
fi

finish=$(date +"%T")
echo "Started Synergy Simulation t = $now"
echo "Ended Synergy Simulation t = $finish"