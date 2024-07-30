#!/bin/bash

ip_address=$(ip -o -4 addr list eno1 | awk '{print $4}' | cut -d/ -f1)

/home1/apps/cuda/12.2/bin/ncu --target-processes all -o dcgan_fu_report --csv --metrics smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active  ./run_dcgan_node.sh "0" $ip_address 1 50051 2 

