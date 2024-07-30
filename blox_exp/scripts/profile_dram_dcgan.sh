#!/bin/bash

ip_address=$(ip -o -4 addr list eno1 | awk '{print $4}' | cut -d/ -f1)

/home1/apps/cuda/12.2/bin/ncu --target-processes all -o dcgan_dram_report --csv --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./run_dcgan_node.sh "0" $ip_address 1 50051 2 

