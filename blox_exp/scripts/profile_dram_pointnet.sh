#!/bin/bash

/home1/apps/cuda/12.2/bin/ncu --target-processes all -o dram_report --csv --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./run_pointnet_node.sh "0" 129.114.44.101 1 50051 2 

