#!/bin/bash

/home1/apps/cuda/12.2/bin/ncu --target-processes all -o fu_report --csv --metrics smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active  ./run_pointnet_node.sh "0" 129.114.44.101 1 50051 2 

