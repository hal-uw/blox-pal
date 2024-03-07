#!/bin/bash

module load tacc-apptainer

singularity pull docker://nvcr.io/nvidia/pytorch:22.06-py3
singularity run --nv docker://nvcr.io/nvidia/pytorch:22.06-py3 /scratch1/08503/rnjain/blox-pal/blox_exp/scripts/run_imagenet.sh



