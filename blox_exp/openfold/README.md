### Instructions for running OpenFold using container

On compute node
```
cd blox-pal/blox_exp/openfold
module load apptainer
apptainer exec --nv /scratch1/00946/zzhang/container/openfold_0.1.sif ./run.sh

```

Note that this runs OpenFold on a single node with 4 GPUs
You can call ncu or nvprof appropriately profile it. 



