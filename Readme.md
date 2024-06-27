# PAL: A Variability-Aware Policy for Cluster Scheduling in GPU-Rich HPC Systems
This repository contains the source code implementation for the submission paper "PAL: A Variability-Aware Policy for Cluster Scheduling in GPU-Rich HPC Systems". The source code is from Blox ("Blox: A Modular Toolkit for Deep Learning Schedulers") with PAL and PM-First placement policies added.

### Pull this respository
```
git clone https://github.com/rutwik-n-jain/hal_uw.git
cd blox-pal
git checkout artifact
```

### Install dependencies
The following steps are used to create a conda virtual environment with the necessary dependencies installed. 

To install conda, follow steps in [Conda Installation](#Conda-Installation)

```
conda create -n envpal python=3.8
conda activate envpal
python3 -m pip install -r requirements.txt
```

Before running simulations, we need to build gRPC stubs.
```
cd blox/deployment
make rpc
```


## Artifact Components

- **A1**: Sia-Philly Trace: Baseline Simulations
- **A2**: Sia-Philly Trace: Varying Locality Penalty
- **A3**: Synergy Trace: Varying Job Load and Schedulers

## Mapping Artifact Components to Paper Elements

| Artifact Component | Contributions Supported | Related Paper Elements |
|--------------------|-------------------------|------------------------|
| **A1**             | C1                      | Section V.B            |
|                    |                         | Figure 10              |
| **A2**             | C2                      | Section V.B            |
|                    |                         | Figure 12              |
| **A3**             | C3                      | Section V.C            |
|                    |                         | Figure 13              |
|                    |                         | Section V.C            |
|                    |                         | Figures 15 and 16      |

## A<sub>1</sub>: Sia Simulations

### Evaluation

We evaluated the performance of our placement policies relative to baselines using the Sia-Philly workload traces. Section IV.B of the paper provides details about workload and cluster configuration, and results are presented in Section V.B.

### Performance

- **PM-First** improves average JCT by 40% geomean (min 5%, max 59%)
- **PAL** improves average JCT by 43% geomean (mix 21%, max 59%) compared to baseline Packed-Sticky placement.

### Computational Time

- The expected computational time of this artifact on a CPU node is around 180 minutes. This can be reduced by running fewer workload traces (instructions for the same are specified in Artifact Execution).
- Artifact setup is expected to take 20 minutes or less, and the artifact analysis script takes 3 minutes.

### Hardware Requirements

Simulations require a CPU machine. All experiments were run with an x86_64 machine (Intel E5-2630 v3 8-core CPUs at 2.40 GHz - Haswell w/ EM64T with 8x 16 GB DDR4 memory) running Ubuntu 18.04.6. Code has also been tested on Mac M1 with Darwin Kernel Version 23.4.0.

### Software Requirements

Simulations use gRPC to communicate, while analysis scripts use matplotlib, seaborn, and other Python libraries to plot several collected metrics. We recommend that users create a virtual environment to install these dependencies.


Before running simulations, we need to build gRPC stubs.
```
cd blox/deployment
make rpc
```

We provide a run script that launches all simulations and produces relevant output logs for each workload trace in the Sia-Philly trace set. 
To reduce computational time, edit line 7 of the script to run fewer traces. 
```
conda activate envpal
./run_sia_sim_baseline.sh
```

To reproduce Figure 10 in the paper, run the following plotting script:
```
python plot_sia_sim_baseline.py
```

## A<sub>2</sub>: Sia Locality-Sweep Simulation

### Evaluation
We analyze how different inter-node locality penalty values affect our policies for the Sia-Philly workloads. As the locality penalty increases, the best-performing baseline (Packed-Sticky) improves its average JCT and approaches
**PM-First**’s and **PAL**’s performance. **PM-First**’s average JCT improvement over Packed-Sticky decreases from 30% to 9% as
the locality penalty increases from 1.0 to 3.0 **PAL** outperforms both PM-First and Packed-Sticky, with benefits over Packed-Sticky only decreasing from 30% to 20% geomean.

### Execution
We provide a run script that varies the inter-node locality penalty from 1.0 to 3.0 in steps of 0.5 and runs simulations at each value to produce relevant output logs. 

```
conda activate envpal
./run_sia_sim_losweep.sh
```

To reproduce Figure 12 in the paper, run the following plotting script:
```
python plot_sia_sim_losweep.py
```

## A<sub>3</sub>: Varying Cluster Contention and Scheduling Policies

### Evaluation
We vary the job arrival rate, and in turn, the contention level on the cluster, and measure JCTs to evaluate the performance of our policies. We use the Synergy workload and simulate a 256 GPU cluster. These experiments are also run with three different scheduling policies - FIFO, LAS, and SRTF. 

### Execution
We provide a run script `run_synergy_baseline.sh` to vary job load from 8 to 14 jobs/hour and run experiments for all 3 schedulers. 

{{ \footnotesize
\begin{minted}{bash}
conda activate envpal
./run_synergy_sim.sh
\end{minted}
}}

Lines 441-447 of `simulator_synergy.py` specify the job load and scheduling policies to run as arrays. These can be modified to reduce the number of experiments, and consequently the execution time for A<sub>3</sub>. 

To reproduce Figure 13 in the paper, run the following plotting script:
```
python plot_synergy_fifo.py
```
To reproduce Figures 15 and 16 in the paper, run the following plotting script:
```
python plot_synergy_las.py
python plot_synergy_srtf.py
```


### Conda Installation
(1) Get the latest version of Anaconda. The code has been tested with Anaconda3 23.3.1
```
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
```
(2) Run the install script to install Anaconda
```
chmod +x Anaconda3-2023.03-1-Linux-x86_64.sh
./Anaconda3-2023.03-1-Linux-x86_64.sh
source ~/.bashrc
conda --version
```
