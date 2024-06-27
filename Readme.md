# PAL: A Variability-Aware Policy for Cluster Scheduling in GPU-Rich HPC Systems
This repository contains the source code implementation for the submission paper "PAL: A Variability-Aware Policy for Cluster Scheduling in GPU-Rich HPC Systems". The source code is from Blox ("Blox: A Modular Toolkit for Deep Learning Schedulers") with PAL and PM-First placement policies added.

### Pull this respository
```
git clone https://github.com/rutwik-n-jain/blox-pal.git
git checkout artifact
```

### Install dependencies
The following steps are used to create a conda virtual environment with the necessary dependencies installed. 

To install conda, follow steps in [Conda Installation](#Conda-Installation)


### A<sub>1</sub>: Sia Simulations





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
