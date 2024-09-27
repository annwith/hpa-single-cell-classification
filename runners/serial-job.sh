#!/bin/bash -l
#PBS  -N solo-job
#PBS  -q serial
#PBS  -l nodes=1:ppn=1

# Load the computing environment we need
module load python/3.8.11-gcc-9.4.0
module load cuda/12.0.0-gcc-12.2.0

# Execute the task
amdahl