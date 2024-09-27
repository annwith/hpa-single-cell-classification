#!/bin/bash -l
#PBS  -N parallel-job-com-4
#PBS  -q testes
#PBS  -l nodes=1:ppn=4

# Load the computing environment we need
module load python/3.8.11-gcc-9.4.0
module load cuda/12.0.0-gcc-12.2.0

# Execute the task
mpiexec amdahl
