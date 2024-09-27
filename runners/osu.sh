#!/bin/bash -l
#PBS  -N osu-benchmark-ucx
#PBS  -q paralela
#PBS  -l nodes=2:ppn=128

module load openmpi4/4.1.1

# Load the computing environment we need
module load singularity/3.8.3-gcc-9.4.0

#singularity exec /home/lovelace/proj/proj1012/p204481/osu_benchmark/osu_micro_benchmarks_ucx.sif mpirun -np 2 osu_bw
mpirun -np 2 singularity exec /home/lovelace/proj/proj1012/p204481/osu_benchmark/osu_micro_benchmarks_ucx.sif osu_bw
