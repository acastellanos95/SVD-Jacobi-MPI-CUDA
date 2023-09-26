#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name BUILDSVDMPICUDA
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o salida.%j.txt
#SBATCH -e err-%j.log
#SBATCH --output=BUILDSVDMPICUDA-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andref.castellanos@cinvestav.mx

export OMP_NUM_THREADS=36
export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

module load numactl-2.0.14-gcc-10.2.0-62op4tg
module load hwloc-2.2.0-gcc-10.2.0-it4o6fq
module load glib-2.66.2-gcc-10.2.0-wcwvaat
module load gcc-10.2.0-gcc-10.2.0-u6fkk3y
module load openmpi/4.0.3/gcc
#module load openmpi/3.0.0/gcc
#module load openmpi/intel
# module load mpi/openmpi
module load cmake-3.18.4-gcc-10.2.0-ch2vvyp

cmake ..
make