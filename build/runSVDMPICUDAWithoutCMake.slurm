#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name SVDMPICUDA
#SBATCH -N 2
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH -o salida.%j.txt
#SBATCH -e err-%j.log
#SBATCH --output=SVDMPICUDA-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andref.castellanos@cinvestav.mx

export OMP_NUM_THREADS=36
export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

module load numactl-2.0.14-gcc-10.2.0-62op4tg hwloc-2.2.0-gcc-10.2.0-it4o6fq glib-2.66.2-gcc-10.2.0-wcwvaat gcc-10.2.0-gcc-10.2.0-u6fkk3y openmpi/4.0.3/gcc cmake-3.18.4-gcc-10.2.0-ch2vvyp binutils-2.35.1-gcc-10.2.0-q56xo3i

mpicxx --showme:link
mpicxx --showme:compile

#cmake ..
#make
nvcc -Xcompiler -fopenmp ../main.cu -o MPISVDCUDA -I/usr/mpi/gcc/openmpi-4.0.3-hfi/include -L/usr/mpi/gcc/openmpi-4.0.3-hfi/lib64 -lmpi -lgomp
#nvcc -Xcompiler -fopenmp  -o MPISVDCUDA  -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -lgomp

#mpiexec --map-by ppr:1:node --bind-to core -x OMP_NUM_THREADS -v -display-map -display-allocation SVD_Jacobi_MPI_OMP 1000
mpiexec --bind-to none -n 2 -display-map -display-allocation MPISVDCUDA 5000
mpiexec --bind-to none -n 2 -display-map -display-allocation MPISVDCUDA 10000
mpiexec --bind-to none -n 2 -display-map -display-allocation MPISVDCUDA 20000
mpiexec --bind-to none -n 2 -display-map -display-allocation MPISVDCUDA 30000
#mpiexec --bind-to hwthread --map-by ppr:1:node:pe=36 -x OMP_NUM_THREADS -v -display-map -display-allocation SVD_Jacobi_MPI_CUDA 10000
#mpiexec --bind-to hwthread --map-by ppr:1:node:pe=36 -x OMP_NUM_THREADS -v -display-map -display-allocation SVD_Jacobi_MPI_CUDA 20000
#mpiexec --bind-to hwthread --map-by ppr:1:node:pe=36 -x OMP_NUM_THREADS -v -display-map -display-allocation SVD_Jacobi_MPI_CUDA 30000