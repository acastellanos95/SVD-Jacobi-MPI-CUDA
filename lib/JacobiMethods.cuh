//
// Created by andre on 6/16/23.
//

#ifndef SVD_JACOBI_MPI_CUDA_LIB_JACOBIMETHODS_CUH_
#define SVD_JACOBI_MPI_CUDA_LIB_JACOBIMETHODS_CUH_

#include "../../../../../../usr/include/c++/11/iostream"
#include "../../../../../../usr/include/c++/11/iomanip"
#include "../../../../../../usr/include/c++/11/set"
#include "../../../../../../usr/include/c++/11/unordered_map"
#include "../../../../../../usr/include/c++/11/random"
#include "../../../../../../usr/lib/gcc/x86_64-linux-gnu/11/include/omp.h"
#include "../../../../../../usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
#include "../../../../../../usr/include/c++/11/vector"
#include "../../../../../../usr/local/cuda/targets/x86_64-linux/include/cublas_v2.h"
#include "../../../../../../usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h"
#include "../../../../../../usr/local/cuda/targets/x86_64-linux/include/cuda_runtime_api.h"
#include "Matrix.cuh"
#include "global.cuh"
#include "Utils.cuh"

namespace Thesis {

enum SVD_OPTIONS {
  AllVec,
  SomeVec,
  NoVec
};
/**
 *
 * @param jobu
 * @param jobv
 * @param m
 * @param n
 * @param A Matrix in a column major order
 * @param lda
 * @param s
 * @param U
 * @param ldu
 * @param V
 * @param ldv
 */
void omp_mpi_cuda_dgesvd_local_matrices(SVD_OPTIONS jobu,
                                   SVD_OPTIONS jobv,
                                   size_t m,
                                   size_t n,
                                   MatrixMPI &A,
                                   size_t lda,
                                   MatrixMPI &s,
                                   MatrixMPI &V,
                                   size_t ldv);

void cuda_dgesvd_kernel(SVD_OPTIONS jobu,
                        SVD_OPTIONS jobv,
                        size_t m,
                        size_t n,
                        Matrix &A,
                        size_t lda,
                        Matrix &s,
                        Matrix &V,
                        size_t ldv);

void omp_mpi_dgesvd_on_the_fly_matrices(SVD_OPTIONS jobu,
                                        SVD_OPTIONS jobv,
                                        size_t m,
                                        size_t n,
                                        MatrixMPI &A,
                                        size_t lda,
                                        MatrixMPI &s,
                                        MatrixMPI &V,
                                        size_t ldv);

__global__ void jacobi_rotation(unsigned int n, double *x, double *y, double c, double s);

} // Thesis

#endif //SVD_JACOBI_MPI_CUDA_LIB_JACOBIMETHODS_CUH_
