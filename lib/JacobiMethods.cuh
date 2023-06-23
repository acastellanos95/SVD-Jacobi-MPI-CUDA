//
// Created by andre on 6/16/23.
//

#ifndef SVD_JACOBI_MPI_CUDA_LIB_JACOBIMETHODS_CUH_
#define SVD_JACOBI_MPI_CUDA_LIB_JACOBIMETHODS_CUH_

#include <iostream>
#include <iomanip>
#include <set>
#include <unordered_map>
#include <random>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
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
void omp_mpi_dgesvd_local_matrices(SVD_OPTIONS jobu,
                                   SVD_OPTIONS jobv,
                                   size_t m,
                                   size_t n,
                                   MatrixMPI &A,
                                   size_t lda,
                                   MatrixMPI &s,
                                   MatrixMPI &V,
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
