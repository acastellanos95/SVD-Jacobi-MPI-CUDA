//
// Created by andre on 6/16/23.
//

#ifndef SVD_JACOBI_MPI_CUDA_LIB_UTILS_CUH_
#define SVD_JACOBI_MPI_CUDA_LIB_UTILS_CUH_

#include "Matrix.cuh"
#include "global.cuh"
#include <tuple>
#include <stdexcept>
#include <functional>
#include <iostream>
#include <cmath>

namespace Thesis {

enum MATRIX_LAYOUT{
  ROW_MAJOR,
  COL_MAJOR
};

std::tuple<double, double> non_sym_Schur_ordered(MATRIX_LAYOUT matrix_layout,
                                                 size_t m,
                                                 size_t n,
                                                 const Matrix &A,
                                                 size_t lda,
                                                 size_t p,
                                                 size_t q,
                                                 double alpha,
                                                 double beta);

std::tuple<double, double> non_sym_Schur_non_ordered(const std::function<size_t(size_t, size_t, size_t)> &iterator,
                                                     const size_t m,
                                                     const size_t n,
                                                     const Matrix &A,
                                                     const size_t lda,
                                                     const size_t p,
                                                     const size_t q,
                                                     const double alpha);

size_t IteratorC(size_t i, size_t j, size_t ld);
size_t IteratorR(size_t i, size_t j, size_t ld);

std::function<size_t(size_t, size_t, size_t)> get_iterator(MATRIX_LAYOUT matrix_layout);

int omp_thread_count();

} // Thesis

#endif //SVD_JACOBI_MPI_CUDA_LIB_UTILS_CUH_
