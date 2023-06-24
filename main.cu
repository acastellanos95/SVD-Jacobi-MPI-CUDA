#include <iostream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <omp.h>
#include <fstream>
#include <random>
//#include <mkl/mkl.h>
#include <set>
#include "mpi.h"
#include "lib/Matrix.cuh"
#include "lib/JacobiMethods.cuh"
#include "lib/Utils.cuh"
#include "lib/global.cuh"
//#include "tests/Tests.cuh"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  //  cudaSetDevice(0);
//  cudaMemPool_t memPool;
//  CHECK_CUDA(cudaDeviceGetDefaultMemPool(&memPool, 0));
//  uint64_t thresholdVal = ULONG_MAX;
//  CHECK_CUDA(cudaMemPoolSetAttribute(
//      memPool, cudaMemPoolAttrReleaseThreshold, (void *)&thresholdVal));

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // SEED!!!
  const unsigned seed = 1000000;

  time_t t;
  tm tm;
  std::stringstream file_output;
  std::ostringstream oss;
  std::string now_time;
  size_t height = std::stoul(argv[1]);
  size_t width = std::stoul(argv[1]);

  MatrixMPI A, V, s, A_copy;

  // Select iterator
  auto iterator = Thesis::IteratorC;

  if (rank == ROOT_RANK) {
    // Initialize other variables

    t = std::time(nullptr);
    tm = *std::localtime(&t);
    file_output = std::stringstream();
    oss = std::ostringstream();
    now_time = std::string();
    oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    now_time = oss.str();

    // Matrix initialization and fill
    A = MatrixMPI(height, width);
    V = MatrixMPI(width, width);
    s = MatrixMPI(1, std::min(A.height, A.width));
    A_copy = MatrixMPI(height, width);

    std::fill_n(V.elements, V.height * V.width, 0.0);
    std::fill_n(A.elements, A.height * A.width, 0.0);
    std::fill_n(s.elements, s.height * s.width, 0.0);
    std::fill_n(A_copy.elements, A_copy.height * A_copy.width, 0.0);

    // Create R matrix
    std::default_random_engine e(seed);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    for (size_t indexRow = 0; indexRow < std::min<size_t>(height, width); ++indexRow) {
      for (size_t indexCol = indexRow; indexCol < std::min<size_t>(height, width); ++indexCol) {
        double value = uniform_dist(e);
        A.elements[iterator(indexRow, indexCol, height)] = value;
        A_copy.elements[iterator(indexRow, indexCol, height)] = value;
      }
    }

#ifdef TESTS
    std::default_random_engine e_test(seed);
    std::uniform_real_distribution<double> uniform_dist_test(0.0, 1.0);
    for (size_t indexRow = 0; indexRow < height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < width; ++indexCol) {
        double value = uniform_dist_test(e_test);
        A.elements[iterator(indexRow, indexCol, height)] = value;
        A_copy.elements[iterator(indexRow, indexCol, height)] = value;
      }
    }
#endif

    file_output << "Number of threads: " << omp_get_num_threads() << '\n';
    file_output << "Dimensions, height: " << height << ", width: " << width << "\n";
    std::cout << "Dimensions, height: " << height << ", width: " << width << "\n";
  }
  // Calculate SVD decomposition
  double ti = omp_get_wtime();
  Thesis::omp_mpi_dgesvd_local_matrices(Thesis::AllVec,
                                        Thesis::AllVec,
                                        height,
                                        width,
                                        A,
                                        height,
                                        s,
                                        V,
                                        width);
//  Thesis::omp_mpi_dgesvd_on_the_fly_matrices(Thesis::AllVec,
//                                        Thesis::AllVec,
//                                        height,
//                                        width,
//                                        A,
//                                        height,
//                                        s,
//                                        V,
//                                        width);
//  Thesis::Tests::test_local_matrix_distribution_in_sublocal_matrices_blocking(height, width, A, height);
//  Thesis::Tests::test_local_matrix_distribution_in_sublocal_matrices_concurrent(height, width, A, height);
//  Thesis::Tests::test_local_matrix_distribution_on_the_fly_concurrent(height, width, A, height);
//  Thesis::Tests::test_local_matrix_distribution_on_the_fly_blocking(height, width, A, height);
//  Thesis::Tests::test_MPI_Isend_Recv();
  double tf = omp_get_wtime();
  double time = tf - ti;

  if (rank == ROOT_RANK) {

    // Report \Sigma
//    file_output << std::fixed << std::setprecision(3) << "sigma: \n";
//    std::cout << std::fixed << std::setprecision(3) << "sigma: \n";
//    for (size_t indexCol = 0; indexCol < std::min(height, width); ++indexCol) {
//      file_output << s.elements[indexCol] << " ";
//      std::cout << s.elements[indexCol] << " ";
//    }
//    file_output << '\n';
//    std::cout << '\n';
//
//    // Report Matrix V
//    file_output << std::fixed << std::setprecision(3) << "V: \n";
//    std::cout << std::fixed << std::setprecision(3) << "V: \n";
//    for (size_t indexRow = 0; indexRow < height; ++indexRow) {
//      for (size_t indexCol = 0; indexCol < width; ++indexCol) {
//        file_output << V.elements[Thesis::IteratorC(indexRow, indexCol, width)] << " ";
//        std::cout << V.elements[Thesis::IteratorC(indexRow, indexCol, width)] << " ";
//      }
//      file_output << '\n';
//      std::cout << '\n';
//    }

    file_output << "SVD MPI+OMP time with U,V calculation: " << time << "\n";
    std::cout << "SVD MPI+OMP time with U,V calculation: " << time << "\n";

    // A - A*
    #pragma omp parallel for
    for (size_t indexRow = 0; indexRow < height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < width; ++indexCol) {
        double value = 0.0;
        for (size_t k_dot = 0; k_dot < width; ++k_dot) {
          value += A.elements[iteratorC(indexRow, k_dot, height)] * s.elements[k_dot]
              * V.elements[iteratorC(indexCol, k_dot, height)];
        }
        A_copy.elements[iteratorC(indexRow, indexCol, height)] -= value;
//        A_copy.elements[iteratorC(indexRow, indexCol, height)] -= A.elements[iteratorC(indexRow, indexCol, height)];
      }
    }

    // Calculate frobenius norm
    double frobenius_norm = 0.0;
#pragma omp parallel for reduction(+:frobenius_norm)
    for (size_t indexRow = 0; indexRow < height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < width; ++indexCol) {
        double value = A_copy.elements[iteratorC(indexRow, indexCol, height)];
        frobenius_norm += value*value;
      }
    }

    file_output << "||A-USVt||_F: " << sqrt(frobenius_norm) << "\n";
    std::cout << "||A-USVt||_F: " << sqrt(frobenius_norm) << "\n";

    std::ofstream file("reporte-dimension-" + std::to_string(height) + "-time-" + now_time + ".txt", std::ofstream::out | std::ofstream::trunc);
    file << file_output.rdbuf();
    file.close();
    A.free(), V.free(), s.free(), A_copy.free();
  }

  MPI_Finalize();

  return 0;
}
