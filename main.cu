#include <iostream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <omp.h>
#include <fstream>
#include <random>
#include <set>
#include <vector>
#include <unordered_map>
#include "mpi.h"
//#include "lib/Matrix.cuh"
//#include "lib/JacobiMethods.cuh"
//#include "lib/Utils.cuh"
//#include "lib/global.cuh"

struct MatrixMPI{
  unsigned long width{};
  unsigned long height{};
  double *elements{};

  MatrixMPI()= default;

  MatrixMPI(unsigned long height, unsigned long width){
    this->width = width;
    this->height = height;

    this->elements = new double [height * width];
  }

  MatrixMPI(std::vector<double> &A, unsigned long height, unsigned long width){
    this->width = width;
    this->height = height;

    this->elements = new double [height * width];
    size_t all_size = height * width;

    std::copy(A.begin(), A.end(), this->elements);
  }

  MatrixMPI(MatrixMPI &matrix_mpi){
    this->width = matrix_mpi.width;
    this->height = matrix_mpi.height;

    this->elements = new double [height * width];
    std::copy(matrix_mpi.elements, matrix_mpi.elements + height * width, this->elements);
  }

  void free(){
    delete []elements;
  }
};

struct Matrix{
  unsigned long width;
  unsigned long height;
  double *elements;

  Matrix(unsigned long height, unsigned long width){
    this->width = width;
    this->height = height;

    this->elements = new double [height * width];
  }

  ~Matrix(){
    delete []elements;
  }
};

struct Vector{
  unsigned long length;
  double *elements;
  ~Vector(){
    delete []elements;
  }
};

struct CUDAMatrix{
  unsigned long width;
  unsigned long height;
  double *elements;

  CUDAMatrix(unsigned long height, unsigned long width){
    this->width = width;
    this->height = height;

    cudaMalloc(&this->elements, height * width * sizeof(double));
    cudaMemset(&this->elements, 0, height * width * sizeof(double));
  }

  explicit CUDAMatrix(Matrix &matrix){
    this->width = matrix.width;
    this->height = matrix.height;

    cudaMalloc(&this->elements, height * width * sizeof(double));
    cudaMemcpy(this->elements, matrix.elements, this->height * this->width * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  void copy_to_host(Matrix &matrix) const{
    cudaMemcpy(matrix.elements,
               this->elements,
               matrix.width * matrix.height * sizeof(double),
               cudaMemcpyDeviceToHost);
  }

  void copy_from_host(Matrix &matrix) const{
    cudaMemcpy(this->elements,
               matrix.elements,
               width * height * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  void copy_from_device(CUDAMatrix &matrix) const{
    cudaMemcpy(this->elements,
               matrix.elements,
               width * height * sizeof(double),
               cudaMemcpyDeviceToDevice);
  }

  void free() const{
    cudaFree(this->elements);
  }
};

struct CUDAVector{
  unsigned long length;
  double *elements;
};

int omp_thread_count() {
  int n = 0;
#pragma omp parallel reduction(+:n)
  n += 1;
  return n;
}

__global__ void jacobi_rotation(unsigned int n, double *x, double *y, double c, double s) {
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    double tmp = x[i];
    x[i] = c * tmp - s * y[i];
    y[i] = s * tmp + c * y[i];
  }
}

// For double precision accuracy in the eigenvalues and eigenvectors, a tolerance of order 10−16 will suffice. Erricos
#define TOLERANCE 1e-16

#define ROOT_RANK 0

#define iteratorR(i,j,ld)(((i)*(ld))+(j))
#define iteratorC(i,j,ld)(((j)*(ld))+(i))

enum SVD_OPTIONS {
  AllVec,
  SomeVec,
  NoVec
};

void cuda_dgesvd_kernel(SVD_OPTIONS jobu,
                        SVD_OPTIONS jobv,
                        size_t m,
                        size_t n,
                        Matrix &A,
                        size_t lda,
                        Matrix &s,
                        Matrix &V,
                        size_t ldv) {
  auto num_of_threads = omp_thread_count();

  int threadsPerBlock = 16;
  dim3 A_blocksPerGrid  (ceil( float(m) / threadsPerBlock ));
  dim3 V_blocksPerGrid  (ceil( float(n) / threadsPerBlock ));

  // Initializing V = 1
  if (jobv == AllVec) {
    for (size_t i = 0; i < n; ++i) {
      V.elements[iteratorC(i, i, ldv)] = 1.0;
    }
  } else if (jobv == SomeVec) {
    for (size_t i = 0; i < std::min(m, n); ++i) {
      V.elements[iteratorC(i, i, ldv)] = 1.0;
    }
  }

  size_t m_ordering = (n + 1) / 2;

#ifdef DEBUG
  // Report Matrix A^T * A
  std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
  for (size_t indexRow = 0; indexRow < m; ++indexRow) {
    for (size_t indexCol = 0; indexCol < n; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < m; ++k_dot){
        value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
      }
      std::cout << value << " ";
    }
    std::cout << '\n';
  }
#endif
  // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
  size_t maxIterations = 1;

  std::vector<double*> d_p_vectors(num_of_threads);
  std::vector<double*> d_q_vectors(num_of_threads);
  std::vector<double*> d_v_p_vectors(num_of_threads);
  std::vector<double*> d_v_q_vectors(num_of_threads);

  for(size_t i = 0; i < num_of_threads; i++){
    cudaMalloc(&d_p_vectors[i], m * sizeof(double));
    cudaMalloc(&d_q_vectors[i], m * sizeof(double));
    cudaMalloc(&d_v_p_vectors[i], n * sizeof(double));
    cudaMalloc(&d_v_q_vectors[i], n * sizeof(double));
  }

  for(auto number_iterations = 0; number_iterations < maxIterations; ++number_iterations){
    // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
    // 1971
    for (size_t k = 1; k < m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;

#pragma omp parallel for private(p, p_trans, q_trans)
      for (size_t q = m_ordering - k + 1; q <= n - k; ++q) {
        size_t thread_id = omp_get_thread_num();
        if (m_ordering - k + 1 <= q && q <= (2 * m_ordering) - (2 * k)) {
          p = ((2 * m_ordering) - (2 * k) + 1) - q;
        } else if ((2 * m_ordering) - (2 * k) < q && q <= (2 * m_ordering) - k - 1) {
          p = ((4 * m_ordering) - (2 * k)) - q;
        } else if ((2 * m_ordering) - k - 1 < q) {
          p = n;
        }

        // Translate to (0,0)
        p_trans = p - 1;
        q_trans = q - 1;

        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
        double tmp_p, tmp_q;
        for (size_t i = 0; i < m; ++i) {
          tmp_p = A.elements[iteratorC(i, p_trans, lda)];
          tmp_q = A.elements[iteratorC(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // Schur
        double c_schur = 1.0, s_schur = 0.0, aqq = gamma, app = beta, apq = alpha;

        if (abs(apq) > 1e-20) {
          double tau = (aqq - app) / (2.0 * apq);
          double t = 0.0;

          if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1 + (tau * tau)));
          } else {
            t = 1.0 / (tau - sqrt(1 + (tau * tau)));
          }

          c_schur = 1.0 / sqrt(1 + (t * t));
          s_schur = t * c_schur;

          cudaMemcpy(d_p_vectors[thread_id], (A.elements + p_trans*lda), m * sizeof(double),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(d_q_vectors[thread_id], (A.elements + q_trans*lda), m * sizeof(double),
                     cudaMemcpyHostToDevice);

//          CUDAMatrix d_p_vector(, m), d_q_vector((A.elements + q_trans*lda), m);

          jacobi_rotation<<<A_blocksPerGrid, threadsPerBlock>>>(m, d_p_vectors[thread_id], d_q_vectors[thread_id], c_schur, s_schur);

          cudaMemcpy((A.elements + p_trans*lda), d_p_vectors[thread_id],m * sizeof(double),
                     cudaMemcpyDeviceToHost);
          cudaMemcpy((A.elements + q_trans*lda), d_q_vectors[thread_id],m * sizeof(double),
                     cudaMemcpyDeviceToHost);
//          d_p_vector.copy_to_host((A.elements + p_trans*lda), m);
//          d_q_vector.copy_to_host((A.elements + q_trans*lda), m);
//
//          d_p_vector.free();
//          d_q_vector.free();

//          if (jobv == AllVec || jobv == SomeVec) {
//            CUDAMatrix d_v_p_vector((V.elements + p_trans*lda), n), d_v_q_vector((V.elements + q_trans*lda), n);

          cudaMemcpy(d_v_p_vectors[thread_id], (V.elements + p_trans*ldv), n * sizeof(double),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(d_v_q_vectors[thread_id], (V.elements + q_trans*ldv), n * sizeof(double),
                     cudaMemcpyHostToDevice);
          jacobi_rotation<<<V_blocksPerGrid, threadsPerBlock>>>(n, d_v_p_vectors[thread_id], d_v_q_vectors[thread_id], c_schur, s_schur);

          cudaMemcpy((V.elements + p_trans*ldv), d_v_p_vectors[thread_id],n * sizeof(double),
                     cudaMemcpyDeviceToHost);
          cudaMemcpy((V.elements + q_trans*ldv), d_v_q_vectors[thread_id],n * sizeof(double),
                     cudaMemcpyDeviceToHost);
//            d_v_p_vector.copy_to_host((V.elements + p_trans*lda), n);
//            d_v_q_vector.copy_to_host((V.elements + q_trans*lda), n);

//            d_v_p_vector.free();
//            d_v_q_vector.free();
//          }
        }
      }
    }

    for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
      size_t thread_id = omp_thread_count();
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;
#pragma omp parallel for private(p, p_trans, q_trans)
      for (size_t q = (4 * m_ordering) - n - k; q < (3 * m_ordering) - k; ++q) {
        if (q < (2 * m_ordering) - k + 1) {
          p = n;
        } else if ((2 * m_ordering) - k + 1 <= q && q <= (4 * m_ordering) - (2 * k) - 1) {
          p = ((4 * m_ordering) - (2 * k)) - q;
        } else if ((4 * m_ordering) - (2 * k) - 1 < q) {
          p = ((6 * m_ordering) - (2 * k) - 1) - q;
        }

        // Translate to (0,0)
        p_trans = p - 1;
        q_trans = q - 1;

        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
        double tmp_p, tmp_q;
        for (size_t i = 0; i < m; ++i) {
          tmp_p = A.elements[iteratorC(i, p_trans, lda)];
          tmp_q = A.elements[iteratorC(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // Schur
        double c_schur = 1.0, s_schur = 0.0, aqq = gamma, app = beta, apq = alpha;

        if (abs(apq) > 1e-20) {
          double tau = (aqq - app) / (2.0 * apq);
          double t = 0.0;

          if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1 + (tau * tau)));
          } else {
            t = 1.0 / (tau - sqrt(1 + (tau * tau)));
          }

          c_schur = 1.0 / sqrt(1 + (t * t));
          s_schur = t * c_schur;

          cudaMemcpy(d_p_vectors[thread_id], (A.elements + p_trans*lda), m * sizeof(double),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(d_q_vectors[thread_id], (A.elements + q_trans*lda), m * sizeof(double),
                     cudaMemcpyHostToDevice);

//          CUDAMatrix d_p_vector(, m), d_q_vector((A.elements + q_trans*lda), m);

          jacobi_rotation<<<A_blocksPerGrid, threadsPerBlock>>>(m, d_p_vectors[thread_id], d_q_vectors[thread_id], c_schur, s_schur);

          cudaMemcpy((A.elements + p_trans*lda), d_p_vectors[thread_id],m * sizeof(double),
                     cudaMemcpyDeviceToHost);
          cudaMemcpy((A.elements + q_trans*lda), d_q_vectors[thread_id],m * sizeof(double),
                     cudaMemcpyDeviceToHost);
//          d_p_vector.copy_to_host((A.elements + p_trans*lda), m);
//          d_q_vector.copy_to_host((A.elements + q_trans*lda), m);
//
//          d_p_vector.free();
//          d_q_vector.free();

//          if (jobv == AllVec || jobv == SomeVec) {
//            CUDAMatrix d_v_p_vector((V.elements + p_trans*lda), n), d_v_q_vector((V.elements + q_trans*lda), n);

          cudaMemcpy(d_v_p_vectors[thread_id], (V.elements + p_trans*ldv), n * sizeof(double),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(d_v_q_vectors[thread_id], (V.elements + q_trans*ldv), n * sizeof(double),
                     cudaMemcpyHostToDevice);
          jacobi_rotation<<<V_blocksPerGrid, threadsPerBlock>>>(m, d_v_p_vectors[thread_id], d_v_q_vectors[thread_id], c_schur, s_schur);

          cudaMemcpy((V.elements + p_trans*ldv), d_v_p_vectors[thread_id],n * sizeof(double),
                     cudaMemcpyDeviceToHost);
          cudaMemcpy((V.elements + q_trans*ldv), d_v_q_vectors[thread_id],n * sizeof(double),
                     cudaMemcpyDeviceToHost);
//            d_v_p_vector.copy_to_host((V.elements + p_trans*lda), n);
//            d_v_q_vector.copy_to_host((V.elements + q_trans*lda), n);

//            d_v_p_vector.free();
//            d_v_q_vector.free();
//          }
        }
      }
    }
  }

  for(size_t i = 0; i < num_of_threads; i++){
    cudaFree(d_p_vectors[i]);
    cudaFree(d_q_vectors[i]);
    cudaFree(d_v_p_vectors[i]);
    cudaFree(d_v_q_vectors[i]);
  }

  std::cout << "How many repetitions?: " << maxIterations << "\n";

  // Compute \Sigma
#pragma omp parallel for
  for (size_t k = 0; k < std::min(m, n); ++k) {
    for (size_t i = 0; i < m; ++i) {
      s.elements[k] += A.elements[iteratorC(i, k, lda)] * A.elements[iteratorC(i, k, lda)];
    }
    s.elements[k] = sqrt(s.elements[k]);
  }

  //Compute U
  if (jobu == AllVec) {
#pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < m; ++j) {
        A.elements[iteratorC(j, i, lda)] = A.elements[iteratorC(j, i, lda)] / s.elements[i];
      }
    }
  } else if (jobu == SomeVec) {
#pragma omp parallel for
    for (size_t k = 0; k < std::min(m, n); ++k) {
      for (size_t i = 0; i < m; ++i) {
        A.elements[iteratorC(i, k, lda)] = A.elements[iteratorC(i, k, lda)] / s.elements[k];
      }
    }
  }

//  delete []ordering_array;
}


void omp_mpi_cuda_dgesvd_local_matrices(SVD_OPTIONS jobu,
                                        SVD_OPTIONS jobv,
                                        size_t m,
                                        size_t n,
                                        MatrixMPI &A,
                                        size_t lda,
                                        MatrixMPI &s,
                                        MatrixMPI &V,
                                        size_t ldv) {
  int threadsPerBlock = 16;
  dim3 A_blocksPerGrid  (ceil( float(m) / threadsPerBlock ));
  dim3 V_blocksPerGrid  (ceil( float(n) / threadsPerBlock ));

  // Get iterator
  size_t num_of_threads = omp_thread_count();

  // Get rank of mpi proccess and size of process
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == ROOT_RANK) {
    // Initializing V = 1
    if (jobv == AllVec) {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        V.elements[iteratorC(i, i, ldv)] = 1.0;
      }
    } else if (jobv == SomeVec) {
#pragma omp parallel for
      for (size_t i = 0; i < std::min(m, n); ++i) {
        V.elements[iteratorC(i, i, ldv)] = 1.0;
      }
    }
  }

  size_t m_ordering = (n + 1) / 2;
  size_t k_ordering_len = n / 2;
  size_t number_of_columns = 2 * ( n/2);
  size_t number_of_last_columns = k_ordering_len % size;
  size_t number_of_columns_except_last = k_ordering_len / size;
  // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
  size_t maxIterations = 1;

  std::vector<double*> d_p_vectors(num_of_threads);
  std::vector<double*> d_q_vectors(num_of_threads);
  std::vector<double*> d_v_p_vectors(num_of_threads);
  std::vector<double*> d_v_q_vectors(num_of_threads);

  for(size_t i = 0; i < num_of_threads; i++){
    cudaMalloc(&d_p_vectors[i], m * sizeof(double));
    cudaMalloc(&d_q_vectors[i], m * sizeof(double));
    cudaMalloc(&d_v_p_vectors[i], n * sizeof(double));
    cudaMalloc(&d_v_q_vectors[i], n * sizeof(double));
  }

  for(auto number_iterations = 0; number_iterations < maxIterations; ++number_iterations){

    // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
    // 1971
    for (size_t k = 1; k < m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;

      /* --------------------------------------- Local variables ----------------------------------------------*/
      // All points vector tuple
      std::vector<std::tuple<size_t, size_t>> local_points;
      // Ordered set of coordinates
      std::set<size_t> local_set_columns;
      // Set converted to vector
      std::vector<size_t> local_set_to_vector;
      // Local map that convert point coordinate to local column index
      std::unordered_map<size_t, size_t> column_index_to_local_column_index;
      /* --------------------------------------- Root rank variables ----------------------------------------------*/
      // Distribution of points by node (index is rank, and content is vector of points). To use for data distribution and extraction.
      std::vector<std::vector<std::tuple<size_t, size_t>>> root_distribution(size);
      // ordered set of points by rank. To use for data distribution and extraction.
      std::vector<std::set<size_t>> root_set_columns_by_rank(size);
      // convert to vector to map
      std::vector<std::vector<size_t>> root_set_columns_vector_by_rank(size);
      // Assign column index to set index.
      std::vector<std::unordered_map<size_t, size_t>> root_column_index_to_local_column_index(size);


      /* ------------------------------------- Query points for k -------------------------------------------------*/
#pragma omp parallel for num_threads(size) private(p, p_trans, q_trans)
      for (size_t q = m_ordering - k + 1; q <= n - k; ++q) {
        if (m_ordering - k + 1 <= q && q <= (2 * m_ordering) - (2 * k)) {
          p = ((2 * m_ordering) - (2 * k) + 1) - q;
        } else if ((2 * m_ordering) - (2 * k) < q && q <= (2 * m_ordering) - k - 1) {
          p = ((4 * m_ordering) - (2 * k)) - q;
        } else if ((2 * m_ordering) - k - 1 < q) {
          p = n;
        }

        // Translate to (0,0)
        p_trans = p - 1;
        q_trans = q - 1;

        if(rank == ROOT_RANK){
//          std::cout << "(" << p_trans << ", " << q_trans << ")\n";
//          std::cout << "(" << omp_get_thread_num() << ")\n";
          root_distribution[omp_get_thread_num()].emplace_back(p_trans, q_trans);
          root_set_columns_by_rank[omp_get_thread_num()].insert(p_trans);
          root_set_columns_by_rank[omp_get_thread_num()].insert(q_trans);
        }

        if(rank == omp_get_thread_num()){
//          std::cout << "local point rank: " << rank << ", (" << p_trans << ", " << q_trans << ")\n";
          local_points.emplace_back(p_trans,q_trans);
          local_set_columns.insert(p_trans);
          local_set_columns.insert(q_trans);
        }
      }

      // convert local set to vector
      local_set_to_vector = std::vector<size_t>(local_set_columns.begin(), local_set_columns.end());
      // map coordinates to local column indices
      size_t local_set_to_vector_size = local_set_to_vector.size();
      for(auto i = 0; i < local_set_to_vector_size; ++i){
        column_index_to_local_column_index[local_set_to_vector[i]] = i;
      }

      if(rank == ROOT_RANK){
        for(auto index_rank = 0; index_rank < size; ++index_rank){
          root_set_columns_vector_by_rank[index_rank] = std::vector<size_t>(root_set_columns_by_rank[index_rank].begin(), root_set_columns_by_rank[index_rank].end());
          size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
          for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
            root_column_index_to_local_column_index[index_rank][root_set_columns_vector_by_rank[index_rank][index_column]] = index_column;
          }
        }
      } else {
        for(auto i = 0; i < local_set_to_vector.size(); ++i){
//          std::cout << "rank: " << rank << ", column_index_to_local_column_index[column_index]: " << column_index_to_local_column_index[i] << ", column index: " << i << '\n';
        }
      }

      // Create local matrix
      MatrixMPI A_local(m, local_set_columns.size());
      MatrixMPI V_local(m, local_set_columns.size());

      /* ------------------------------------- Distribute A -------------------------------------------------*/
      if(rank == ROOT_RANK){

        // Create matrix by rank and send
        for(auto index_rank = 1; index_rank < size; ++index_rank){
          std::vector<double> tmp_matrix;
          for(auto column_index: root_set_columns_vector_by_rank[index_rank]){
            tmp_matrix.insert(tmp_matrix.end(), A.elements + m*column_index, A.elements + (m*(column_index + 1)));
          }
/*

//          std::cout << "send rank: " << index_rank << ", local matrix size: " << tmp_matrix.size() << ", expected matrix size: " << m * root_set_columns_by_rank[index_rank].size() << '\n';
//
//          for(auto index_row = 0; index_row < m; ++index_row){
//            size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
//            for(auto index_col = 0; index_col < root_set_columns_vector_by_rank_for_index_rank_size; ++index_col){
//              std::cout << tmp_matrix[iteratorC(index_row, index_col, m)] << ", ";
//            }
//            std::cout << "\n";
//          }
*/
          MatrixMPI A_rank(tmp_matrix, m, root_set_columns_by_rank[index_rank].size());

          tmp_matrix.clear();
          auto return_status = MPI_Send(A_rank.elements, m * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
          }

          A_rank.free();
        }
      } else {
        MPI_Status status;
        auto return_status = MPI_Recv(A_local.elements, m * local_set_columns.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
        }
      }
/*
      std::cout << "receive rank: " << rank << ", local matrix size: " << A_local.height * A_local.width << ", expected matrix size: " << m * local_set_columns.size() << '\n';

      for(auto index_row = 0; index_row < m; ++index_row){
        for(auto index_col = 0; index_col < local_set_columns.size(); ++index_col){
          std::cout << A_local.elements[iteratorC(index_row, index_col, m)] << ", ";
        }
        std::cout << "\n";
      }
 */
/*

      // Calculate frobenius norm of A_local - A
      double frobenius_norm = 0.0;
      // Iterate columns assigned to this rank
      for(auto column_index: local_set_columns){
        for(size_t index_row = 0; index_row < m; ++index_row){
          double sustraction = (A_local.elements[iteratorC(index_row, column_index_to_local_column_index[column_index], m)] - A_test.elements[iteratorC(index_row, column_index, m)]);
          frobenius_norm += sustraction * sustraction;
        }
      }

      std::cout << "||A-USVt||_F: " << sqrt(frobenius_norm) << "\n";
*/
      /* ------------------------------------- Distribute V -------------------------------------------------*/
      if(rank == ROOT_RANK){
        // Create matrix by rank and send
        for(auto index_rank = 1; index_rank < size; ++index_rank){
          std::vector<double> tmp_matrix;
          for(auto column_index: root_set_columns_vector_by_rank[index_rank]){
            tmp_matrix.insert(tmp_matrix.end(), V.elements + m*column_index, V.elements + (m*(column_index + 1)));
          }
/*

//          std::cout << "send rank: " << index_rank << ", local matrix size: " << tmp_matrix.size() << ", expected matrix size: " << m * root_set_columns_by_rank[index_rank].size() << '\n';
//
//          for(auto index_row = 0; index_row < m; ++index_row){
//            size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
//            for(auto index_col = 0; index_col < root_set_columns_vector_by_rank_for_index_rank_size; ++index_col){
//              std::cout << tmp_matrix[iteratorC(index_row, index_col, m)] << ", ";
//            }
//            std::cout << "\n";
//          }
*/
          MatrixMPI V_rank(tmp_matrix, m, root_set_columns_by_rank[index_rank].size());

          tmp_matrix.clear();
          auto return_status = MPI_Send(V_rank.elements, m * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
          }

          V_rank.free();
        }
      } else {
        MPI_Status status;
        auto return_status = MPI_Recv(V_local.elements, m * local_set_columns.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
        }
      }


      /* ------------------------------------- Solve local problem -------------------------------------------------*/

      if(rank == ROOT_RANK){
        size_t local_points_size = local_points.size();
#pragma omp parallel for
        for(size_t index_local_points = 0; index_local_points < local_points_size; ++index_local_points){
          size_t thread_id = omp_get_thread_num();
          // Extract point
          std::tuple<size_t, size_t> point = local_points[index_local_points];

          size_t p_point = std::get<0>(point);
          size_t q_point = std::get<1>(point);

//        std::cout << "p: " << p_point << ", q:" << q_point << ", p_transform: " << p_point_to_local_index << ", q_transform: " << q_point_to_local_index << '\n';

          double alpha = 0.0, beta = 0.0, gamma = 0.0;
          // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
          double tmp_p, tmp_q;
          for (size_t i = 0; i < m; ++i) {
            tmp_p = A.elements[iteratorC(i, p_point, lda)];
            tmp_q = A.elements[iteratorC(i, q_point, lda)];
            alpha += tmp_p * tmp_q;
            beta += tmp_p * tmp_p;
            gamma += tmp_q * tmp_q;
          }

          // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
          double convergence_value = std::abs(alpha) / sqrt(beta * gamma);
          // Schur
          double c_schur = 1.0, s_schur = 0.0;

          if(std::abs(alpha) > TOLERANCE){
            double tau = (gamma - beta) / (2.0 * alpha);
            double t = 0.0;

            if (tau >= 0) {
              t = 1.0 / (tau + sqrt(1 + (tau * tau)));
            } else {
              t = 1.0 / (tau - sqrt(1 + (tau * tau)));
            }

            c_schur = 1.0 / sqrt(1 + (t * t));
            s_schur = t * c_schur;

            cudaMemcpy(d_p_vectors[thread_id], (A.elements + p_point*lda), m * sizeof(double),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_q_vectors[thread_id], (A.elements + q_point*lda), m * sizeof(double),
                       cudaMemcpyHostToDevice);

//          CUDAMatrix d_p_vector(, m), d_q_vector((A.elements + q_trans*lda), m);

            jacobi_rotation<<<A_blocksPerGrid, threadsPerBlock>>>(m, d_p_vectors[thread_id], d_q_vectors[thread_id], c_schur, s_schur);

            cudaMemcpy((A.elements + p_point*lda), d_p_vectors[thread_id],m * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy((A.elements + q_point*lda), d_q_vectors[thread_id],m * sizeof(double),
                       cudaMemcpyDeviceToHost);
//          d_p_vector.copy_to_host((A.elements + p_trans*lda), m);
//          d_q_vector.copy_to_host((A.elements + q_trans*lda), m);
//
//          d_p_vector.free();
//          d_q_vector.free();

//          if (jobv == AllVec || jobv == SomeVec) {
//            CUDAMatrix d_v_p_vector((V.elements + p_trans*lda), n), d_v_q_vector((V.elements + q_trans*lda), n);

            cudaMemcpy(d_v_p_vectors[thread_id], (V.elements + p_point*ldv), n * sizeof(double),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_v_q_vectors[thread_id], (V.elements + q_point*ldv), n * sizeof(double),
                       cudaMemcpyHostToDevice);
            jacobi_rotation<<<V_blocksPerGrid, threadsPerBlock>>>(n, d_v_p_vectors[thread_id], d_v_q_vectors[thread_id], c_schur, s_schur);

            cudaMemcpy((V.elements + p_point*ldv), d_v_p_vectors[thread_id],n * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy((V.elements + q_point*ldv), d_v_q_vectors[thread_id],n * sizeof(double),
                       cudaMemcpyDeviceToHost);
//            d_v_p_vector.copy_to_host((V.elements + p_trans*lda), n);
//            d_v_q_vector.copy_to_host((V.elements + q_trans*lda), n);

//            d_v_p_vector.free();
//            d_v_q_vector.free();
//          }
          }
        }
      } else {
        size_t local_points_size = local_points.size();
#pragma omp parallel for
        for(size_t index_local_points = 0; index_local_points < local_points_size; ++index_local_points){
          size_t thread_id = omp_get_thread_num();
          // Extract point
          std::tuple<size_t, size_t> point = local_points[index_local_points];

          size_t p_point = std::get<0>(point);
          size_t q_point = std::get<1>(point);

          size_t p_point_to_local_index = column_index_to_local_column_index[p_point];
          size_t q_point_to_local_index = column_index_to_local_column_index[q_point];

//        std::cout << "p: " << p_point << ", q:" << q_point << ", p_transform: " << p_point_to_local_index << ", q_transform: " << q_point_to_local_index << '\n';

          double alpha = 0.0, beta = 0.0, gamma = 0.0;
          // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
          double tmp_p, tmp_q;
          for (size_t i = 0; i < m; ++i) {
            tmp_p = A_local.elements[iteratorC(i, p_point_to_local_index, lda)];
            tmp_q = A_local.elements[iteratorC(i, q_point_to_local_index, lda)];
            alpha += tmp_p * tmp_q;
            beta += tmp_p * tmp_p;
            gamma += tmp_q * tmp_q;
          }

          // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
          double convergence_value = std::abs(alpha) / sqrt(beta * gamma);
          // Schur
          double c_schur = 1.0, s_schur = 0.0;

          if(std::abs(alpha) > TOLERANCE){
            double tau = (gamma - beta) / (2.0 * alpha);
            double t = 0.0;

            if (tau >= 0) {
              t = 1.0 / (tau + sqrt(1 + (tau * tau)));
            } else {
              t = 1.0 / (tau - sqrt(1 + (tau * tau)));
            }

            c_schur = 1.0 / sqrt(1 + (t * t));
            s_schur = t * c_schur;

            cudaMemcpy(d_p_vectors[thread_id], (A_local.elements + p_point_to_local_index*lda), m * sizeof(double),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_q_vectors[thread_id], (A_local.elements + q_point_to_local_index*lda), m * sizeof(double),
                       cudaMemcpyHostToDevice);

//          CUDAMatrix d_p_vector(, m), d_q_vector((A.elements + q_trans*lda), m);

            jacobi_rotation<<<A_blocksPerGrid, threadsPerBlock>>>(m, d_p_vectors[thread_id], d_q_vectors[thread_id], c_schur, s_schur);

            cudaMemcpy((A_local.elements + p_point_to_local_index*lda), d_p_vectors[thread_id],m * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy((A_local.elements + q_point_to_local_index*lda), d_q_vectors[thread_id],m * sizeof(double),
                       cudaMemcpyDeviceToHost);
//          d_p_vector.copy_to_host((A.elements + p_trans*lda), m);
//          d_q_vector.copy_to_host((A.elements + q_trans*lda), m);
//
//          d_p_vector.free();
//          d_q_vector.free();

//          if (jobv == AllVec || jobv == SomeVec) {
//            CUDAMatrix d_v_p_vector((V.elements + p_trans*lda), n), d_v_q_vector((V.elements + q_trans*lda), n);

            cudaMemcpy(d_v_p_vectors[thread_id], (V_local.elements + p_point_to_local_index*ldv), n * sizeof(double),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_v_q_vectors[thread_id], (V_local.elements + q_point_to_local_index*ldv), n * sizeof(double),
                       cudaMemcpyHostToDevice);
            jacobi_rotation<<<V_blocksPerGrid, threadsPerBlock>>>(n, d_v_p_vectors[thread_id], d_v_q_vectors[thread_id], c_schur, s_schur);

            cudaMemcpy((V_local.elements + p_point_to_local_index*ldv), d_v_p_vectors[thread_id],n * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy((V_local.elements + q_point_to_local_index*ldv), d_v_q_vectors[thread_id],n * sizeof(double),
                       cudaMemcpyDeviceToHost);
//            d_v_p_vector.copy_to_host((V.elements + p_trans*lda), n);
//            d_v_q_vector.copy_to_host((V.elements + q_trans*lda), n);

//            d_v_p_vector.free();
//            d_v_q_vector.free();
//          }
          }
        }
      }

      /* ----------------------------------- Gather local A solutions ------------------------------------------------*/
      if(rank == ROOT_RANK){
        for(size_t index_rank = 1; index_rank < size; ++index_rank){
          // Create local matrix
          MatrixMPI A_gather(m, root_set_columns_by_rank[index_rank].size());

          MPI_Status status;
          auto return_status = MPI_Recv(A_gather.elements, m * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD, &status);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
          }

          /* ------------------------------------- Check distribution receive difference -------------------------------------------------*/
          /*// Calculate frobenius norm of A_local - A
          double frobenius_norm_received = 0.0;
          // Iterate columns assigned to this rank
          for(auto column_index: root_set_columns_by_rank[index_rank]){
            for(size_t index_row = 0; index_row < m; ++index_row){
              double sustraction = (A_gather.elements[iteratorC(index_row, root_column_index_to_local_column_index[index_rank][column_index], m)] - A.elements[iteratorC(index_row, column_index, m)]);
              frobenius_norm_received += sustraction * sustraction;
            }
          }

          std::cout << "Gather ||A-USVt||_F: " << sqrt(frobenius_norm_received) << "\n";*/

          size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
#pragma omp parallel for
          for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
            for(size_t index_row = 0; index_row < m; ++index_row){
              A.elements[iteratorC(index_row, root_set_columns_vector_by_rank[index_rank][index_column], lda)] = A_gather.elements[iteratorC(index_row, index_column, lda)];
            }
          }

          A_gather.free();
        }
      } else {
        auto return_status = MPI_Send(A_local.elements, m * local_set_columns.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
        }
      }

      /* ----------------------------------- Gather local V solutions ------------------------------------------------*/
      if(rank == ROOT_RANK){
        for(size_t index_rank = 1; index_rank < size; ++index_rank){
          // Create local matrix
          MatrixMPI V_gather(m, root_set_columns_by_rank[index_rank].size());

          MPI_Status status;
          auto return_status = MPI_Recv(V_gather.elements, n * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD, &status);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
          }

          /* ------------------------------------- Check distribution receive difference -------------------------------------------------*/
          /*// Calculate frobenius norm of A_local - A
          double frobenius_norm_received = 0.0;
          // Iterate columns assigned to this rank
          for(auto column_index: root_set_columns_by_rank[index_rank]){
            for(size_t index_row = 0; index_row < m; ++index_row){
              double sustraction = (A_gather.elements[iteratorC(index_row, root_column_index_to_local_column_index[index_rank][column_index], m)] - A.elements[iteratorC(index_row, column_index, m)]);
              frobenius_norm_received += sustraction * sustraction;
            }
          }

          std::cout << "Gather ||A-USVt||_F: " << sqrt(frobenius_norm_received) << "\n";*/

          size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
#pragma omp parallel for
          for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
            for(size_t index_row = 0; index_row < n; ++index_row){
              V.elements[iteratorC(index_row, root_set_columns_vector_by_rank[index_rank][index_column], lda)] = V_gather.elements[iteratorC(index_row, index_column, lda)];
            }
          }

          V_gather.free();
        }
      } else {
        auto return_status = MPI_Send(V_local.elements, n * local_set_columns.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
        }
      }

      // Free local matrix
      A_local.free();
      V_local.free();

      MPI_Barrier(MPI_COMM_WORLD);
    }

    for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;

      /* --------------------------------------- Local variables ----------------------------------------------*/
      // All points vector tuple
      std::vector<std::tuple<size_t, size_t>> local_points;
      // Ordered set of coordinates
      std::set<size_t> local_set_columns;
      // Set converted to vector
      std::vector<size_t> local_set_to_vector;
      // Local map that convert point coordinate to local column index
      std::unordered_map<size_t, size_t> column_index_to_local_column_index;
      /* --------------------------------------- Root rank variables ----------------------------------------------*/
      // Distribution of points by node (index is rank, and content is vector of points). To use for data distribution and extraction.
      std::vector<std::vector<std::tuple<size_t, size_t>>> root_distribution(size);
      // ordered set of points by rank. To use for data distribution and extraction.
      std::vector<std::set<size_t>> root_set_columns_by_rank(size);
      // convert to vector to map
      std::vector<std::vector<size_t>> root_set_columns_vector_by_rank(size);
      // Assign column index to set index.
      std::vector<std::unordered_map<size_t, size_t>> root_column_index_to_local_column_index(size);


      /* ------------------------------------- Query points for k -------------------------------------------------*/
#pragma omp parallel for num_threads(size) private(p, p_trans, q_trans)
      for (size_t q = (4 * m_ordering) - n - k; q < (3 * m_ordering) - k; ++q) {
        if (q < (2 * m_ordering) - k + 1) {
          p = n;
        } else if ((2 * m_ordering) - k + 1 <= q && q <= (4 * m_ordering) - (2 * k) - 1) {
          p = ((4 * m_ordering) - (2 * k)) - q;
        } else if ((4 * m_ordering) - (2 * k) - 1 < q) {
          p = ((6 * m_ordering) - (2 * k) - 1) - q;
        }

        // Translate to (0,0)
        p_trans = p - 1;
        q_trans = q - 1;

        if(rank == ROOT_RANK){
//          std::cout << "(" << p_trans << ", " << q_trans << ")\n";
//          std::cout << "(" << omp_get_thread_num() << ")\n";
          root_distribution[omp_get_thread_num()].emplace_back(p_trans, q_trans);
          root_set_columns_by_rank[omp_get_thread_num()].insert(p_trans);
          root_set_columns_by_rank[omp_get_thread_num()].insert(q_trans);
        }

        if(rank == omp_get_thread_num()){
//          std::cout << "local point rank: " << rank << ", (" << p_trans << ", " << q_trans << ")\n";
          local_points.emplace_back(p_trans,q_trans);
          local_set_columns.insert(p_trans);
          local_set_columns.insert(q_trans);
        }
      }

      // convert local set to vector
      local_set_to_vector = std::vector<size_t>(local_set_columns.begin(), local_set_columns.end());
      // map coordinates to local column indices
      size_t local_set_to_vector_size = local_set_to_vector.size();
      for(auto i = 0; i < local_set_to_vector_size; ++i){
        column_index_to_local_column_index[local_set_to_vector[i]] = i;
      }

      if(rank == ROOT_RANK){
        for(auto index_rank = 0; index_rank < size; ++index_rank){
          root_set_columns_vector_by_rank[index_rank] = std::vector<size_t>(root_set_columns_by_rank[index_rank].begin(), root_set_columns_by_rank[index_rank].end());
          size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
          for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
            root_column_index_to_local_column_index[index_rank][root_set_columns_vector_by_rank[index_rank][index_column]] = index_column;
          }
        }
      }

      // Create local matrix
      MatrixMPI A_local(m, local_set_columns.size());
      MatrixMPI V_local(m, local_set_columns.size());

      /* ------------------------------------- Distribute A -------------------------------------------------*/
      if(rank == ROOT_RANK){

        // Create matrix by rank and send
        for(auto index_rank = 1; index_rank < size; ++index_rank){
          std::vector<double> tmp_matrix;
          for(auto column_index: root_set_columns_vector_by_rank[index_rank]){
            tmp_matrix.insert(tmp_matrix.end(), A.elements + m*column_index, A.elements + (m*(column_index + 1)));
          }
/*

//          std::cout << "send rank: " << index_rank << ", local matrix size: " << tmp_matrix.size() << ", expected matrix size: " << m * root_set_columns_by_rank[index_rank].size() << '\n';
//
//          for(auto index_row = 0; index_row < m; ++index_row){
//            size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
//            for(auto index_col = 0; index_col < root_set_columns_vector_by_rank_for_index_rank_size; ++index_col){
//              std::cout << tmp_matrix[iteratorC(index_row, index_col, m)] << ", ";
//            }
//            std::cout << "\n";
//          }
*/
          MatrixMPI A_rank(tmp_matrix, m, root_set_columns_by_rank[index_rank].size());

          tmp_matrix.clear();
          auto return_status = MPI_Send(A_rank.elements, m * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
          }

          A_rank.free();
        }
      } else {
        MPI_Status status;
        auto return_status = MPI_Recv(A_local.elements, m * local_set_columns.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
        }
      }
/*
      std::cout << "receive rank: " << rank << ", local matrix size: " << A_local.height * A_local.width << ", expected matrix size: " << m * local_set_columns.size() << '\n';

      for(auto index_row = 0; index_row < m; ++index_row){
        for(auto index_col = 0; index_col < local_set_columns.size(); ++index_col){
          std::cout << A_local.elements[iteratorC(index_row, index_col, m)] << ", ";
        }
        std::cout << "\n";
      }
 */
/*

      // Calculate frobenius norm of A_local - A
      double frobenius_norm = 0.0;
      // Iterate columns assigned to this rank
      for(auto column_index: local_set_columns){
        for(size_t index_row = 0; index_row < m; ++index_row){
          double sustraction = (A_local.elements[iteratorC(index_row, column_index_to_local_column_index[column_index], m)] - A_test.elements[iteratorC(index_row, column_index, m)]);
          frobenius_norm += sustraction * sustraction;
        }
      }

      std::cout << "||A-USVt||_F: " << sqrt(frobenius_norm) << "\n";
*/
      /* ------------------------------------- Distribute V -------------------------------------------------*/
      if(rank == ROOT_RANK){
        // Create matrix by rank and send
        for(auto index_rank = 1; index_rank < size; ++index_rank){
          std::vector<double> tmp_matrix;
          for(auto column_index: root_set_columns_vector_by_rank[index_rank]){
            tmp_matrix.insert(tmp_matrix.end(), V.elements + m*column_index, V.elements + (m*(column_index + 1)));
          }
/*

//          std::cout << "send rank: " << index_rank << ", local matrix size: " << tmp_matrix.size() << ", expected matrix size: " << m * root_set_columns_by_rank[index_rank].size() << '\n';
//
//          for(auto index_row = 0; index_row < m; ++index_row){
//            size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
//            for(auto index_col = 0; index_col < root_set_columns_vector_by_rank_for_index_rank_size; ++index_col){
//              std::cout << tmp_matrix[iteratorC(index_row, index_col, m)] << ", ";
//            }
//            std::cout << "\n";
//          }
*/
          MatrixMPI V_rank(tmp_matrix, m, root_set_columns_by_rank[index_rank].size());

          tmp_matrix.clear();
          auto return_status = MPI_Send(V_rank.elements, n * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
          }

          V_rank.free();
        }
      } else {
        MPI_Status status;
        auto return_status = MPI_Recv(V_local.elements, n * local_set_columns.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
        }
      }


      /* ------------------------------------- Solve local problem -------------------------------------------------*/

      if(rank == ROOT_RANK){
        size_t local_points_size = local_points.size();
#pragma omp parallel for
        for(size_t index_local_points = 0; index_local_points < local_points_size; ++index_local_points){
          size_t thread_id = omp_get_thread_num();
          // Extract point
          std::tuple<size_t, size_t> point = local_points[index_local_points];

          size_t p_point = std::get<0>(point);
          size_t q_point = std::get<1>(point);

//        std::cout << "p: " << p_point << ", q:" << q_point << ", p_transform: " << p_point_to_local_index << ", q_transform: " << q_point_to_local_index << '\n';

          double alpha = 0.0, beta = 0.0, gamma = 0.0;
          // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
          double tmp_p, tmp_q;
          for (size_t i = 0; i < m; ++i) {
            tmp_p = A.elements[iteratorC(i, p_point, lda)];
            tmp_q = A.elements[iteratorC(i, q_point, lda)];
            alpha += tmp_p * tmp_q;
            beta += tmp_p * tmp_p;
            gamma += tmp_q * tmp_q;
          }

          // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
          double convergence_value = std::abs(alpha) / sqrt(beta * gamma);
          // Schur
          double c_schur = 1.0, s_schur = 0.0;

          if(std::abs(alpha) > TOLERANCE){
            double tau = (gamma - beta) / (2.0 * alpha);
            double t = 0.0;

            if (tau >= 0) {
              t = 1.0 / (tau + sqrt(1 + (tau * tau)));
            } else {
              t = 1.0 / (tau - sqrt(1 + (tau * tau)));
            }

            c_schur = 1.0 / sqrt(1 + (t * t));
            s_schur = t * c_schur;

            cudaMemcpy(d_p_vectors[thread_id], (A.elements + p_point*lda), m * sizeof(double),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_q_vectors[thread_id], (A.elements + q_point*lda), m * sizeof(double),
                       cudaMemcpyHostToDevice);

//          CUDAMatrix d_p_vector(, m), d_q_vector((A.elements + q_trans*lda), m);

            jacobi_rotation<<<A_blocksPerGrid, threadsPerBlock>>>(m, d_p_vectors[thread_id], d_q_vectors[thread_id], c_schur, s_schur);

            cudaMemcpy((A.elements + p_point*lda), d_p_vectors[thread_id],m * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy((A.elements + q_point*lda), d_q_vectors[thread_id],m * sizeof(double),
                       cudaMemcpyDeviceToHost);
//          d_p_vector.copy_to_host((A.elements + p_trans*lda), m);
//          d_q_vector.copy_to_host((A.elements + q_trans*lda), m);
//
//          d_p_vector.free();
//          d_q_vector.free();

//          if (jobv == AllVec || jobv == SomeVec) {
//            CUDAMatrix d_v_p_vector((V.elements + p_trans*lda), n), d_v_q_vector((V.elements + q_trans*lda), n);

            cudaMemcpy(d_v_p_vectors[thread_id], (V.elements + p_point*ldv), n * sizeof(double),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_v_q_vectors[thread_id], (V.elements + q_point*ldv), n * sizeof(double),
                       cudaMemcpyHostToDevice);
            jacobi_rotation<<<V_blocksPerGrid, threadsPerBlock>>>(n, d_v_p_vectors[thread_id], d_v_q_vectors[thread_id], c_schur, s_schur);

            cudaMemcpy((V.elements + p_point*ldv), d_v_p_vectors[thread_id],n * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy((V.elements + q_point*ldv), d_v_q_vectors[thread_id],n * sizeof(double),
                       cudaMemcpyDeviceToHost);
//            d_v_p_vector.copy_to_host((V.elements + p_trans*lda), n);
//            d_v_q_vector.copy_to_host((V.elements + q_trans*lda), n);

//            d_v_p_vector.free();
//            d_v_q_vector.free();
//          }
          }
        }
      } else {
        size_t local_points_size = local_points.size();
#pragma omp parallel for
        for(size_t index_local_points = 0; index_local_points < local_points_size; ++index_local_points){
          size_t thread_id = omp_get_thread_num();
          // Extract point
          std::tuple<size_t, size_t> point = local_points[index_local_points];

          size_t p_point = std::get<0>(point);
          size_t q_point = std::get<1>(point);

          size_t p_point_to_local_index = column_index_to_local_column_index[p_point];
          size_t q_point_to_local_index = column_index_to_local_column_index[q_point];

//        std::cout << "p: " << p_point << ", q:" << q_point << ", p_transform: " << p_point_to_local_index << ", q_transform: " << q_point_to_local_index << '\n';

          double alpha = 0.0, beta = 0.0, gamma = 0.0;
          // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
          double tmp_p, tmp_q;
          for (size_t i = 0; i < m; ++i) {
            tmp_p = A_local.elements[iteratorC(i, p_point_to_local_index, lda)];
            tmp_q = A_local.elements[iteratorC(i, q_point_to_local_index, lda)];
            alpha += tmp_p * tmp_q;
            beta += tmp_p * tmp_p;
            gamma += tmp_q * tmp_q;
          }

          // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
          double convergence_value = std::abs(alpha) / sqrt(beta * gamma);
          // Schur
          double c_schur = 1.0, s_schur = 0.0;

          if(std::abs(alpha) > TOLERANCE){
            double tau = (gamma - beta) / (2.0 * alpha);
            double t = 0.0;

            if (tau >= 0) {
              t = 1.0 / (tau + sqrt(1 + (tau * tau)));
            } else {
              t = 1.0 / (tau - sqrt(1 + (tau * tau)));
            }

            c_schur = 1.0 / sqrt(1 + (t * t));
            s_schur = t * c_schur;

            cudaMemcpy(d_p_vectors[thread_id], (A_local.elements + p_point_to_local_index*lda), m * sizeof(double),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_q_vectors[thread_id], (A_local.elements + q_point_to_local_index*lda), m * sizeof(double),
                       cudaMemcpyHostToDevice);

//          CUDAMatrix d_p_vector(, m), d_q_vector((A.elements + q_trans*lda), m);

            jacobi_rotation<<<A_blocksPerGrid, threadsPerBlock>>>(m, d_p_vectors[thread_id], d_q_vectors[thread_id], c_schur, s_schur);

            cudaMemcpy((A_local.elements + p_point_to_local_index*lda), d_p_vectors[thread_id],m * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy((A_local.elements + q_point_to_local_index*lda), d_q_vectors[thread_id],m * sizeof(double),
                       cudaMemcpyDeviceToHost);
//          d_p_vector.copy_to_host((A.elements + p_trans*lda), m);
//          d_q_vector.copy_to_host((A.elements + q_trans*lda), m);
//
//          d_p_vector.free();
//          d_q_vector.free();

//          if (jobv == AllVec || jobv == SomeVec) {
//            CUDAMatrix d_v_p_vector((V.elements + p_trans*lda), n), d_v_q_vector((V.elements + q_trans*lda), n);

            cudaMemcpy(d_v_p_vectors[thread_id], (V_local.elements + p_point_to_local_index*ldv), n * sizeof(double),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_v_q_vectors[thread_id], (V_local.elements + q_point_to_local_index*ldv), n * sizeof(double),
                       cudaMemcpyHostToDevice);
            jacobi_rotation<<<V_blocksPerGrid, threadsPerBlock>>>(n, d_v_p_vectors[thread_id], d_v_q_vectors[thread_id], c_schur, s_schur);

            cudaMemcpy((V_local.elements + p_point_to_local_index*ldv), d_v_p_vectors[thread_id],n * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy((V_local.elements + q_point_to_local_index*ldv), d_v_q_vectors[thread_id],n * sizeof(double),
                       cudaMemcpyDeviceToHost);
//            d_v_p_vector.copy_to_host((V.elements + p_trans*lda), n);
//            d_v_q_vector.copy_to_host((V.elements + q_trans*lda), n);

//            d_v_p_vector.free();
//            d_v_q_vector.free();
//          }
          }
        }
      }

      /* ----------------------------------- Gather local A solutions ------------------------------------------------*/
      if(rank == ROOT_RANK){
        for(size_t index_rank = 1; index_rank < size; ++index_rank){
          // Create local matrix
          MatrixMPI A_gather(m, root_set_columns_by_rank[index_rank].size());

          MPI_Status status;
          auto return_status = MPI_Recv(A_gather.elements, m * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD, &status);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
          }

          /* ------------------------------------- Check distribution receive difference -------------------------------------------------*/
          /*// Calculate frobenius norm of A_local - A
          double frobenius_norm_received = 0.0;
          // Iterate columns assigned to this rank
          for(auto column_index: root_set_columns_by_rank[index_rank]){
            for(size_t index_row = 0; index_row < m; ++index_row){
              double sustraction = (A_gather.elements[iteratorC(index_row, root_column_index_to_local_column_index[index_rank][column_index], m)] - A.elements[iteratorC(index_row, column_index, m)]);
              frobenius_norm_received += sustraction * sustraction;
            }
          }

          std::cout << "Gather ||A-USVt||_F: " << sqrt(frobenius_norm_received) << "\n";*/

          size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
#pragma omp parallel for
          for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
            for(size_t index_row = 0; index_row < m; ++index_row){
              A.elements[iteratorC(index_row, root_set_columns_vector_by_rank[index_rank][index_column], lda)] = A_gather.elements[iteratorC(index_row, index_column, lda)];
            }
          }

          A_gather.free();
        }
      } else {
        auto return_status = MPI_Send(A_local.elements, m * local_set_columns.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
        }
      }

      /* ----------------------------------- Gather local V solutions ------------------------------------------------*/
      if(rank == ROOT_RANK){
        for(size_t index_rank = 1; index_rank < size; ++index_rank){
          // Create local matrix
          MatrixMPI V_gather(m, root_set_columns_by_rank[index_rank].size());

          MPI_Status status;
          auto return_status = MPI_Recv(V_gather.elements, n * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD, &status);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
          }

          /* ------------------------------------- Check distribution receive difference -------------------------------------------------*/
          /*// Calculate frobenius norm of A_local - A
          double frobenius_norm_received = 0.0;
          // Iterate columns assigned to this rank
          for(auto column_index: root_set_columns_by_rank[index_rank]){
            for(size_t index_row = 0; index_row < m; ++index_row){
              double sustraction = (A_gather.elements[iteratorC(index_row, root_column_index_to_local_column_index[index_rank][column_index], m)] - A.elements[iteratorC(index_row, column_index, m)]);
              frobenius_norm_received += sustraction * sustraction;
            }
          }

          std::cout << "Gather ||A-USVt||_F: " << sqrt(frobenius_norm_received) << "\n";*/

          size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
#pragma omp parallel for
          for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
            for(size_t index_row = 0; index_row < n; ++index_row){
              V.elements[iteratorC(index_row, root_set_columns_vector_by_rank[index_rank][index_column], lda)] = V_gather.elements[iteratorC(index_row, index_column, lda)];
            }
          }

          V_gather.free();
        }
      } else {
        auto return_status = MPI_Send(V_local.elements, n * local_set_columns.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
        }
      }

      // Free local matrix
      A_local.free();
      V_local.free();

      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  for(size_t i = 0; i < num_of_threads; i++){
    cudaFree(d_p_vectors[i]);
    cudaFree(d_q_vectors[i]);
    cudaFree(d_v_p_vectors[i]);
    cudaFree(d_v_q_vectors[i]);
  }

  if(rank == ROOT_RANK){
    // Compute \Sigma
#pragma omp parallel for
    for (size_t k = 0; k < std::min(m, n); ++k) {
      for (size_t i = 0; i < m; ++i) {
        s.elements[k] += A.elements[iteratorC(i, k, lda)] * A.elements[iteratorC(i, k, lda)];
      }
      s.elements[k] = sqrt(s.elements[k]);
    }

    //Compute U
    if (jobu == AllVec) {
      // We expect here squared matrix
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          A.elements[iteratorC(j, i, lda)] /= s.elements[i];
        }
      }
    } else if (jobu == SomeVec) {
      // We expect here squared matrix
//      #pragma omp parallel for
//      for (size_t k = 0; k < std::min(m, n); ++k) {
//        for (size_t i = 0; i < m; ++i) {
//          A.elements[iteratorC(j, i, lda)] /= s.elements[i];
//        }
//      }
    }
  }
}


int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  // Set number of threads
  omp_set_dynamic(0);
  omp_set_num_threads(36);

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

  std::cout << "Number of threads: " << omp_thread_count() << '\n';

  std::cout << "hi from rank: " << rank << '\n';

  {
    // Build matrix A and R
    /* -------------------------------- Test 1 (Squared matrix SVD) OMP -------------------------------- */
    file_output
        << "-------------------------------- Test 1 (Squared matrix SVD) OMP --------------------------------\n";
    std::cout
        << "-------------------------------- Test 1 (Squared matrix SVD) OMP --------------------------------\n";

    const size_t height = 1000;
    const size_t width = 1000;

    std::cout << "Dimensions, height: " << height << ", width: " << width << "\n";

    Matrix A(height, width), V(width, width), s(1, std::min(A.height, A.width)), A_copy(height, width);

    const unsigned long A_height = A.height, A_width = A.width;

    std::fill_n(V.elements, V.height * V.width, 0.0);
    std::fill_n(A.elements, A.height * A.width, 0.0);
    std::fill_n(A_copy.elements, A_copy.height * A_copy.width, 0.0);

    // Create R matrix
    std::random_device random_device;
    std::mt19937 mt_19937(random_device());
    std::default_random_engine e(seed);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    for (size_t indexRow = 0; indexRow < std::min<size_t>(A_height, A_width); ++indexRow) {
      for (size_t indexCol = indexRow; indexCol < std::min<size_t>(A_height, A_width); ++indexCol) {
        double value = uniform_dist(mt_19937);
        A.elements[iteratorC(indexRow, indexCol, A_height)] = value;
        A_copy.elements[iteratorC(indexRow, indexCol, A_height)] = value;
      }
    }

    // Calculate SVD decomposition
    double ti = omp_get_wtime();
    cuda_dgesvd_kernel(AllVec,
                       AllVec,
                       A.height,
                       A.width,
                       A,
                       A_height,
                       s,
                       V,
                       A_width);
    double tf = omp_get_wtime();
    double time = tf - ti;

    std::cout << "SVD CUDA Kernel time with U,V calculation: " << time << "\n";

#pragma omp parallel for
    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
        double value = 0.0;
        for (size_t k_dot = 0; k_dot < A_width; ++k_dot) {
          value += A.elements[iteratorC(indexRow, k_dot, A_height)] * s.elements[k_dot]
              * V.elements[iteratorC(indexCol, k_dot, A_height)];
        }
        A_copy.elements[iteratorC(indexRow, indexCol, A_height)] -= value;
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

    std::cout << "||A-USVt||_F: " << sqrt(frobenius_norm) << "\n";
  }

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
        A.elements[iteratorC(indexRow, indexCol, height)] = value;
        A_copy.elements[iteratorC(indexRow, indexCol, height)] = value;
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
  omp_mpi_cuda_dgesvd_local_matrices(AllVec,
                                             AllVec,
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

    // Report Matrix V
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

#ifdef ERASE
int main1(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  // Set number of threads
  omp_set_dynamic(0);
  omp_set_num_threads(36);

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

  std::cout << "Number of threads: " << Thesis::omp_thread_count() << '\n';

  std::cout << "hi from rank: " << rank << '\n';

  if (rank == ROOT_RANK) {
    {
      // Build matrix A and R
      /* -------------------------------- Test 1 (Squared matrix SVD) OMP -------------------------------- */
      file_output
          << "-------------------------------- Test 1 (Squared matrix SVD) OMP --------------------------------\n";
      std::cout
          << "-------------------------------- Test 1 (Squared matrix SVD) OMP --------------------------------\n";

      const size_t height = 1000;
      const size_t width = 1000;

      std::cout << "Dimensions, height: " << height << ", width: " << width << "\n";

      Matrix A(height, width), V(width, width), s(1, std::min(A.height, A.width)), A_copy(height, width);

      const unsigned long A_height = A.height, A_width = A.width;

      std::fill_n(V.elements, V.height * V.width, 0.0);
      std::fill_n(A.elements, A.height * A.width, 0.0);
      std::fill_n(A_copy.elements, A_copy.height * A_copy.width, 0.0);

      // Create a random matrix
//    std::default_random_engine e(seed);
//    std::uniform_real_distribution<double> uniform_dist(1.0, 2.0);
//    for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
//      for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
//        double value = uniform_dist(e);
//        A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] = value;
//      }
//    }

      // Select iterator
      auto iterator = Thesis::IteratorC;

      // Create R matrix
      std::random_device random_device;
      std::mt19937 mt_19937(random_device());
      std::default_random_engine e(seed);
      std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
      for (size_t indexRow = 0; indexRow < std::min<size_t>(A_height, A_width); ++indexRow) {
        for (size_t indexCol = indexRow; indexCol < std::min<size_t>(A_height, A_width); ++indexCol) {
          double value = uniform_dist(mt_19937);
          A.elements[iteratorC(indexRow, indexCol, A_height)] = value;
          A_copy.elements[iteratorC(indexRow, indexCol, A_height)] = value;
        }
      }

#ifdef REPORT
      // Report Matrix A
  file_output << std::fixed << std::setprecision(3) << "A: \n";
  std::cout << std::fixed << std::setprecision(3) << "A: \n";
  for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
    for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
      file_output << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
      std::cout << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
    }
    file_output << '\n';
    std::cout << '\n';
  }
  // Report Matrix A^T * A
//    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
//    for (size_t indexRow = 0; indexRow < A.width; ++indexRow) {
//      for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
//        double value = 0.0;
//        for(size_t k_dot = 0; k_dot < A.height; ++k_dot){
//          value += A.elements[Thesis::IteratorC(k_dot, indexRow, A.height)] * A.elements[Thesis::IteratorC(k_dot, indexCol, A.height)];
//        }
//        std::cout << value << " ";
//      }
//      std::cout << '\n';
//    }
#endif

      // Calculate SVD decomposition
      double ti = omp_get_wtime();
      Thesis::cuda_dgesvd_kernel(Thesis::AllVec,
                                 Thesis::AllVec,
                                 A.height,
                                 A.width,
                                 A,
                                 A_height,
                                 s,
                                 V,
                                 A_width);
      double tf = omp_get_wtime();
      double time = tf - ti;

      std::cout << "SVD CUDA Kernel time with U,V calculation: " << time << "\n";

      #pragma omp parallel for
      for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
        for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
          double value = 0.0;
          for (size_t k_dot = 0; k_dot < A_width; ++k_dot) {
            value += A.elements[iteratorC(indexRow, k_dot, A_height)] * s.elements[k_dot]
                * V.elements[iteratorC(indexCol, k_dot, A_height)];
          }
          A_copy.elements[iteratorC(indexRow, indexCol, A_height)] -= value;
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

      std::cout << "||A-USVt||_F: " << sqrt(frobenius_norm) << "\n";
    }
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
  Thesis::omp_mpi_cuda_dgesvd_local_matrices(Thesis::AllVec,
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

    // Report Matrix V
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
#endif
