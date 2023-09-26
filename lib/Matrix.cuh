//
// Created by andre on 6/16/23.
//

#ifndef SVD_JACOBI_MPI_CUDA_LIB_MATRIX_CUH_
#define SVD_JACOBI_MPI_CUDA_LIB_MATRIX_CUH_

#include "../../../../../../usr/include/c++/11/vector"

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

#endif //SVD_JACOBI_MPI_CUDA_LIB_MATRIX_CUH_
