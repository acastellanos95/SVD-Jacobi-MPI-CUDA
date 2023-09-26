cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)

project(SVD_Jacobi_MPI_CUDA LANGUAGES CUDA CXX)

# Find MPI package
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_DIRS})

# Find CUDA package
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA architecture (change according to your GPU architecture)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_60)

# Find OpenMP package
find_package(OpenMP REQUIRED)
# Add executable
add_executable(SVD_Jacobi_MPI_CUDA main.cu lib/Matrix.cuh lib/JacobiMethods.cu lib/JacobiMethods.cuh lib/global.cuh lib/Utils.cu lib/Utils.cuh)

# Link MPI and OpenMP libraries
target_link_libraries(SVD_Jacobi_MPI_CUDA PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX gomp)