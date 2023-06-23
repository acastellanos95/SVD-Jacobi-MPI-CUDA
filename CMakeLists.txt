cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER_ID NVIDIA)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)
set(CMAKE_CUDA_STANDARD 17)

project(SVD_Jacobi_MPI_CUDA LANGUAGES CUDA CXX)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

IF (OPENMP_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS} --std=c++11 -O3")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

#target_compile_options(SVD_Thesis PUBLIC)
add_executable(SVD_Jacobi_MPI_CUDA main.cu lib/Matrix.cuh lib/JacobiMethods.cu lib/JacobiMethods.cuh lib/global.cuh lib/Utils.cu lib/Utils.cuh)

set_target_properties(SVD_Jacobi_MPI_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(SVD_Jacobi_MPI_CUDA cublas curand MPI::MPI_CXX)
