mpiexec -n 3 -v -display-map -display-allocation -bind-to core ./SVD_Jacobi_MPI_CUDA 5000
mpiexec -n 3 -v -display-map -display-allocation -bind-to core ./SVD_Jacobi_MPI_CUDA 10000
mpiexec -n 3 -v -display-map -display-allocation -bind-to core ./SVD_Jacobi_MPI_CUDA 15000
mpiexec -n 3 -v -display-map -display-allocation -bind-to core ./SVD_Jacobi_MPI_CUDA 20000
mpiexec -n 3 -v -display-map -display-allocation -bind-to core ./SVD_Jacobi_MPI_CUDA 30000
mpiexec -n 3 -v -display-map -display-allocation -bind-to core ./SVD_Jacobi_MPI_CUDA 50000
