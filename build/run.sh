while IFS=" " read -r ref_d
do
   echo $ref_d
   text="$ref_d"
#   mpiexec -n 3 -v -display-map -display-allocation --get-stack-traces --timeout 50 -bind-to core .SVD_Jacobi_MPI_OMP ref_d > std_out_$text.out
   mpiexec -n 2 -v -display-map -display-allocation -bind-to core ./SVD_Jacobi_MPI_CUDA $ref_d > std_out_$text.out
done < "dimensions.dat"