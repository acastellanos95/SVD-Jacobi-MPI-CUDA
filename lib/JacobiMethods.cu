//
// Created by andre on 6/16/23.
//

#include "JacobiMethods.cuh"

namespace Thesis {

/***************************************************************************
    Purpose
    -------
    sequential_dgesvd computes the singular value decomposition (SVD)
    of a real M-by-N with m>>n matrix A using Jacobi one sided
    algorithm with no parallelism, optionally computing the left
    and/or right singular vectors. The SVD is written like

        A = U * SIGMA * transpose(V)

    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
    V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.

    Note on one sided Jacobi:

        V = ((IxJ_0)xJ_1,...)
        U = A\sigma^{-1}

    Note that the routine returns VT = V**T, not V.

    Arguments
    ---------
    @param[in]
    jobu    SVD_OPTIONS
            Specifies options for computing all or part of the matrix U:
      -     = AllVec:        all M columns of U are returned in array U:
      -     = SomeVec:       the first min(m,n) columns of U (the left singular
                                  vectors) are returned in the array U;
      -     = NoVec:         no columns of U (no left singular vectors) are
                                  computed.

    @param[in]
    jobvt   SVD_OPTIONS
            Specifies options for computing all or part of the matrix V**T:
      -     = AllVec:        all N rows of V**T are returned in the array VT;
      -     = SomeVec:       the first min(m,n) rows of V**T (the right singular
                                  vectors) are returned in the array VT;
      -     = NoVec:         no rows of V**T (no right singular vectors) are
                                  computed.
    \n

    @param[in]
    m       INTEGER
            The number of rows of the input matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the input matrix A.  N >= 0.

    @param[in]
    matrix_layout_A MATRIX_LAYOUT
            The layout of the matrix A. It can only be
            ROW_MAJOR or COL_MAJOR.

    @param[in,out]
    A       DOUBLE PRECISION array, dimension (M,N)
            On entry, the M-by-N matrix A.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.

    @param[out]
    s       DOUBLE PRECISION array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i+1).

    @param[out]
    U       DOUBLE PRECISION array in major column order, dimension (LDU,UCOL)
            (LDU,M) if JOBU = AllVec or (LDU,min(M,N)) if JOBU = SomeVec.
      -     If JOBU = AllVec, U contains the M-by-M orthogonal matrix U;
      -     if JOBU = SomeVec, U contains the first min(m,n) columns of U
            (the left singular vectors, stored columnwise);
      -     if JOBU = NoVec, U is not referenced.

    @param[in]
    ldu     INTEGER
            The leading dimension of the array U.  LDU >= 1; if
            JOBU = SomeVec or AllVec, LDU >= M.

    @param[out]
    V      DOUBLE PRECISION array in major column order, dimension (LDV,N)
      -     If JOBVT = AllVec, VT contains the N-by-N orthogonal matrix V**T;
      -     if JOBVT = SomeVec, VT contains the first min(m,n) rows of V**T
            (the right singular vectors, stored rowwise);
      -     if JOBVT = NoVec, VT is not referenced.

    @param[in]
    ldv    INTEGER
            The leading dimension of the array VT.  LDVT >= 1;
      -     if JOBVT = AllVec, LDVT >= N;
      -     if JOBVT = SomeVec , LDVT >= min(M,N).

    @param[out]
    work    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the required LWORK.
            if INFO > 0, WORK(2:MIN(M,N)) contains the unconverged
            superdiagonal elements of an upper bidiagonal matrix B
            whose diagonal is in S (not necessarily sorted). B
            satisfies A = U * B * VT, so it has the same singular values
            as A, and singular vectors related by U and VT.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If lwork = -1, a workspace query is assumed.  The optimal
            size for the WORK array is calculated and stored in WORK[0],
            and no other work except argument checking is performed.
    \n
            Let mx = max(M,N) and mn = min(M,N).
            The threshold for mx >> mn is currently mx >= 1.6*mn.
            For job: N=None, O=Overwrite, S=Some, A=All.
            Paths below assume M >= N; for N > M swap jobu and jobvt.
    \n
            Because of varying nb for different subroutines, formulas below are
            an upper bound. Querying gives an exact number.
            The optimal block size nb can be obtained through magma_get_dgesvd_nb(M,N).
            For many cases, there is a fast algorithm, and a slow algorithm that
            uses less workspace. Here are sizes for both cases.
    \n
            Optimal lwork (fast algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any                  3*mn + 2*mn*nb
            Path 2:   jobu=O, jobvt=N        mn*mn +     3*mn + 2*mn*nb
                                   or        mn*mn + max(3*mn + 2*mn*nb, mn + mx*mn)
            Path 3:   jobu=O, jobvt=A,S      mn*mn +     3*mn + 2*mn*nb
                                   or        mn*mn + max(3*mn + 2*mn*nb, mn + mx*mn)
            Path 4:   jobu=S, jobvt=N        mn*mn +     3*mn + 2*mn*nb
            Path 5:   jobu=S, jobvt=O      2*mn*mn +     3*mn + 2*mn*nb
            Path 6:   jobu=S, jobvt=A,S      mn*mn +     3*mn + 2*mn*nb
            Path 7:   jobu=A, jobvt=N        mn*mn + max(3*mn + 2*mn*nb, mn + mx*nb)
            Path 8:   jobu=A, jobvt=O      2*mn*mn + max(3*mn + 2*mn*nb, mn + mx*nb)
            Path 9:   jobu=A, jobvt=A,S      mn*mn + max(3*mn + 2*mn*nb, mn + mx*nb)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  3*mn + (mx + mn)*nb
    \n
            Optimal lwork (slow algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any    n/a
            Path 2:   jobu=O, jobvt=N      3*mn + (mx + mn)*nb
            Path 3-9:                      3*mn + max(2*mn*nb, mx*nb)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  n/a
    \n
            MAGMA requires the optimal sizes above, while LAPACK has the same
            optimal sizes but the minimum sizes below.
    \n
            LAPACK minimum lwork (fast algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any              5*mn
            Path 2:   jobu=O, jobvt=N        mn*mn + 5*mn
            Path 3:   jobu=O, jobvt=A,S      mn*mn + 5*mn
            Path 4:   jobu=S, jobvt=N        mn*mn + 5*mn
            Path 5:   jobu=S, jobvt=O      2*mn*mn + 5*mn
            Path 6:   jobu=S, jobvt=A,S      mn*mn + 5*mn
            Path 7:   jobu=A, jobvt=N        mn*mn + max(5*mn, mn + mx)
            Path 8:   jobu=A, jobvt=O      2*mn*mn + max(5*mn, mn + mx)
            Path 9:   jobu=A, jobvt=A,S      mn*mn + max(5*mn, mn + mx)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  max(3*mn + mx, 5*mn)
    \n
            LAPACK minimum lwork (slow algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any    n/a
            Path 2-9:                      max(3*mn + mx, 5*mn)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  n/a

    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  if DBDSQR did not converge, INFO specifies how many
                superdiagonals of an intermediate bidiagonal form B
                did not converge to zero. See the description of WORK
                above for details.
*********************************************************************************/


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
  auto iterator = get_iterator(COL_MAJOR);
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

void cuda_dgesvd_kernel(SVD_OPTIONS jobu,
                        SVD_OPTIONS jobv,
                        size_t m,
                        size_t n,
                        Matrix &A,
                        size_t lda,
                        Matrix &s,
                        Matrix &V,
                        size_t ldv) {
  auto num_of_threads = Thesis::omp_thread_count();

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

/***************************************************************************
    Purpose
    -------
    Applies the Jacobi rotation cosine and sine to both arrays x and y.
    In the following way:

    x[i] = c * x[i] - s * y[i]
    y[i] = s * x[i] + c * y[i]
    Arguments
    ---------
    @param[in]
    n       int
            number of elements in array to apply the rotation.

    @param[in,out]
    x       DOUBLE PRECISION array dimension at least n
            The x array to be overwritten.

    @param[in,out]
    y       DOUBLE PRECISION array dimension at least n
            The y array to be overwritten.

    @param[in]
    c       DOUBLE
            Cosine of Jacobi rotation.

    @param[in]
    s       DOUBLE PRECISION
            Sine of Jacobi rotation.
*********************************************************************************/
__global__ void jacobi_rotation(unsigned int n, double *x, double *y, double c, double s) {
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    double tmp = x[i];
    x[i] = c * tmp - s * y[i];
    y[i] = s * tmp + c * y[i];
  }
}

} // Thesis