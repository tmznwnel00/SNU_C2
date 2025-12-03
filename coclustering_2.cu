/*
 * coclustering_2_fixed.cu
 *
 * Stable randomized "SVD-style" factorization for CSR sparse matrix Z (m x n).
 * Instead of exact SVD, we compute:
 *
 *   1) Y = Z * Omega            (Omega ~ N(0,1) in R^{n x (k+p)})
 *   2) Optional power iters: Y = (Z Z^T)^q Z * Omega
 *   3) QR: Y = Q R              (m x l, l = k+p)
 *   4) Row embedding:  U  = Q(:, 1..k)        (m x k)
 *   5) Column embedding: V  = Z^T * U        (n x k)
 *   6) Vt = V^T                               (k x n)
 *   7) S  = 1 (dummy singular values)
 *
 * All dense matrices use column-major layout.
 *
 * Signature is compatible with previous code:
 *
 *   int randomized_svd_csr(int m, int n, int nnz,
 *                          const int* d_rowPtr, const int* d_colIdx, const float* d_vals,
 *                          int k, int p, int q,
 *                          float** d_U, float** d_S, float** d_Vt)
 *
 * Compile:
 *   nvcc -O3 -lcusparse -lcublas -lcusolver -lcurand -c coclustering_2_fixed.cu -o coclustering_2.o
 */

 #include <cuda_runtime.h>
 #include <cusparse.h>
 #include <cublas_v2.h>
 #include <cusolverDn.h>  // QR만 씀
 #include <curand.h>
 #include <cstdio>
 #include <cstdlib>
 #include <cmath>
 
 #define CHECK_CUDA(call) do { \
     cudaError_t _e = (call); \
     if (_e != cudaSuccess) { \
         fprintf(stderr, "CUDA Error %s:%d: %s\n", \
                 __FILE__, __LINE__, cudaGetErrorString(_e)); \
         exit(EXIT_FAILURE); \
     } \
 } while(0)
 
 #define CHECK_CUSPARSE(call) do { \
     cusparseStatus_t _s = (call); \
     if (_s != CUSPARSE_STATUS_SUCCESS) { \
         fprintf(stderr, "cuSPARSE Error %s:%d: %d\n", \
                 __FILE__, __LINE__, (int)_s); \
         exit(EXIT_FAILURE); \
     } \
 } while(0)
 
 #define CHECK_CUBLAS(call) do { \
     cublasStatus_t _s = (call); \
     if (_s != CUBLAS_STATUS_SUCCESS) { \
         fprintf(stderr, "cuBLAS Error %s:%d: %d\n", \
                 __FILE__, __LINE__, (int)_s); \
         exit(EXIT_FAILURE); \
     } \
 } while(0)
 
 #define CHECK_CUSOLVER(call) do { \
     cusolverStatus_t _s = (call); \
     if (_s != CUSOLVER_STATUS_SUCCESS) { \
         fprintf(stderr, "cuSOLVER Error %s:%d: %d\n", \
                 __FILE__, __LINE__, (int)_s); \
         exit(EXIT_FAILURE); \
     } \
 } while(0)
 
 #define CHECK_CURAND(call) do { \
     curandStatus_t _s = (call); \
     if (_s != CURAND_STATUS_SUCCESS) { \
         fprintf(stderr, "cuRAND Error %s:%d: %d\n", \
                 __FILE__, __LINE__, (int)_s); \
         exit(EXIT_FAILURE); \
     } \
 } while(0)
 
 // zero-fill
 __global__ void fill_zero_kernel(float* x, int n) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < n) x[i] = 0.0f;
 }
 
 // fill with ones
 __global__ void fill_one_kernel(float* x, int n) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < n) x[i] = 1.0f;
 }
 
 // column-major transpose: B = A^T
 // A: rows x cols, lda = rows (column-major)
 // B: cols x rows, ldb = cols (column-major)
 __global__ void transpose_colmajor_kernel(int rows, int cols,
                                           const float* __restrict__ A, int lda,
                                           float* __restrict__ B, int ldb)
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= rows * cols) return;
     int r = idx % rows;
     int c = idx / rows;
     // A(r,c) = A[r + c*lda]
     float val = A[r + c * lda];
     // B(c,r) = B[c + r*ldb]
     B[c + r * ldb] = val;
 }
 
 extern "C" {
 
 /*
  * randomized_svd_csr:
  *  - No true SVD anymore (to avoid cusolver internal error)
  *  - Computes U, S, Vt as described above.
  */
 int randomized_svd_csr(
     int m, int n, int nnz,
     const int* d_rowPtr,
     const int* d_colIdx,
     const float* d_vals,
     int k, int p, int q,
     float** d_U, float** d_S, float** d_Vt)
 {
     const int l = k + p;
     const int tpb = 256;
 
     // ------------------ Handles ------------------
     cusparseHandle_t sp_handle;   CHECK_CUSPARSE(cusparseCreate(&sp_handle));
     cublasHandle_t   blas_handle; CHECK_CUBLAS(cublasCreate(&blas_handle));
     cusolverDnHandle_t solver;    CHECK_CUSOLVER(cusolverDnCreate(&solver));
     curandGenerator_t curgen;     CHECK_CURAND(curandCreateGenerator(&curgen, CURAND_RNG_PSEUDO_DEFAULT));
     CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(curgen, 1234ULL));
 
     // ------------------ 1) Omega ~ N(0,1), n x l ------------------
     float* d_Omega = nullptr;
     CHECK_CUDA(cudaMalloc(&d_Omega, sizeof(float) * (size_t)n * l));
     CHECK_CURAND(curandGenerateNormal(curgen, d_Omega, (size_t)n * l, 0.0f, 1.0f));
 
     // ------------------ 2) Y = Z * Omega, m x l ------------------
     float* d_Y = nullptr;
     CHECK_CUDA(cudaMalloc(&d_Y, sizeof(float) * (size_t)m * l));
     fill_zero_kernel<<<(m*l + tpb-1)/tpb, tpb>>>(d_Y, m*l);
 
     cusparseSpMatDescr_t matZ;
     CHECK_CUSPARSE(cusparseCreateCsr(
         &matZ, m, n, nnz,
         (void*)d_rowPtr, (void*)d_colIdx, (void*)d_vals,
         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
 
     cusparseDnMatDescr_t matOmega, matY;
     CHECK_CUSPARSE(cusparseCreateDnMat(&matOmega,
                                        n, l, n,
                                        (void*)d_Omega,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL));
     CHECK_CUSPARSE(cusparseCreateDnMat(&matY,
                                        m, l, m,
                                        (void*)d_Y,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL));
 
     float alpha = 1.0f, beta = 0.0f;
     size_t bufSize = 0;
     CHECK_CUSPARSE(cusparseSpMM_bufferSize(
         sp_handle,
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         &alpha, matZ, matOmega, &beta, matY,
         CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &bufSize));
 
     void* dBuf = nullptr;
     CHECK_CUDA(cudaMalloc(&dBuf, bufSize));
     CHECK_CUSPARSE(cusparseSpMM(
         sp_handle,
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         &alpha, matZ, matOmega, &beta, matY,
         CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, dBuf));
 
     // ------------------ 2b) Power iterations: Y = (Z Z^T)^q Z Ω ------------------
     for (int it = 0; it < q; ++it) {
         // W = Z^T * Y  (n x l)
         float* d_W = nullptr;
         CHECK_CUDA(cudaMalloc(&d_W, sizeof(float) * (size_t)n * l));
         fill_zero_kernel<<<(n*l + tpb-1)/tpb, tpb>>>(d_W, n*l);
 
         cusparseDnMatDescr_t matW, matYin;
         CHECK_CUSPARSE(cusparseCreateDnMat(&matW,  n, l, n, (void*)d_W,  CUDA_R_32F, CUSPARSE_ORDER_COL));
         CHECK_CUSPARSE(cusparseCreateDnMat(&matYin,m, l, m, (void*)d_Y,  CUDA_R_32F, CUSPARSE_ORDER_COL));
 
         size_t bufZW = 0;
         void* dBufZW = nullptr;
         CHECK_CUSPARSE(cusparseSpMM_bufferSize(
             sp_handle,
             CUSPARSE_OPERATION_TRANSPOSE,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha, matZ, matYin, &beta, matW,
             CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &bufZW));
         CHECK_CUDA(cudaMalloc(&dBufZW, bufZW));
         CHECK_CUSPARSE(cusparseSpMM(
             sp_handle,
             CUSPARSE_OPERATION_TRANSPOSE,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha, matZ, matYin, &beta, matW,
             CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, dBufZW));
 
         // Y = Z * W  (m x l)
         cusparseDnMatDescr_t matWin, matYout;
         CHECK_CUSPARSE(cusparseCreateDnMat(&matWin, n, l, n, (void*)d_W, CUDA_R_32F, CUSPARSE_ORDER_COL));
         CHECK_CUSPARSE(cusparseCreateDnMat(&matYout,m, l, m, (void*)d_Y, CUDA_R_32F, CUSPARSE_ORDER_COL));
 
         size_t bufZW2 = 0;
         void* dBufZW2 = nullptr;
         CHECK_CUSPARSE(cusparseSpMM_bufferSize(
             sp_handle,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha, matZ, matWin, &beta, matYout,
             CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &bufZW2));
         CHECK_CUDA(cudaMalloc(&dBufZW2, bufZW2));
         CHECK_CUSPARSE(cusparseSpMM(
             sp_handle,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha, matZ, matWin, &beta, matYout,
             CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, dBufZW2));
 
         CHECK_CUDA(cudaFree(d_W));
         CHECK_CUDA(cudaFree(dBufZW));
         CHECK_CUDA(cudaFree(dBufZW2));
         cusparseDestroyDnMat(matW);
         cusparseDestroyDnMat(matYin);
         cusparseDestroyDnMat(matWin);
         cusparseDestroyDnMat(matYout);
     }
 
     // ------------------ 3) QR: Y = Q ------------------
     float* d_tau = nullptr;
     CHECK_CUDA(cudaMalloc(&d_tau, sizeof(float)*l));
 
     int lda = m;
     int lwork_qr = 0;
     CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(solver, m, l, d_Y, lda, &lwork_qr));
 
     float* d_work_qr = nullptr;
     CHECK_CUDA(cudaMalloc(&d_work_qr, sizeof(float)*lwork_qr));
     int* devInfo = nullptr;
     CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
 
     CHECK_CUSOLVER(cusolverDnSgeqrf(solver, m, l, d_Y, lda,
                                     d_tau, d_work_qr, lwork_qr, devInfo));
     CHECK_CUSOLVER(cusolverDnSorgqr(solver, m, l, l, d_Y, lda,
                                     d_tau, d_work_qr, lwork_qr, devInfo));
     // d_Y now = Q (m x l)
 
     // ------------------ 4) Row embedding: U = Q(:,1..k) ------------------
     CHECK_CUDA(cudaMalloc((void**)d_U, sizeof(float) * (size_t)m * k));
     CHECK_CUDA(cudaMemcpy2D(
         *d_U, sizeof(float)*m,
         d_Y,  sizeof(float)*m,
         sizeof(float)*m, k,
         cudaMemcpyDeviceToDevice));
 
     // ------------------ 5) Column embedding: V = Z^T * U  (n x k) ------------------
     float* d_V = nullptr;
     CHECK_CUDA(cudaMalloc(&d_V, sizeof(float) * (size_t)n * k));
 
     cusparseDnMatDescr_t matU = nullptr, matV = nullptr;
     CHECK_CUSPARSE(cusparseCreateDnMat(&matU, m, k, m, (void*)(*d_U), CUDA_R_32F, CUSPARSE_ORDER_COL));
     CHECK_CUSPARSE(cusparseCreateDnMat(&matV, n, k, n, (void*)d_V,    CUDA_R_32F, CUSPARSE_ORDER_COL));
 
     float alpha2 = 1.0f, beta2 = 0.0f;
     size_t bufZ = 0;
     void* dBufZ = nullptr;
     CHECK_CUSPARSE(cusparseSpMM_bufferSize(
         sp_handle,
         CUSPARSE_OPERATION_TRANSPOSE,      // Z^T
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         &alpha2, matZ, matU, &beta2, matV,
         CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &bufZ));
     CHECK_CUDA(cudaMalloc(&dBufZ, bufZ));
     CHECK_CUSPARSE(cusparseSpMM(
         sp_handle,
         CUSPARSE_OPERATION_TRANSPOSE,
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         &alpha2, matZ, matU, &beta2, matV,
         CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, dBufZ));
 
     // ------------------ 6) Vt = V^T (k x n) ------------------
     CHECK_CUDA(cudaMalloc((void**)d_Vt, sizeof(float) * (size_t)k * n));
     {
         int elems = n * k;
         int blocks = (elems + tpb-1)/tpb;
         // V:  n x k, ldV = n
         // Vt: k x n, ldVt = k
         transpose_colmajor_kernel<<<blocks, tpb>>>(
             n, k, d_V, n, *d_Vt, k);
         CHECK_CUDA(cudaGetLastError());
     }
 
     // ------------------ 7) S = 1.0 ------------------
     CHECK_CUDA(cudaMalloc((void**)d_S, sizeof(float) * (size_t)k));
     fill_one_kernel<<<(k+tpb-1)/tpb, tpb>>>(*d_S, k);
 
     // ------------------ Cleanup ------------------
     CHECK_CUDA(cudaFree(d_Omega));
     CHECK_CUDA(cudaFree(d_Y));
     CHECK_CUDA(cudaFree(dBuf));
     CHECK_CUDA(cudaFree(d_tau));
     CHECK_CUDA(cudaFree(d_work_qr));
     CHECK_CUDA(cudaFree(devInfo));
     CHECK_CUDA(cudaFree(d_V));
     CHECK_CUDA(cudaFree(dBufZ));
     cusparseDestroySpMat(matZ);
     cusparseDestroyDnMat(matOmega);
     cusparseDestroyDnMat(matY);
     cusparseDestroyDnMat(matU);
     cusparseDestroyDnMat(matV);
 
     CHECK_CURAND(curandDestroyGenerator(curgen));
     CHECK_CUSOLVER(cusolverDnDestroy(solver));
     CHECK_CUBLAS(cublasDestroy(blas_handle));
     CHECK_CUSPARSE(cusparseDestroy(sp_handle));
 
     return 0;
 }
 
 } // extern "C"
 