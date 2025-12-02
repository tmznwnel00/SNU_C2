/*
 * coclustering_2.cu
 *
 * GPU-only Randomized Truncated SVD optimized for sparse input Z (CSR/COO).
 * Strategy:
 *  1) Random projection Y = Z * Omega  (cusparseSpMM)
 *  2) Optional power iterations: Y <- Z*(Z^T * Y)
 *  3) QR factorization Y = Q R  (cusolver geqrf + orgqr)
 *  4) Small matrix B = Q^T * Z   (computed as (Z^T * Q)^T using cusparseSpMM)
 *  5) SVD of B (l x n) using cuSOLVER (gesvd)
 *  6) Reconstruct U = Q * U_small, S, V = V_small
 *
 * Notes/assumptions:
 * - Input sparse matrix Z is on device in CSR format: d_rowPtr (m+1), d_colIdx (nnz), d_vals (nnz)
 * - All arrays are float (32-bit)
 * - Caller supplies device pointers for outputs (or we allocate and return newly allocated pointers)
 * - Uses cuSPARSE, cuBLAS, cuSOLVER, cuRAND
 *
 * Compile:
 *   nvcc -O3 -lcusparse -lcublas -lcusolver -lcurand -c coclustering_2.cu -o coclustering_2.o
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUSPARSE(call) do { \
    cusparseStatus_t _s = (call); \
    if (_s != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE Error %s:%d: %d\n", __FILE__, __LINE__, (int)_s); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error %s:%d: %d\n", __FILE__, __LINE__, (int)_s); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUSOLVER(call) do { \
    cusolverStatus_t _s = (call); \
    if (_s != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER Error %s:%d: %d\n", __FILE__, __LINE__, (int)_s); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CURAND(call) do { \
    curandStatus_t _s = (call); \
    if (_s != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "cuRAND Error %s:%d: %d\n", __FILE__, __LINE__, (int)_s); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// simple kernel to zero device memory
__global__ static void fill_zero(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

extern "C" {

// Randomized SVD for CSR sparse matrix Z on device.
// Inputs:
//   m,n,nnz: matrix dims and nonzeros
//   d_rowPtr (m+1), d_colIdx (nnz), d_vals (nnz): CSR representation on device
//   k: target rank
//   p: oversample (recommended 5-20)
//   q: power iterations (recommended 0-2)
// Outputs (allocated on device by this function):
//   *d_U  (m x k), *d_S (k), *d_Vt (k x n)
// Returns 0 on success
int randomized_svd_csr(int m, int n, int nnz,
                       const int* d_rowPtr, const int* d_colIdx, const float* d_vals,
                       int k, int p, int q,
                       float** d_U, float** d_S, float** d_Vt) {
    int l = k + p; // projection dimension

    cusparseHandle_t sp_handle = nullptr;
    cublasHandle_t blas_handle = nullptr;
    cusolverDnHandle_t solver_handle = nullptr;
    curandGenerator_t curand_gen = nullptr;

    CHECK_CUSPARSE(cusparseCreate(&sp_handle));
    CHECK_CUBLAS(cublasCreate(&blas_handle));
    CHECK_CUSOLVER(cusolverDnCreate(&solver_handle));
    CHECK_CURAND(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(curand_gen, 1234ULL));

    // Allocate Omega (n x l) dense on device, column-major for cuBLAS
    float* d_Omega = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_Omega, sizeof(float) * (size_t)n * l));
    // Generate normal random numbers into Omega
    CHECK_CURAND(curandGenerateNormal(curand_gen, d_Omega, (size_t)n * l, 0.0f, 1.0f));

    // Step 1: Y = Z * Omega  -> Y (m x l)
    float* d_Y = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_Y, sizeof(float) * (size_t)m * l));
    // zero Y
    int tpb = 128;
    int gy = (m * l + tpb - 1) / tpb;
    fill_zero<<<gy, tpb>>>(d_Y, m * l);
    CHECK_CUDA(cudaGetLastError());

    // Create sparse descriptor for Z
    cusparseSpMatDescr_t matZ = nullptr;
    CHECK_CUSPARSE(cusparseCreateCsr(&matZ, m, n, nnz,
                                     (void*)d_rowPtr, (void*)d_colIdx, (void*)d_vals,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Create dense descriptors
    cusparseDnMatDescr_t matOmega = nullptr;
    cusparseDnMatDescr_t matY = nullptr;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matOmega, n, l, n, (void*)d_Omega, CUDA_R_32F, CUSPARSE_ORDER_COLUMN));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matY, m, l, m, (void*)d_Y, CUDA_R_32F, CUSPARSE_ORDER_COLUMN));

    float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matZ, matOmega, &beta, matY, CUDA_R_32F,
                                           CUSPARSE_MM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matZ, matOmega, &beta, matY, CUDA_R_32F,
                                CUSPARSE_MM_ALG_DEFAULT, dBuffer));

    // Optional power iterations: for t in 1..q: Y = Z*(Z^T * Y)
    for (int iter = 0; iter < q; ++iter) {
        // compute W = Z^T * Y  -> W (n x l)
        float* d_W = nullptr;
        CHECK_CUDA(cudaMalloc((void**)&d_W, sizeof(float) * (size_t)n * l));
        // zero W
        int gw = (n * l + tpb - 1) / tpb;
        fill_zero<<<gw, tpb>>>(d_W, n * l);
        CHECK_CUDA(cudaGetLastError());

        cusparseDnMatDescr_t matW = nullptr;
        cusparseDnMatDescr_t matY2 = nullptr;
        CHECK_CUSPARSE(cusparseCreateDnMat(&matW, n, l, n, (void*)d_W, CUDA_R_32F, CUSPARSE_ORDER_COLUMN));
        CHECK_CUSPARSE(cusparseCreateDnMat(&matY2, m, l, m, (void*)d_Y, CUDA_R_32F, CUSPARSE_ORDER_COLUMN));

        // Z^T * Y : use operation TRANSPOSE on sparse matrix
        size_t buf2 = 0;
        void* dBuf2 = nullptr;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha, matZ, matY2, &beta, matW, CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, &buf2));
        CHECK_CUDA(cudaMalloc(&dBuf2, buf2));
        CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matZ, matY2, &beta, matW, CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, dBuf2));

        // Y = Z * W
        // prepare descriptors for W
        cusparseDnMatDescr_t matW2 = nullptr;
        cusparseDnMatDescr_t matYout = nullptr;
        CHECK_CUSPARSE(cusparseCreateDnMat(&matW2, n, l, n, (void*)d_W, CUDA_R_32F, CUSPARSE_ORDER_COLUMN));
        CHECK_CUSPARSE(cusparseCreateDnMat(&matYout, m, l, m, (void*)d_Y, CUDA_R_32F, CUSPARSE_ORDER_COLUMN));

        size_t buf3 = 0;
        void* dBuf3 = nullptr;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha, matZ, matW2, &beta, matYout, CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, &buf3));
        CHECK_CUDA(cudaMalloc(&dBuf3, buf3));
        CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matZ, matW2, &beta, matYout, CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, dBuf3));

        // cleanup W temporaries
        CHECK_CUDA(cudaFree(d_W));
        CHECK_CUDA(cudaFree(dBuf2));
        CHECK_CUDA(cudaFree(dBuf3));
        if (matW) { cusparseDestroyDnMat(matW); matW = nullptr; }
        if (matY2) { cusparseDestroyDnMat(matY2); matY2 = nullptr; }
        if (matW2) { cusparseDestroyDnMat(matW2); matW2 = nullptr; }
        if (matYout) { cusparseDestroyDnMat(matYout); matYout = nullptr; }
    }

    // QR factorization Y = Q R  (use cusolver geqrf + orgqr)
    // cuSOLVER expects column-major layout; we used column-major descriptors above
    float* d_tau = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_tau, sizeof(float) * l));

    int lda = m; // leading dimension for Y (m x l)
    int work_size = 0;
    CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(solver_handle, m, l, d_Y, lda, &work_size));
    float* d_work = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_work, sizeof(float) * work_size));
    int* devInfo = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));

    CHECK_CUSOLVER(cusolverDnSgeqrf(solver_handle, m, l, d_Y, lda, d_tau, d_work, work_size, devInfo));
    // generate Q in-place (overwrites d_Y with Q)
    CHECK_CUSOLVER(cusolverDnSorgqr(solver_handle, m, l, l, d_Y, lda, d_tau, d_work, work_size, devInfo));
    // now d_Y contains Q (m x l)

    // Compute small matrix B = Q^T * Z  (size l x n)
    // We'll compute B_T = Z^T * Q  (n x l) and then SVD on B = B_T^T
    float* d_Bt = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_Bt, sizeof(float) * (size_t)n * l));
    // zero
    int gb = (n * l + tpb - 1) / tpb;
    fill_zero<<<gb, tpb>>>(d_Bt, n * l);
    CHECK_CUDA(cudaGetLastError());

    // descriptors
    cusparseDnMatDescr_t matQ = nullptr;
    cusparseDnMatDescr_t matBt = nullptr;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matQ, m, l, m, (void*)d_Y, CUDA_R_32F, CUSPARSE_ORDER_COLUMN));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matBt, n, l, n, (void*)d_Bt, CUDA_R_32F, CUSPARSE_ORDER_COLUMN));

    // Compute B_T = Z^T * Q
    size_t bufB = 0;
    void* dBufB = nullptr;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matZ, matQ, &beta, matBt, CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, &bufB));
    CHECK_CUDA(cudaMalloc(&dBufB, bufB));
    CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matZ, matQ, &beta, matBt, CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, dBufB));

    // Now we need B of size l x n, stored column-major. B = (d_Bt)^T
    // We'll allocate d_B (l x n) and launch a kernel to transpose l x n (small l)
    float* d_B = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeof(float) * (size_t)l * n));

    // simple transpose kernel for l x n, small l so performance is fine
    // launch with one thread per element
    {
        int elems = l * n;
        int tp = 256;
        int gp = (elems + tp - 1) / tp;
        // inline lambda-ish kernel via CUDA dynamic parallelism not available; implement small kernel below
        // We'll implement a tiny device lambda via <<<>>> below using a separate kernel defined later
    }

    // Define and launch transpose kernel (host-side launch requires kernel; we'll implement below)
    // We'll use a simple device kernel declared after this function via extern; to keep patch self-contained,
    // implement a static kernel here by launching via a helper below. For simplicity, do CPU transpose using cublas? Better to write small kernel.

    // Implement transpose kernel here (device function)
    // Because apply_patch requires contiguous file content, we'll place the kernel after extern "C" end and call it by name.

    // call transpose kernel
    // kernel will be named transpose_colmajor(m,n,src, dst) but here l x n transpose
    // We'll launch with elements = l*n and compute (i,j)
    int elems = l * n;
    int tp = 256;
    int gp = (elems + tp - 1) / tp;
    // forward declaration - kernel defined later in file
    extern __global__ void transpose_kernel(int rows, int cols, const float* src, float* dst);
    transpose_kernel<<<gp, tp>>>(n, l, d_Bt, d_B); // src is n x l -> dst is l x n
    CHECK_CUDA(cudaGetLastError());

    // Now perform SVD on B (l x n) to get U_small (l x l), S (l), Vt (l x n)
    // We'll compute full SVD and then take top-k
    int mB = l;
    int nB = n;
    float* d_Usmall = nullptr;
    float* d_Sfull = nullptr;
    float* d_Vtfull = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_Usmall, sizeof(float) * (size_t)mB * mB));
    CHECK_CUDA(cudaMalloc((void**)&d_Sfull, sizeof(float) * (size_t)mB));
    CHECK_CUDA(cudaMalloc((void**)&d_Vtfull, sizeof(float) * (size_t)mB * nB));

    // cusolver gesvd requires workspace
    int lwork_svd = 0;
    signed char jobu = 'A';
    signed char jobvt = 'A';
    CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(solver_handle, mB, nB, &lwork_svd));
    float* d_work_svd = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_work_svd, sizeof(float) * lwork_svd));
    int* devInfo_svd = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&devInfo_svd, sizeof(int)));

    // gesvd overwrites input; B is column-major l x n
    // workspace: arrays required
    // Note: cusolverDnSgesvd requires signed char* for jobu/jobvt
    signed char jobu_c = 'S'; // compute thin U (min(m,n)) but mB <= nB so S yields mB x mB U
    signed char jobvt_c = 'S'; // compute thin Vt (mB x nB)
    CHECK_CUSOLVER(cusolverDnSgesvd(solver_handle, jobu_c, jobvt_c, mB, nB, d_B, mB, d_Sfull, d_Usmall, mB, d_Vtfull, mB, d_work_svd, lwork_svd, nullptr, devInfo_svd));

    // Now reconstruct U = Q * Usmall (m x l)*(l x l) -> take first k columns
    float* d_Ufull = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_Ufull, sizeof(float) * (size_t)m * l));
    // Use cuBLAS: d_Ufull = d_Y (Q) * d_Usmall
    const float c_one = 1.0f, c_zero = 0.0f;
    // Q (m x l) column-major, Usmall (l x l) column-major -> result m x l
    CHECK_CUBLAS(cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, l, l, &c_one, d_Y, m, d_Usmall, l, &c_zero, d_Ufull, m));

    // Allocate outputs trimmed to k
    CHECK_CUDA(cudaMalloc((void**)d_U, sizeof(float) * (size_t)m * k));
    CHECK_CUDA(cudaMalloc((void**)d_S, sizeof(float) * (size_t)k));
    CHECK_CUDA(cudaMalloc((void**)d_Vt, sizeof(float) * (size_t)k * n));

    // copy first k columns of Ufull into d_U (they are column-major contiguous blocks)
    CHECK_CUDA(cudaMemcpy2D(*d_U, sizeof(float) * m, d_Ufull, sizeof(float) * m, sizeof(float) * m, k, cudaMemcpyDeviceToDevice));
    // copy first k singular values
    CHECK_CUDA(cudaMemcpy(*d_S, d_Sfull, sizeof(float) * (size_t)k, cudaMemcpyDeviceToDevice));
    // copy first k rows of Vtfull (Vtfull is mB x n in column-major with leading dim mB)
    // We need k x n output in row-major expected? We'll keep column-major with leading dim k
    // Copy row-major slices: copy first k rows -> in memory they are the first k rows of Vtfull interleaved
    // Simplest: for each row r in 0..k-1, copy n elements with stride mB
    for (int r = 0; r < k; ++r) {
        // src pointer in device: &d_Vtfull[r] (since column-major, element (r,0) at offset r)
        CHECK_CUDA(cudaMemcpy2D((*d_Vt) + (size_t)r * n, sizeof(float) * n, d_Vtfull + r, sizeof(float) * mB, sizeof(float) * n, 1, cudaMemcpyDeviceToDevice));
        // The above is not ideal: copying row by row; acceptable for small k
    }

    // cleanup
    CHECK_CUDA(cudaFree(d_Omega));
    CHECK_CUDA(cudaFree(d_Y));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(d_tau));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaFree(d_Bt));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_Usmall));
    CHECK_CUDA(cudaFree(d_Sfull));
    CHECK_CUDA(cudaFree(d_Vtfull));
    CHECK_CUDA(cudaFree(d_work_svd));
    CHECK_CUDA(cudaFree(devInfo_svd));
    if (matZ) cusparseDestroySpMat(matZ);
    if (matOmega) cusparseDestroyDnMat(matOmega);
    if (matY) cusparseDestroyDnMat(matY);
    if (matQ) cusparseDestroyDnMat(matQ);
    if (matBt) cusparseDestroyDnMat(matBt);

    CHECK_CURAND(curandDestroyGenerator(curand_gen));
    CHECK_CUSOLVER(cusolverDnDestroy(solver_handle));
    CHECK_CUBLAS(cublasDestroy(blas_handle));
    CHECK_CUSPARSE(cusparseDestroy(sp_handle));

    return 0;
}

} // extern "C"

// transpose kernel: src is rows x cols (column-major?), but we will treat src as row-major n x l to transpose to l x n
__global__ void transpose_kernel(int rows, int cols, const float* src, float* dst) {
    // src: rows x cols (row-major) -> dst: cols x rows (row-major)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int r = idx / cols;
    int c = idx % cols;
    dst[c * rows + r] = src[r * cols + c];
}
