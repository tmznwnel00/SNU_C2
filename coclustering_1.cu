/*
 * coclustering_1.cu
 *
 * GPU helpers for coclustering normalization.
 * - Computes row/column degree vectors using cuSPARSE SpMV (d_r = A * 1, d_c = A^T * 1)
 * - In-place normalization kernel: Z_ij <- X_ij / sqrt(d_r[i] * d_c[j])
 *
 * Assumptions:
 * - Input sparse matrix is stored on device in CSR (rowPtr, colIdx, values) or COO (rowIdx, colIdx, values)
 * - All arrays live on device (GPU)
 * - Values are 32-bit floats
 *
 * Compile example:
 *   nvcc -O3 -lcusparse -o coclustering_1.o -c coclustering_1.cu
 *
 */

#include <cuda_runtime.h>
#include <cusparse.h>
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

// Fill dense vector with ones
__global__ static void fill_ones_kernel(int n, float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f;
}

// Build per-nnz row index array from CSR rowPtr: for idx in [rowPtr[r], rowPtr[r+1]) set row_idx[idx]=r
__global__ static void build_row_indices_kernel(int m, const int* __restrict__ rowPtr, int* __restrict__ row_idx) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;
    int start = rowPtr[r];
    int end = rowPtr[r+1];
    for (int k = start; k < end; ++k) row_idx[k] = r;
}

// Per-nnz normalization: values[i] /= sqrt( dr[row_idx[i]] * dc[col_idx[i]] )
__global__ static void normalize_inplace_kernel(int nnz, float* __restrict__ vals, const int* __restrict__ row_idx, const int* __restrict__ col_idx, const float* __restrict__ dr, const float* __restrict__ dc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int r = row_idx[i];
    int c = col_idx[i];
    float s = dr[r] * dc[c];
    float scale = (s > 0.f) ? sqrtf(s) : 1.0f;
    vals[i] = vals[i] / scale;
}

extern "C" {

// Compute row and column degree vectors for CSR matrix using cuSPARSE SpMV.
// Inputs (device pointers): rowPtr (size m+1), colIdx (size nnz), vals (size nnz)
// Outputs (device pointers): d_r (size m), d_c (size n)
// Note: this routine allocates a small temporary buffer and a ones vector internally.
void compute_degrees_csr(int m, int n, int nnz, const int* d_rowPtr, const int* d_colIdx, const float* d_vals, float* d_r, float* d_c) {
    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create sparse matrix descriptor (CSR)
    cusparseSpMatDescr_t matA = nullptr;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
                                     m, n, nnz,
                                     (void*)d_rowPtr, (void*)d_colIdx, (void*)d_vals,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Temporary dense vector of ones (size n)
    float* d_ones = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_ones, sizeof(float) * (size_t)n));
    int tpb = 256;
    int g = (n + tpb - 1) / tpb;
    fill_ones_kernel<<<g, tpb>>>(n, d_ones);
    CHECK_CUDA(cudaGetLastError());

    // Descriptors for dense vectors
    cusparseDnVecDescr_t vecX = nullptr; // ones
    cusparseDnVecDescr_t vecY = nullptr; // result

    // Compute d_r = A * 1 (size m)
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, (void*)d_ones, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, m, (void*)d_r, CUDA_R_32F));

    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                           CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    // Compute d_c = A^T * 1 (size n) -> reuse ones vector of size m for right-hand side
    // Need ones vector of length m
    float* d_ones_m = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_ones_m, sizeof(float) * (size_t)m));
    int gm = (m + tpb - 1) / tpb;
    fill_ones_kernel<<<gm, tpb>>>(m, d_ones_m);
    CHECK_CUDA(cudaGetLastError());

    // recreate vecX/vecY for transpose
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, m, (void*)d_ones_m, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, n, (void*)d_c, CUDA_R_32F));

    // buffer may need different size; requery
    size_t bufferSize2 = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,
                                           &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                           CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize2));
    if (bufferSize2 > bufferSize) {
        CHECK_CUDA(cudaFree(dBuffer));
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize2));
    }

    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    // cleanup
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(d_ones));
    CHECK_CUDA(cudaFree(d_ones_m));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}

// Wrapper to normalize CSR in-place. Allocates a temporary row_idx array and runs normalization kernel.
// Inputs (device): rowPtr (m+1), colIdx (nnz), vals (nnz), dr (m), dc (n)
void normalize_csr_inplace(int m, int n, int nnz, const int* d_rowPtr, const int* d_colIdx, float* d_vals, const float* d_dr, const float* d_dc) {
    int* d_row_idx = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_row_idx, sizeof(int) * (size_t)nnz));
    int tpb = 128;
    int g = (m + tpb - 1) / tpb;
    build_row_indices_kernel<<<g, tpb>>>(m, d_rowPtr, d_row_idx);
    CHECK_CUDA(cudaGetLastError());

    int g2 = (nnz + tpb - 1) / tpb;
    normalize_inplace_kernel<<<g2, tpb>>>(nnz, d_vals, d_row_idx, d_colIdx, d_dr, d_dc);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_row_idx));
}

// Wrapper to normalize COO in-place. rowIdx/coIIdx/vals all device pointers.
void normalize_coo_inplace(int nnz, float* d_vals, const int* d_rowIdx, const int* d_colIdx, const float* d_dr, const float* d_dc) {
    int tpb = 128;
    int g = (nnz + tpb - 1) / tpb;
    normalize_inplace_kernel<<<g, tpb>>>(nnz, d_vals, d_rowIdx, d_colIdx, d_dr, d_dc);
    CHECK_CUDA(cudaGetLastError());
}

} // extern "C"
