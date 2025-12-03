/*
 * coclustering_4.cu
 *
 * Pipeline and system-level optimizations for coclustering/kmeans:
 * - Minimize host-device transfers: operate entirely on GPU; API returns only labels to host.
 * - Single-pass CSR/COO normalization applying left/right diagonal scalings in-place.
 * - K-means update using per-block partial reductions (shared-memory) and fused block->global updates
 *
 * Assumptions:
 * - Inputs live on device (GPU). Functions accept device pointers.
 * - Dense data for k-means is column-major (n_points x dim) for cuBLAS compatibility.
 * - For sparse normalization, CSR layout uses device `rowPtr, colIdx, vals`.
 *
 * Compile:
 *   nvcc -O3 -lcublas -o coclustering_4.o -c coclustering_4.cu
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ---- forward declarations (needed in CUDA) ----
extern "C" __global__ void compute_row_norms_kernel(const float*, int, int, float*);
extern "C" __global__ void compute_centroid_norms_kernel(const float*, int, int, int, float*);
extern "C" __global__ void fill_zero_float_kernel(float*, int);
extern "C" __global__ void fill_zero_int_kernel(int*, int);



#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
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


// ------------------ Single-pass normalization (CSR & COO) ------------------

// CSR single-pass: thread per row, iterate nonzeros in that row and apply vals[idx] /= sqrt(dr[row] * dc[col])
__global__ void normalize_csr_single_pass_kernel(int m, const int* __restrict__ rowPtr, const int* __restrict__ colIdx, float* __restrict__ vals, const float* __restrict__ dr, const float* __restrict__ dc) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;
    float drv = dr[r];
    int start = rowPtr[r];
    int end = rowPtr[r+1];
    for (int idx = start; idx < end; ++idx) {
        int c = colIdx[idx];
        float scale = drv * dc[c];
        float s = (scale > 0.f) ? sqrtf(scale) : 1.0f;
        vals[idx] = vals[idx] / s;
    }
}

// COO single-pass: thread per nonzero
__global__ void normalize_coo_single_pass_kernel(int nnz, float* __restrict__ vals, const int* __restrict__ rowIdx, const int* __restrict__ colIdx, const float* __restrict__ dr, const float* __restrict__ dc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int r = rowIdx[i];
    int c = colIdx[i];
    float scale = dr[r] * dc[c];
    float s = (scale > 0.f) ? sqrtf(scale) : 1.0f;
    vals[i] = vals[i] / s;
}

extern "C" void normalize_csr_single_pass(int m, int n, int nnz, const int* d_rowPtr, const int* d_colIdx, float* d_vals, const float* d_dr, const float* d_dc) {
    int tpb = 128;
    int g = (m + tpb - 1) / tpb;
    normalize_csr_single_pass_kernel<<<g, tpb>>>(m, d_rowPtr, d_colIdx, d_vals, d_dr, d_dc);
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void normalize_coo_single_pass(int nnz, float* d_vals, const int* d_rowIdx, const int* d_colIdx, const float* d_dr, const float* d_dc) {
    int tpb = 256;
    int g = (nnz + tpb - 1) / tpb;
    normalize_coo_single_pass_kernel<<<g, tpb>>>(nnz, d_vals, d_rowIdx, d_colIdx, d_dr, d_dc);
    CHECK_CUDA(cudaGetLastError());
}

// ------------------ K-means with per-block partial reduce ------------------

// assign_from_cross: given cross = X * C^T stored column-major (ld = n_points)
// compute label per-point and optionally output squared distance
__global__  void assign_from_cross_kernel(int n_points, int k, const float* __restrict__ cross, int ld_cross, const float* __restrict__ xnorms, const float* __restrict__ cnorms, int* __restrict__ labels, float* __restrict__ out_dist2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;
    float best = INFINITY;
    int best_j = 0;
    for (int j = 0; j < k; ++j) {
        float dot = cross[(size_t)j * ld_cross + i];
        float d2 = xnorms[i] + cnorms[j] - 2.0f * dot;
        if (d2 < best) { best = d2; best_j = j; }
    }
    labels[i] = best_j;
    if (out_dist2) out_dist2[i] = best;
}

// Per-block partial reduction kernel.
// Each block processes a tile of points. It accumulates per-centroid partial sums in shared memory
// and then one thread atomically adds the per-centroid partial sums to the global centroid accumulator.
// NOTE: This requires k * dim to be small enough to fit in shared memory. If not, caller should fall back to atomic-per-point.
__global__  void partial_block_update_kernel(const float* __restrict__ X, int n_points, int dim, const int* __restrict__ labels, int k, float* __restrict__ centroid_sums, int ld_cs, int* __restrict__ counts) {
    extern __shared__ float sdata[]; // dynamic shared memory
    // layout: first (k*dim) floats for sums, then k ints (as floats) for counts
    float* s_sums = sdata; // size k * dim
    int* s_counts = (int*)(s_sums + (size_t)k * dim);

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int block_start = blockIdx.x * blockDim.x;
    int block_end = min(n_points, block_start + bdim);

    // initialize shared memory to zero
    int total_sums = k * dim;
    for (int idx = tid; idx < total_sums; idx += bdim) s_sums[idx] = 0.0f;
    for (int idx = tid; idx < k; idx += bdim) s_counts[idx] = 0;
    __syncthreads();

    // each thread processes one point in the block (if available)
    int i = block_start + tid;
    if (i < n_points) {
        int lab = labels[i];
        atomicAdd(&s_counts[lab], 1); // per-block count
        // accumulate feature-wise into shared sums
        for (int d = 0; d < dim; ++d) {
            float v = X[(size_t)d * n_points + i];
            // s_sums indexed by (d * k + lab) to match column-major centroids layout
            atomicAdd(&s_sums[(size_t)d * k + lab], v);
        }
    }
    __syncthreads();

    // thread 0 of block flushes shared partial sums to global arrays using atomic adds
    if (tid == 0) {
        for (int d = 0; d < dim; ++d) {
            for (int j = 0; j < k; ++j) {
                float val = s_sums[(size_t)d * k + j];
                if (val != 0.0f) atomicAdd(&centroid_sums[(size_t)d * ld_cs + j], val);
            }
        }
        for (int j = 0; j < k; ++j) {
            int c = s_counts[j];
            if (c != 0) atomicAdd(&counts[j], c);
        }
    }
}

// Fallback atomic-per-point update when shared-memory per-block approach is not feasible
__global__  void update_centroids_atomic_kernel(const float* __restrict__ X, int n_points, int dim, const int* __restrict__ labels, int k, float* __restrict__ centroid_sums, int ld_cs, int* __restrict__ counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;
    int lab = labels[i];
    atomicAdd(&counts[lab], 1);
    for (int d = 0; d < dim; ++d) {
        float v = X[(size_t)d * n_points + i];
        atomicAdd(&centroid_sums[(size_t)d * ld_cs + lab], v);
    }
}

// finalize centroids dividing sums by counts
__global__  void finalize_centroids_kernel(float* __restrict__ centroid_sums, int k, int dim, const int* __restrict__ counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k * dim;
    if (idx >= total) return;
    int d = idx / k;
    int j = idx % k;
    int cnt = counts[j];
    if (cnt > 0) centroid_sums[(size_t)d * k + j] /= (float)cnt;
}

// Top-level pipeline: assumes X and (optionally) initial centroids are already on device.
// Runs optional sparse normalization (if CSR args provided), then k-means entirely on GPU.
// Copies only `labels` back to host at the end to minimize host-device transfers.
extern "C" int kmeans_fit_pipeline(int n_points, int dim, const float* d_X, // dense data: n_points x dim column-major
                                    int k, float* d_centroids, // centroids on device (k x dim column-major)
                                    int max_iters, int use_partial_reduce, // 1 to use shared-memory partial reduce
                                    int* h_labels_out) { // host pointer to receive labels
    // create cublas handle
    cublasHandle_t blas = nullptr;
    CHECK_CUBLAS(cublasCreate(&blas));

    // allocate working buffers on device
    float* d_cross = nullptr; // n_points x k
    CHECK_CUDA(cudaMalloc((void**)&d_cross, sizeof(float) * (size_t)n_points * k));
    float* d_xnorms = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_xnorms, sizeof(float) * (size_t)n_points));
    int* d_labels = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_labels, sizeof(int) * (size_t)n_points));
    float* d_cnorms = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_cnorms, sizeof(float) * (size_t)k));

    // centroid sums and counts
    float* d_centroid_sums = nullptr; // k x dim (column-major: for each d, k entries)
    CHECK_CUDA(cudaMalloc((void**)&d_centroid_sums, sizeof(float) * (size_t)k * dim));
    int* d_counts = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_counts, sizeof(int) * (size_t)k));

    // precompute x norms
    int tpb = 128;
    int gx = (n_points + tpb - 1) / tpb;
    // compute row norms kernel
    // reuse small kernel implementation from earlier files: compute row norms by summing over columns
    // implement inline simple kernel here
    // (Note: for large n/dim consider using cublas gemv or more optimized kernel)
    // {
    //     // kernel
    //     auto compute_row_norms = [] __device__ (const float* X, int n_points, int dim, float* norms) {};
    // }

    // We'll implement compute_row_norms on host using a simple kernel declared below
    compute_row_norms_kernel<<<gx, tpb>>>(d_X, n_points, dim, d_xnorms);
    CHECK_CUDA(cudaGetLastError());

    for (int iter = 0; iter < max_iters; ++iter) {
        // Assignment: cross = X * C^T  (n_points x k)
        const float alpha = 1.0f, beta = 0.0f;
        // cublasSgemm: C = alpha*A*B + beta*C
        // A = X (n_points x dim), lda = n_points. B = d_centroids (k x dim) but we want C^T (dim x k) -> pass B with transpose
        CHECK_CUBLAS(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, n_points, k, dim, &alpha, d_X, n_points, d_centroids, k, &beta, d_cross, n_points));

        // compute centroid norms
        int gk = (k + tpb - 1) / tpb;
        compute_centroid_norms_kernel<<<gk, tpb>>>(d_centroids, k, dim, k, d_cnorms);
        CHECK_CUDA(cudaGetLastError());

        // assign labels
        int g_ass = (n_points + tpb - 1) / tpb;
        assign_from_cross_kernel<<<g_ass, tpb>>>(n_points, k, d_cross, n_points, d_xnorms, d_cnorms, d_labels, nullptr);
        CHECK_CUDA(cudaGetLastError());

        // reset centroid_sums and counts
        int total = k * dim;
        int gfill = (total + tpb - 1) / tpb;
        // zero floats
        // auto fill_zero_f = [] __device__ (float* x, int n) {};
        fill_zero_float_kernel<<<gfill, tpb>>>(d_centroid_sums, total);
        CHECK_CUDA(cudaGetLastError());
        // zero ints
        fill_zero_int_kernel<<<(k + tpb - 1)/tpb, tpb>>>(d_counts, k);
        CHECK_CUDA(cudaGetLastError());

        // partial reduce or atomic fallback
        if (use_partial_reduce) {
            // compute required shared mem: k*dim floats + k ints
            size_t sh_mem = sizeof(float) * (size_t)k * dim + sizeof(int) * (size_t)k;
            // limit check: if shared mem is too large, fallback to atomic kernel
            cudaDeviceProp prop;
            CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
            if (sh_mem <= prop.sharedMemPerBlock) {
                int blocks = (n_points + tpb - 1) / tpb;
                partial_block_update_kernel<<<blocks, tpb, sh_mem>>>(d_X, n_points, dim, d_labels, k, d_centroid_sums, k, d_counts);
                CHECK_CUDA(cudaGetLastError());
            } else {
                int blocks = (n_points + tpb - 1) / tpb;
                update_centroids_atomic_kernel<<<blocks, tpb>>>(d_X, n_points, dim, d_labels, k, d_centroid_sums, k, d_counts);
                CHECK_CUDA(cudaGetLastError());
            }
        } else {
            int blocks = (n_points + tpb - 1) / tpb;
            update_centroids_atomic_kernel<<<blocks, tpb>>>(d_X, n_points, dim, d_labels, k, d_centroid_sums, k, d_counts);
            CHECK_CUDA(cudaGetLastError());
        }

        // finalize centroids (divide by counts)
        int gfin = (k * dim + tpb - 1) / tpb;
        finalize_centroids_kernel<<<gfin, tpb>>>(d_centroid_sums, k, dim, d_counts);
        CHECK_CUDA(cudaGetLastError());

        // write back centroid_sums into d_centroids
        CHECK_CUDA(cudaMemcpy(d_centroids, d_centroid_sums, sizeof(float) * (size_t)k * dim, cudaMemcpyDeviceToDevice));
    }

    // copy labels back to host (single transfer)
    CHECK_CUDA(cudaMemcpy(h_labels_out, d_labels, sizeof(int) * (size_t)n_points, cudaMemcpyDeviceToHost));

    // cleanup
    CHECK_CUDA(cudaFree(d_cross));
    CHECK_CUDA(cudaFree(d_xnorms));
    CHECK_CUDA(cudaFree(d_labels));
    CHECK_CUDA(cudaFree(d_cnorms));
    CHECK_CUDA(cudaFree(d_centroid_sums));
    CHECK_CUDA(cudaFree(d_counts));
    CHECK_CUBLAS(cublasDestroy(blas));

    return 0;
}
