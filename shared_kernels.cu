// shared_kernels.cu
#include <cuda_runtime.h>

extern "C" {

// 1) row norms
__global__ void compute_row_norms_kernel(const float* X, int n_points, int dim, float* norms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;
    float s = 0.0f;
    for (int d = 0; d < dim; ++d)
        s += X[(size_t)d * n_points + i] * X[(size_t)d * n_points + i];
    norms[i] = s;
}

// 2) centroid norms
__global__ void compute_centroid_norms_kernel(const float* centroids, int k, int dim, int cur_k, float* cnorms) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cur_k) return;
    float s = 0.0f;
    for (int d = 0; d < dim; ++d)
        s += centroids[(size_t)d * k + j] * centroids[(size_t)d * k + j];
    cnorms[j] = s;
}

// 3) zero float
__global__ void fill_zero_float_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

// 4) zero int
__global__ void fill_zero_int_kernel(int* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0;
}

} // extern "C"
