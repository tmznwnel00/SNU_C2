/*******************************************************
 * cocluster_main.cu
 *
 * Full pipeline runner for GPU-based Spectral Co-clustering.
 * Uses helper libraries:
 *  - coclustering_1.cu   (CSR degree + normalization)
 *  - coclustering_2.cu   (Randomized SVD)
 *  - coclustering_3.cu   (GPU k-means)
 *  - coclustering_4.cu   (Optimized k-means pipeline)
 *
 * Build:
 *   nvcc -c -O3 coclustering_1.cu -o coc1.o -lcusparse -lcublas
 *   nvcc -c -O3 coclustering_2.cu -o coc2.o -lcusparse -lcublas -lcusolver -lcurand
 *   nvcc -c -O3 coclustering_3.cu -o coc3.o -lcublas -lcurand
 *   nvcc -c -O3 coclustering_4.cu -o coc4.o -lcublas
 *
 *   nvcc cocluster_main.cu coc1.o coc2.o coc3.o coc4.o \
 *        -o cocluster \
 *        -lcusparse -lcublas -lcusolver -lcurand
 *******************************************************/

 #include <cuda_runtime.h>
 #include <cusparse.h>
 #include <cublas_v2.h>
 #include <cstdio>
 #include <vector>
 #include <fstream>
 #include <sstream>
 #include <iostream>
 #include <cmath>
 #include <chrono>
 
 // ===== extern "C" 함수들 (helper.cu 들에서 정의됨) =====
 extern "C" {
 void compute_degrees_csr(int m, int n, int nnz,
                          const int*, const int*, const float*, float*, float*);
 void normalize_csr_inplace(int m, int n, int nnz,
                            const int*, const int*, float*, const float*, const float*);
 int randomized_svd_csr(int m, int n, int nnz,
                        const int*, const int*, const float*,
                        int k, int p, int q,
                        float** d_U, float** d_S, float** d_Vt);
 int kmeans_fit(int n_points, int dim, const float* d_X, int k,
                int max_iters, int minibatch, float tol,
                float* d_centroids, int* d_labels);
 int kmeans_fit_pipeline(int n_points, int dim, const float* d_X,
                         int k, float* d_centroids,
                         int max_iters, int use_partial_reduce,
                         int* h_labels_out);
 }

__global__ void transpose_colmajor_small(int rows, int cols,
    const float* __restrict__ A, int lda,
    float* __restrict__ B, int ldb)
{
// B = A^T, column-major
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int total = rows * cols;
if (idx >= total) return;

int r = idx % rows;   // row index in A
int c = idx / rows;   // col index in A
// A(r,c) = A[r + c*lda]
float val = A[r + c * lda];
// B(c,r) = B[c + r*ldb]
B[c + r * ldb] = val;
}


 // =====================
 // CPU: Edge list → CSR
 // =====================
 void load_edge_list_to_CSR(const std::string& filename,
                            int& n, int& m, int& nnz,
                            std::vector<int>& rowPtr,
                            std::vector<int>& colIdx,
                            std::vector<float>& vals)
 {
     std::ifstream fin(filename);
     if (!fin.is_open()) {
         printf("ERROR: Cannot open %s\n", filename.c_str());
         exit(1);
     }
 
     int u, v;
     std::vector<std::pair<int,int>> edges;
     int maxNode = -1;
 
     while (fin >> u >> v) {
         edges.push_back({u, v});
         edges.push_back({v, u});  // undirected
         maxNode = std::max(maxNode, std::max(u, v));
     }
 
     m = n = maxNode + 1;
     nnz = edges.size();
 
     rowPtr.assign(m + 1, 0);
     colIdx.resize(nnz);
     vals.resize(nnz, 1.0f);
 
     // Count
     for (auto& e : edges) rowPtr[e.first + 1]++;
 
     // Prefix sum
     for (int i = 1; i <= m; i++) rowPtr[i] += rowPtr[i - 1];
 
     // Fill
     std::vector<int> counter(m, 0);
     for (auto& e : edges) {
         int r = e.first;
         int idx = rowPtr[r] + counter[r]++;
         colIdx[idx] = e.second;
     }
 }
 
 // ========= Helper ===============
 double ms(const std::chrono::steady_clock::time_point& a,
           const std::chrono::steady_clock::time_point& b)
 {
     return std::chrono::duration<double, std::milli>(b - a).count();
 }
 
 // ==================
 //        MAIN
 // ==================
 int main(int argc, char** argv)
 {
     if (argc < 2) {
         printf("Usage: ./cocluster <edge_list.txt>\n");
         return 0;
     }
 
     std::string file = argv[0+1];
 
     // ======== Load and build CSR ========
     int m, n, nnz;
     std::vector<int> h_rowPtr, h_colIdx;
     std::vector<float> h_vals;
 
     load_edge_list_to_CSR(file, n, m, nnz, h_rowPtr, h_colIdx, h_vals);
     printf("[INFO] Loaded %s: n=%d, nnz=%d\n", file.c_str(), n, nnz);
 
     // ======== Send CSR to GPU ========
     int *d_rowPtr, *d_colIdx;
     float *d_vals;
 
     cudaMalloc(&d_rowPtr, sizeof(int)*(m+1));
     cudaMalloc(&d_colIdx, sizeof(int)*nnz);
     cudaMalloc(&d_vals, sizeof(float)*nnz);
 
     cudaMemcpy(d_rowPtr, h_rowPtr.data(), sizeof(int)*(m+1), cudaMemcpyHostToDevice);
     cudaMemcpy(d_colIdx, h_colIdx.data(), sizeof(int)*nnz, cudaMemcpyHostToDevice);
     cudaMemcpy(d_vals, h_vals.data(), sizeof(float)*nnz, cudaMemcpyHostToDevice);
 
     // ========== Step 1: Degree + Normalize ==========
     std::vector<float> h_dr(m), h_dc(n);
     float *d_dr, *d_dc;
     cudaMalloc(&d_dr, sizeof(float)*m);
     cudaMalloc(&d_dc, sizeof(float)*n);
 
     auto t1 = std::chrono::steady_clock::now();
     compute_degrees_csr(m, n, nnz, d_rowPtr, d_colIdx, d_vals, d_dr, d_dc);
     normalize_csr_inplace(m, n, nnz, d_rowPtr, d_colIdx, d_vals, d_dr, d_dc);
     auto t2 = std::chrono::steady_clock::now();
     double step1_ms = ms(t1, t2);
 
     printf("[STEP 1] normalization: %.3f ms\n", step1_ms);
 
     // ========== Step 2: Randomized SVD ==========
     float *d_U = nullptr, *d_S = nullptr, *d_Vt = nullptr;
     int k = 10;   // user can change
     int p = 10;
     int q = 1;
 
     auto t3 = std::chrono::steady_clock::now();
     randomized_svd_csr(m, n, nnz, d_rowPtr, d_colIdx, d_vals, k, p, q, &d_U, &d_S, &d_Vt);
     auto t4 = std::chrono::steady_clock::now();
     double step2_ms = ms(t3, t4);
 
     printf("[STEP 2] randomized SVD: %.3f ms\n", step2_ms);
 
     // ========= Step 3: k-means on U =========
     int dim = k;
     int n_points = m;
     float *d_centroids;
     cudaMalloc(&d_centroids, sizeof(float)*k*k);
     int *d_labels;
     cudaMalloc(&d_labels, sizeof(int)*n_points);
 
     auto t5 = std::chrono::steady_clock::now();
     kmeans_fit(n_points, dim, d_U, k, 10, 0, 1e-4, d_centroids, d_labels);
     auto t6 = std::chrono::steady_clock::now();
     double step3_ms = ms(t5, t6);
 
     printf("[STEP 3] k-means(U): %.3f ms\n", step3_ms);

     //-------------------------------------------------------------
    // (NEW) Column clustering using V (n x k)
    //
    // Input: d_Vt (k x n) column-major
    // Convert Vt → V (n x k):
    //   V(i,j) = Vt(j,i)
    //-------------------------------------------------------------
    float* d_V = nullptr;   // n x k
    (cudaMalloc((void**)&d_V, sizeof(float) * (size_t)n * k));

    {
        int elems = n * k;
        int TP = 256;
        int GP = (elems + TP - 1) / TP;

        // V has shape n x k, ldV = n
        // Vt has shape k x n, ldVt = k
        transpose_colmajor_small<<<GP, TP>>>(
            k,        // rows in Vt
            n,        // cols in Vt
            d_Vt,     // source Vt
            k,        // lda
            d_V,      // destination V
            n         // ldb
        );
        (cudaGetLastError());
    }

    // Now run k-means on V (n x k)
    float* d_centroids_col = nullptr;
    (cudaMalloc((void**)&d_centroids_col, sizeof(float) * (size_t)k * k));

    int* d_labels_col = nullptr;
    (cudaMalloc((void**)&d_labels_col, sizeof(int) * (size_t)n));

    auto t5c = std::chrono::steady_clock::now();
    kmeans_fit(n,      // n_points = number of columns
            k,      // dim = embedding dimension
            d_V,    // X = V
            k,      // k clusters
            10,     // max_iters
            0,      // minibatch off
            1e-4,
            d_centroids_col,
            d_labels_col);
    auto t6c = std::chrono::steady_clock::now();
    double step3_col_ms = ms(t5c, t6c);

    printf("[STEP 3-COL] k-means(V): %.3f ms\n", step3_col_ms);

    // Copy to host
    std::vector<int> h_labels_col(n);
    (cudaMemcpy(h_labels_col.data(), d_labels_col,
                        sizeof(int) * (size_t)n,
                        cudaMemcpyDeviceToHost));

 
     // ========= Step 4: optimized pipeline(K-means) =========
     std::vector<int> h_labels(n_points);
 
     auto t7 = std::chrono::steady_clock::now();
     kmeans_fit_pipeline(n_points, dim, d_U, k, d_centroids, 10, 1, h_labels.data());
     auto t8 = std::chrono::steady_clock::now();
     double step4_ms = ms(t7, t8);
 
     printf("[STEP 4] optimized pipeline: %.3f ms\n", step4_ms);
 
     // ========= print final labels =========
     printf("ROW:");
     for (int i = 0; i < n_points; i++) printf(" %d", h_labels[i]);
     printf("\n");

     printf("COL:");
    for (int j = 0; j < n; j++) printf(" %d", h_labels_col[j]);
    printf("\n");

 
     return 0;
 }
 