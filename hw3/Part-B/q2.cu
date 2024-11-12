#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void addKernel(const float* A, const float* B, float* C, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void addVectorsOnGPU(float* h_A, float* h_B, float* h_C, size_t N, int numBlocks, int numThreads) {
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the kernel and measure time
    auto start = std::chrono::high_resolution_clock::now();
    addKernel<<<numBlocks, numThreads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Wait for GPU to finish
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate and display elapsed time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time with " << numBlocks << " blocks and " << numThreads << " threads per block: "
              << elapsed.count() << " seconds" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <K> <numBlocks>" << std::endl;
        return 1;
    }

    int K = std::atoi(argv[1]);
    int numBlocks = std::atoi(argv[2]);
    if (K <= 0 || numBlocks <= 0) {
        std::cerr << "K and numBlocks must be positive integers." << std::endl;
        return 1;
    }

    // Total number of elements
    size_t N = K * 1000000;

    // Allocate memory on the host
    float* h_A = (float*)malloc(N * sizeof(float));
    float* h_B = (float*)malloc(N * sizeof(float));
    float* h_C = (float*)malloc(N * sizeof(float));

    // Initialize arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    std::cout << "Running with K = " << K << " million elements (" << N << " total elements) and " 
             
