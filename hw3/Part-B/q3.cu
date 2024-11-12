#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void addKernel(const float* A, const float* B, float* C, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void addVectorsOnGPU(float* A, float* B, float* C, size_t N, int numThreads, bool singleBlock) {
    // We don't need separate device pointers with Unified Memory,
    // since A, B, and C are allocated in unified memory space.

    int numBlocks = singleBlock ? 1 : (N + numThreads - 1) / numThreads;

    // Launch the kernel and measure time
    auto start = std::chrono::high_resolution_clock::now();
    addKernel<<<numBlocks, numThreads>>>(A, B, C, N);
    cudaDeviceSynchronize(); // Wait for GPU to finish
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate and display elapsed time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time with " << numBlocks << " block(s) and " << numThreads << " threads per block: "
              << elapsed.count() << " seconds" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <K>" << std::endl;
        return 1;
    }

    int K = std::atoi(argv[1]);
    if (K <= 0) {
        std::cerr << "K must be a positive integer." << std::endl;
        return 1;
    }

    // Total number of elements
    size_t N = K * 1000000;

    // Allocate Unified Memory accessible from CPU or GPU
    float *A, *B, *C;
    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));
    cudaMallocManaged(&C, N * sizeof(float));

    // Initialize arrays on the host (CPU)
    for (size_t i = 0; i < N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    std::cout << "Running with K = " << K << " million elements (" << N << " total elements)\n";

    // Scenario 1: One block with one thread
    std::cout << "\nScenario 1: One block with 1 thread\n";
    addVectorsOnGPU(A, B, C, N, 1, true);

    // Scenario 2: One block with 256 threads
    std::cout << "\nScenario 2: One block with 256 threads\n";
    addVectorsOnGPU(A, B, C, N, 256, true);

    // Scenario 3: Multiple blocks with 256 threads per block
    std::cout << "\nScenario 3: Multiple blocks with 256 threads per block\n";
    addVectorsOnGPU(A, B, C, N, 256, false);

    // Free Unified Memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
