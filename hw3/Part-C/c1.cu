#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define H 1024
#define W 1024
#define C 3
#define FH 3
#define FW 3
#define K 64
#define P 1

// Kernel for performing convolution
__global__ void convolutionKernel(const float* I0, const float* F, float* O) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && k < K) {
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    int ix = x + i;
                    int iy = y + j;
                    sum += F[k * C * FH * FW + c * FH * FW + (FW - 1 - i) * FW + (FH - 1 - j)] 
                           * I0[c * (W + 2 * P) * (H + 2 * P) + iy * (W + 2 * P) + ix];
                }
            }
        }
        O[k * W * H + y * W + x] = sum;
    }
}

// Host function to initialize tensors
void initializeInput(float* I) {
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < H; ++x) {
            for (int y = 0; y < W; ++y) {
                I[c * H * W + x * W + y] = c * (x + y);
            }
        }
    }
}

void initializeFilter(float* F) {
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }
}

void addPadding(const float* I, float* I0) {
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < H + 2 * P; ++x) {
            for (int y = 0; y < W + 2 * P; ++y) {
                if (x < P || x >= H + P || y < P || y >= W + P) {
                    I0[c * (H + 2 * P) * (W + 2 * P) + x * (W + 2 * P) + y] = 0.0f;
                } else {
                    I0[c * (H + 2 * P) * (W + 2 * P) + x * (W + 2 * P) + y] = I[c * H * W + (x - P) * W + (y - P)];
                }
            }
        }
    }
}

int main() {
    // Allocate Unified Memory for input, filter, and output tensors
    float *I, *I0, *F, *O;
    cudaMallocManaged(&I, C * H * W * sizeof(float));
    cudaMallocManaged(&I0, C * (H + 2 * P) * (W + 2 * P) * sizeof(float));
    cudaMallocManaged(&F, K * C * FH * FW * sizeof(float));
    cudaMallocManaged(&O, K * W * H * sizeof(float));

    // Initialize input tensor I and filter F
    initializeInput(I);
    initializeFilter(F);
    addPadding(I, I0);

    // Define the block and grid sizes
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
                 (H + blockDim.y - 1) / blockDim.y,
                 (K + blockDim.z - 1) / blockDim.z);

    // Run the convolution kernel and measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    convolutionKernel<<<gridDim, blockDim>>>(I0, F, O);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate and display elapsed time
    std::chrono::duration<float> elapsed = end - start;
    std::cout << "Kernel execution time: " << elapsed.count() << " seconds" << std::endl;

    // Calculate checksum
    float checksum = 0.0f;
    for (int i = 0; i < K * W * H; ++i) {
        checksum += O[i];
    }
    std::cout << "Checksum (sum of all elements in O): " << checksum << std::endl;

    // Free Unified Memory
    cudaFree(I);
    cudaFree(I0);
    cudaFree(F);
    cudaFree(O);

    return 0;
}
