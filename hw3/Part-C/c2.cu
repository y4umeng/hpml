#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define C 3
#define H 1024
#define W 1024
#define FW 3
#define FH 3
#define K 64
#define P 1  // Padding size

// Kernel function to perform 2D convolution using shared memory and tiling
__global__ void convolutionShared(const double* I0, const double* F, double* O, size_t width, size_t height) {
    extern __shared__ double tile[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Shared memory allocation
    int tileWidth = blockDim.x + FW - 1;
    int tileHeight = blockDim.y + FH - 1;
    
    // Loop over filters
    for (int k = 0; k < K; ++k) {
        if (x < width && y < height) {
            double result = 0.0;

            // Load tile from global memory to shared memory
            for (int c = 0; c < C; ++c) {
                for (int i = 0; i < tileHeight; i += blockDim.y) {
                    for (int j = 0; j < tileWidth; j += blockDim.x) {
                        int loadX = blockIdx.x * blockDim.x - P + tx + j;
                        int loadY = blockIdx.y * blockDim.y - P + ty + i;
                        if (loadX >= 0 && loadX < width + 2 * P && loadY >= 0 && loadY < height + 2 * P) {
                            tile[(ty + i) * tileWidth + (tx + j)] = I0[(c * (width + 2 * P) * (height + 2 * P)) + loadY * (width + 2 * P) + loadX];
                        } else {
                            tile[(ty + i) * tileWidth + (tx + j)] = 0.0;
                        }
                    }
                }
                __syncthreads();

                // Convolution for the output
                if (tx < blockDim.x && ty < blockDim.y) {
                    for (int i = 0; i < FH; ++i) {
                        for (int j = 0; j < FW; ++j) {
                            result += F[(k * C * FH * FW) + (c * FH * FW) + (FH - 1 - i) * FW + (FW - 1 - j)] *
                                      tile[(ty + i) * tileWidth + (tx + j)];
                        }
                    }
                }
                __syncthreads();
            }
            // Write the result back to the global memory
            if (x < width && y < height) {
                O[(k * width * height) + y * width + x] = result;
            }
        }
    }
}

int main() {
    size_t size_I0 = C * (H + 2 * P) * (W + 2 * P) * sizeof(double);
    size_t size_F = K * C * FH * FW * sizeof(double);
    size_t size_O = K * H * W * sizeof(double);

    // Allocate memory on the host
    double* I0;
    double* F;
    double* O;

    cudaMallocManaged(&I0, size_I0);
    cudaMallocManaged(&F, size_F);
    cudaMallocManaged(&O, size_O);

    // Initialize input and filter data
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < H; ++x) {
            for (int y = 0; y < W; ++y) {
                I0[c * (H + 2 * P) * (W + 2 * P) + (x + P) * (W + 2 * P) + (y + P)] = c * (x + y);
            }
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y);
    size_t sharedMemSize = (blockDim.x + FW - 1) * (blockDim.y + FH - 1) * sizeof(double);

    // Launch the kernel and measure the time
    auto start = std::chrono::high_resolution_clock::now();
    convolutionShared<<<gridDim, blockDim, sharedMemSize>>>(I0, F, O, W, H);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time with shared memory and tiling: " << elapsed.count() << " seconds" << std::endl;

    // Calculate checksum
    double checksum = 0.0;
    for (int k = 0; k < K; ++k) {
        for (int x = 0; x < H; ++x) {
            for (int y = 0; y < W; ++y) {
                checksum += O[k * W * H + x * W + y];
            }
        }
    }
    std::cout << "Checksum: " << checksum << std::endl;

    // Free memory
    cudaFree(I0);
    cudaFree(F);
    cudaFree(O);

    return 0;
}
