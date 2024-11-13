#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define H 1024
#define W 1024
#define C 3
#define K 64
#define FW 3
#define FH 3
#define P 1  // Padding

__global__ void tiledConvolution(const double* I0, const double* F, double* O, int width, int height) {
    extern __shared__ double tile[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    double result = 0.0;

    if (x < width && y < height) {
        for (int k = 0; k < K; ++k) {
            result = 0.0;

            for (int c = 0; c < C; ++c) {
                // Load tile with padding
                int tile_x = threadX;
                int tile_y = threadY;
                int input_x = x - P + tile_x;
                int input_y = y - P + tile_y;

                if (input_x >= 0 && input_x < width + 2 * P && input_y >= 0 && input_y < height + 2 * P) {
                    tile[tile_y * blockDim.x + tile_x] = I0[c * (width + 2 * P) * (height + 2 * P) + input_y * (width + 2 * P) + input_x];
                } else {
                    tile[tile_y * blockDim.x + tile_x] = 0.0;
                }

                __syncthreads();

                // Perform convolution
                for (int i = 0; i < FH; ++i) {
                    for (int j = 0; j < FW; ++j) {
                        int tx = FW - 1 - i;
                        int ty = FH - 1 - j;
                        int ix = threadX + i;
                        int iy = threadY + j;

                        if (ix < blockDim.x && iy < blockDim.y) {
                            result += F[(k * C + c) * FW * FH + tx * FW + ty] * tile[iy * blockDim.x + ix];
                        }
                    }
                }
                __syncthreads();
            }
            O[k * width * height + y * width + x] = result;
        }
    }
}

int main() {
    size_t imageSize = C * (H + 2 * P) * (W + 2 * P) * sizeof(double);
    size_t filterSize = K * C * FH * FW * sizeof(double);
    size_t outputSize = K * H * W * sizeof(double);

    double *I0, *F, *O;

    cudaMallocManaged(&I0, imageSize);
    cudaMallocManaged(&F, filterSize);
    cudaMallocManaged(&O, outputSize);

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
                    F[(k * C + c) * FH * FW + i * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x, (H + blockSize.y - 1) / blockSize.y);
    size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(double);

    auto start = std::chrono::high_resolution_clock::now();
    tiledConvolution<<<gridSize, blockSize, sharedMemSize>>>(I0, F, O, W, H);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution Time: " << elapsed.count() << " seconds" << std::endl;

    double checksum = 0.0;
    for (int k = 0; k < K; ++k) {
        for (int x = 0; x < H; ++x) {
            for (int y = 0; y < W; ++y) {
                checksum += O[k * H * W + x * W + y];
            }
        }
    }

    std::cout << "Checksum: " << checksum << std::endl;

    cudaFree(I0);
    cudaFree(F);
    cudaFree(O);

    return 0;
}
