#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Problem dimensions
#define H 1024
#define W 1024
#define C 3
#define FH 3
#define FW 3
#define K 64
#define P 1

// Tile dimensions
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Shared memory tile size includes padding
#define SHARED_TILE_WIDTH (TILE_WIDTH + FW - 1)
#define SHARED_TILE_HEIGHT (TILE_HEIGHT + FH - 1)

__global__ void convolution_tiled_kernel(double* I0, double* F, double* O) {
    __shared__ double shared_input[C][SHARED_TILE_HEIGHT][SHARED_TILE_WIDTH];
    __shared__ double shared_filter[K][C][FH][FW];
    
    // Global indices
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    int k = blockIdx.z;
    
    // Local thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Load filters into shared memory
    // Each thread in the block helps load the filters for all K outputs
    for (int f_k = 0; f_k < K; f_k++) {
        for (int c = 0; c < C; c++) {
            for (int i = threadIdx.y; i < FH; i += TILE_HEIGHT) {
                for (int j = threadIdx.x; j < FW; j += TILE_WIDTH) {
                    if (i < FH && j < FW) {
                        shared_filter[f_k][c][i][j] = F[f_k*C*FH*FW + c*FH*FW + i*FW + j];
                    }
                }
            }
        }
    }
    
    // Ensure all filters are loaded
    __syncthreads();
    
    // Check if this thread needs to compute an output
    if (x < H && y < W) {
        // Load input tile into shared memory for each channel
        for (int c = 0; c < C; c++) {
            for (int i = ty; i < SHARED_TILE_HEIGHT; i += TILE_HEIGHT) {
                for (int j = tx; j < SHARED_TILE_WIDTH; j += TILE_WIDTH) {
                    int global_x = blockIdx.x * TILE_WIDTH + i - P;
                    int global_y = blockIdx.y * TILE_HEIGHT + j - P;
                    
                    if (global_x >= 0 && global_x < H+2*P && global_y >= 0 && global_y < W+2*P) {
                        shared_input[c][i][j] = I0[c*(H+2*P)*(W+2*P) + global_x*(W+2*P) + global_y];
                    } else {
                        shared_input[c][i][j] = 0.0;
                    }
                }
            }
        }
        
        // Ensure all input data is loaded
        __syncthreads();
        
        // Compute convolution for this output pixel
        double sum = 0.0;
        
        // Local indices in the shared memory tile
        int local_x = threadIdx.x;
        int local_y = threadIdx.y;
        
        for (int c = 0; c < C; c++) {
            for (int j = 0; j < FH; j++) {
                for (int i = 0; i < FW; i++) {
                    sum += shared_filter[k][c][FH-1-j][FW-1-i] * 
                           shared_input[c][local_x + i][local_y + j];
                }
            }
        }
        
        // Write output
        O[k*H*W + x*W + y] = sum;
    }
}

int main() {
    // Allocate host memory
    double *h_I = (double*)malloc(C * H * W * sizeof(double));
    double *h_I0 = (double*)malloc(C * (H+2*P) * (W+2*P) * sizeof(double));
    double *h_F = (double*)malloc(K * C * FH * FW * sizeof(double));
    double *h_O = (double*)malloc(K * H * W * sizeof(double));
    
    // Initialize input tensor I
    for (int c = 0; c < C; c++) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                h_I[c*H*W + x*W + y] = c * (x + y);
            }
        }
    }
    
    // Initialize filters F
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FH; i++) {
                for (int j = 0; j < FW; j++) {
                    h_F[k*C*FH*FW + c*FH*FW + i*FW + j] = (c + k) * (i + j);
                }
            }
        }
    }
    
    // Create padded input I0 (initialized to 0)
    memset(h_I0, 0, C * (H+2*P) * (W+2*P) * sizeof(double));
    for (int c = 0; c < C; c++) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                h_I0[c*(H+2*P)*(W+2*P) + (x+P)*(W+2*P) + (y+P)] = h_I[c*H*W + x*W + y];
            }
        }
    }
    
    // Allocate device memory
    double *d_I0, *d_F, *d_O;
    cudaMalloc(&d_I0, C * (H+2*P) * (W+2*P) * sizeof(double));
    cudaMalloc(&d_F, K * C * FH * FW * sizeof(double));
    cudaMalloc(&d_O, K * H * W * sizeof(double));
    
    // Copy data to device
    cudaMemcpy(d_I0, h_I0, C * (H+2*P) * (W+2*P) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, K * C * FH * FW * sizeof(double), cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    dim3 blockDim(TILE_WIDTH, TILE_HEIGHT, 1);
    dim3 gridDim((H + TILE_WIDTH - 1) / TILE_WIDTH,
                 (W + TILE_HEIGHT - 1) / TILE_HEIGHT,
                 K);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch kernel
    convolution_tiled_kernel<<<gridDim, blockDim>>>(d_I0, d_F, d_O);
    
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);
    
    // Copy result back to host
    cudaMemcpy(h_O, d_O, K * H * W * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Calculate checksum
    double checksum = 0.0;
    for (int k = 0; k < K; k++) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                checksum += h_O[k*H*W + x*W + y];
            }
        }
    }
    printf("Checksum: %.6e\n", checksum);
    
    // Cleanup
    cudaFree(d_I0);
    cudaFree(d_F);
    cudaFree(d_O);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_I);
    free(h_I0);
    free(h_F);
    free(h_O);
    
    return 0;
}