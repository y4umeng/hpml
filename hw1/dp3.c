#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mkl_cblas.h>

float bdp(long N, float *pA, float *pB) {
    float R = cblas_sdot(N, pA, 1, pB, 1);
    return R; 
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <vector_size> <repetitions>\n", argv[0]);
        return EXIT_FAILURE;
    }

    long N = atol(argv[1]);
    int iter = atoi(argv[2]);
    struct timespec start, end;

    // Allocate memory for input arrays
    float *pA = (float *)malloc(N * sizeof(float));
    float *pB = (float *)malloc(N * sizeof(float));
    
    if (pA == NULL || pB == NULL) {
        perror("Failed to allocate memory");
        return EXIT_FAILURE;
    }

    // Initialize arrays
    for (long i = 0; i < N; i++) {
        pA[i] = 1.0f;
        pB[i] = 1.0f;
    }

    double total_time = 0.0;

    // Perform the measurements
    for (int i = 0; i < iter; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        float ans = bdp(N, pA, pB);
        clock_gettime(CLOCK_MONOTONIC, &end);

        // Print result to ensure computation is performed
        printf("Dot product result: %.2f\n", ans);    

        double time_taken = (end.tv_sec - start.tv_sec) + 
                            (end.tv_nsec - start.tv_nsec) / 1e9;

        if (i >= iter / 2) { // Only average the second half
            total_time += time_taken;
        }

        // printf("Iteration %d: %f seconds\n", i + 1, time_taken);
    }

    // Compute average time for the second half (Arithmetic Mean)
    double average_time = total_time / (iter / 2);
    
    // Compute bandwidth and FLOP (Harmonic Mean)
    double bandwidth = (2.0 * N * sizeof(float) / average_time) / (1024 * 1024 * 1024); // GB/sec
    double flops = (2.0 * N / average_time); // FLOP/sec

    // Print results
    printf("N: %ld  <T>: %.6f sec  B: %.6f GB/sec  F: %.6f FLOP/sec\n", 
           N, average_time, bandwidth, flops);

    // Free allocated memory
    free(pA);
    free(pB);

    return EXIT_SUCCESS;
}