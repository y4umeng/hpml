#include <stdio.h>
#include <time.h>
#include <stdlib.h>

float dpunroll(long N, float *pA, float *pB) {
  float R = 0.0;
  int j;
  for (j=0;j<N;j+=4)
    R += pA[j]*pB[j] + pA[j+1]*pB[j+1] \
           + pA[j+2]*pB[j+2] + pA[j+3] * pB[j+3];
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
        float ans = dpunroll(N, pA, pB);
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

    // Compute average time for the second half
    double average_time = total_time / (iter / 2);
    
    // Compute bandwidth and FLOP
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