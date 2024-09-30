import numpy as np
import time
import sys

# for a simple loop
def dp(N,A,B):
    R = 0.0;
    for j in range(0,N):
       R += A[j]*B[j]
    return R

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <vector_size> <repetitions>")
        return

    N = int(sys.argv[1])
    repetitions = int(sys.argv[2])

    # Initialize arrays
    pA = np.ones(N, dtype=np.float32)
    pB = np.ones(N, dtype=np.float32)

    total_time = 0.0

    # Perform the measurements
    for i in range(repetitions):
        start_time = time.monotonic()
        ans = np.dot(pA, pB)
        end_time = time.monotonic()
        print(f"Answer: {ans}")
        time_taken = end_time - start_time

        if i >= repetitions // 2:  # Only average the second half
            total_time += time_taken

    # Compute average time for the second half
    average_time = total_time / (repetitions // 2)
    
    # Compute bandwidth and FLOP
    bandwidth = (2.0 * N * 4) / average_time / (1024 ** 3)  # GB/sec (4 bytes for float32)
    flops = (2.0 * N) / average_time  # FLOP/sec

    # Print results
    print(f"N: {N}  <T>: {average_time:.6f} sec  B: {bandwidth:.6f} GB/sec  F: {flops:.6f} FLOP/sec")

if __name__ == "__main__":
    main()
