import numpy as np
import time

# Set matrix size (e.g., 500 x 500)
N = 400
A = np.random.rand(N, N)
B = np.random.rand(N, N)

# Time the multiplication
start_time = time.time()

# Standard matrix multiplication (3 nested loops)
C = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i][j] += A[i][k] * B[k][j]

end_time = time.time()

print(f"Matrix multiplication completed in {end_time - start_time:.4f} seconds")
