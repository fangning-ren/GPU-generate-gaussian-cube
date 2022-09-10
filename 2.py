
from numba import cuda, float32
import numpy as np

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

@cuda.jit(debug = True, opt = False)
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

import math
import time


t = time.time()
A = np.random.random((2000, 2000)).astype(np.float32)
B = np.random.random((2000, 2000)).astype(np.float32)
t = time.time()
A @ B
print(time.time()-t)
A = cuda.to_device(np.random.random((2048, 2048)).astype(np.float32))
B = cuda.to_device(np.random.random((2048, 2048)).astype(np.float32))
C = cuda.to_device(np.empty_like(A))

tpb = (TPB,TPB)
bpgx = math.ceil(A.shape[0] / tpb[0])
bpgy = math.ceil(A.shape[1] / tpb[1])

fast_matmul[(bpgx, bpgy), tpb](A,B,C)
t = time.time()
fast_matmul[(bpgx, bpgy), tpb](A,B,C)
cuda.synchronize()
print(time.time()-t)
