import numpy as np
import cupy as cp
import cupyx
import time

from numba import njit, cuda

@njit
def add_cpu(C, a):
    for aa in a:
        C += aa
a = np.random.uniform(0.0, 1.0, size  = (100)).astype(np.float32)
add_cpu(np.zeros((10, 10, 10), dtype = np.float32), a)

@cuda.jit
def add_gpu(C, a):
    i, j, k = cuda.grid(3)
    for n in range(len(a)):
        C[i,j,k] *= n
threadsperblock = (8, 8, 8)
blockspergrid_x = np.ceil(10 / threadsperblock[0]).astype(np.int32)
blockspergrid_y = np.ceil(10 / threadsperblock[1]).astype(np.int32)
blockspergrid_z = np.ceil(10 / threadsperblock[2]).astype(np.int32)
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
a = cuda.to_device(np.random.uniform(0.0, 1.0, size  = (100)).astype(np.float32))
t = time.time()
add_gpu[blockspergrid, threadsperblock](cuda.to_device(np.zeros((10, 10, 10), dtype = np.float32)), a)


# cube_cpu = np.zeros((512, 512, 512), dtype = np.float32)
# a = np.random.uniform(0.0, 1.0, size  = (500)).astype(np.float32)
# t = time.time()
# add_cpu(cube_cpu, a)
# print("cpu:  " + str(time.time()-t))

cube_gpu = cuda.to_device(np.zeros((512, 512, 512), dtype = np.float32))
threadsperblock = (8, 8, 8)
blockspergrid_x = np.ceil(cube_gpu.shape[0] / threadsperblock[0]).astype(np.int32)
blockspergrid_y = np.ceil(cube_gpu.shape[1] / threadsperblock[1]).astype(np.int32)
blockspergrid_z = np.ceil(cube_gpu.shape[2] / threadsperblock[2]).astype(np.int32)
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
a = cuda.to_device(np.random.uniform(0.0, 1.0, size  = (500)).astype(np.float32))
t = time.time()
add_gpu[blockspergrid, threadsperblock](cube_gpu, a)
cuda.synchronize()
print("gpu:  " + str(time.time()-t))
