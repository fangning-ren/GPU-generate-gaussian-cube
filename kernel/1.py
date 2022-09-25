import cupy as cp
import numpy as np

squared_diff = cp.ElementwiseKernel(
   'float32 x, float32 y',
   'float32 z',
   'z = (x - y) * (x - y)',
   'squared_diff')

x = cp.arange(10, dtype=np.float32).reshape(2, 5)
y = 1.0
print(squared_diff(x, y))
print(squared_diff(x, 5))