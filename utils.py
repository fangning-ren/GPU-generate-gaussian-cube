from numba import cuda, njit, float32, int32
import numpy as np
from math import exp, sqrt


DEFAULT_CUBE_SHAPE = (128, 128, 128)
CUDA_BLOCK_SHAPE = (8, 8, 8)
CUDA_THREAD_PER_BLOCK = 512
TPB = 512
SPB = 64



class MyLogger:
    def __init__(self):
        pass

    def log(self, s):
        print("\n" + s, end = "")

    def log_add(self, s):
        print(s, end = "")

@njit
def rot(x, y, z, ax, ay, az):
    x = np.cos(az)*x - np.sin(az)*y
    y = np.sin(az)*x + np.cos(az)*y

    x = np.cos(ay)*x + np.sin(ay)*z
    z =-np.sin(ay)*x + np.cos(ay)*z

    y = np.cos(ax)*y - np.sin(ax)*z
    z = np.sin(ax)*y + np.cos(ax)*z
    return x, y, z

@njit
def orbital_value_kernel_cpu(V:np.ndarray, xmin, ymin, zmin, dx, dy, dz, normcoeffs, contracts, positions, powers, atomidxs, coeffs):
    # A faster version of calculating cube files by reducing the call of exp(). Has 1/3 time cost of version 1
    # basis contribution smaller than 1.0e-9 are neglected because of the accuracy of float32
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            for k in range(V.shape[2]):
                V[i,j,k] = 0
                for n in range(len(normcoeffs)):
                    x = xmin + dx * i - positions[n,0]
                    y = ymin + dy * j - positions[n,1]
                    z = zmin + dz * k - positions[n,2]
                    V[i,j,k] += coeffs[n] * normcoeffs[n] * x**powers[n,0] * y**powers[n,1] * z**powers[n,2] * exp(-contracts[n] * (x*x+y*y+z*z))

@cuda.jit
def orbital_value_kernel_v1(V:np.ndarray, xmin, ymin, zmin, dx, dy, dz, normcoeffs, contracts, positions, powers, atomidxs, coeffs):
    # the most traditional way to calculate cube files.
    i, j, k = cuda.grid(3)
    if i < V.shape[0] and j < V.shape[1] and k < V.shape[2]:
        V[i,j,k] = 0
        for n in range(len(normcoeffs)):
            x = xmin + dx * i - positions[n,0]
            y = ymin + dy * j - positions[n,1]
            z = zmin + dz * k - positions[n,2]
            V[i,j,k] += coeffs[n] * normcoeffs[n] * x**powers[n,0] * y**powers[n,1] * z**powers[n,2] * exp(-contracts[n] * (x*x+y*y+z*z))


@cuda.jit(opt = True, max_registers = 64)
def orbital_value_kernel_v2(
    V:cuda.cudadrv.devicearray.DeviceNDArray,               # The volume should be filled
    xmin, ymin, zmin, dx, dy, dz,                           # The parameters of cube
    normcoeffs, contracts, positions, powers, atomidxs,     # The parameters of every GTFs. it is an readonly device array
    coeffs                                                  # coefficient of GTFs of the current orbital
    ):
    # 目前最快版本。
    # A faster version of calculating cube files by reducing the call of exp(). Has 1/3 time cost of version 1
    # basis contribution smaller than 1.0e-9 are neglected because of the accuracy of float32
    i, j, k = cuda.grid(3)
    x0, y0, z0 = xmin + dx * i, ymin + dy * j, zmin + dz * k
    if i < V.shape[0] and j < V.shape[1] and k < V.shape[2]:
        Vijk = float32(0.0)
        aid = -1
        for n in range(len(normcoeffs)):    
            if aid != atomidxs[n]:      # This if never generate branches because the atomidx for every threads is the same!
                aid = atomidxs[n]
                x = x0 - positions[n,0]
                y = y0 - positions[n,1]
                z = z0 - positions[n,2]
                r2 = x*x+y*y+z*z
            ar2 = contracts[n] * r2
            if ar2 >= float32(25.0):         # This may branch. But it depends on the spacial position of the grid point of this thread so mostly it will not branch
                continue
            Vijk += coeffs[n] * normcoeffs[n] * x**powers[n,0] * y**powers[n,1] * z**powers[n,2] * exp(-ar2)    # The most time comsuming part
        V[i,j,k] = Vijk

@cuda.jit(opt = True, max_registers = 64)
def orbital_value_kernel_v3(
    V:cuda.cudadrv.devicearray.DeviceNDArray,               # The volume should be filled
    xmin, ymin, zmin, dx, dy, dz,                           # The parameters of cube
    normcoeffs, contracts, positions, powers, atomidxs,     # The parameters of every GTFs. it is an readonly device array
    coeffs                                                  # coefficient of GTFs of the current orbital
    ):
    # 不知道为什么，使用了共享内存以后，速度反而变慢了。opt选项对执行速度有显著影响。加了opt以后v2和v3不再有显著差异。怀疑numba编译器自动创建了shared memory来加速这一过程
    # 这个kernel算出来的结果有问题，某些位置的数据不对。不要使用！
    # A blocked version of calculating cube files
    # previous version uses n^3*ngtf*9 reading and n^3 writing
    # move some data into the shared memory before calculating to reduce read and write time
    # 首先，创建一个block内部共享内存，然后把他妈的gtf参数，并行地，扔进去......
    # thread per block 可能小于gtf数量。如果遇到此情形，则每个thread必须赋值两次。
    # 然后，从这个共享内存里存着的GTF信息里面读取各项参数计算某个节点的值
    # ngtf = normcoeffs.shape[0]
    i, j, k = cuda.grid(3)
    x0, y0, z0 = xmin + dx * i, ymin + dy * j, zmin + dz * k
    if i < V.shape[0] and j < V.shape[1] and k < V.shape[2]:    # set the original data to zero to enable accumulation. Must do boundary check
        V[i,j,k] = 0
    s_normcoeffs    = cuda.shared.array(shape = (SPB,), dtype=float32)
    s_contracts     = cuda.shared.array(shape = (SPB,), dtype = float32)
    s_positions     = cuda.shared.array(shape = (SPB, 3), dtype = float32)
    s_powers        = cuda.shared.array(shape = (SPB, 3), dtype = int32)
    s_atomidxs      = cuda.shared.array(shape = (SPB,), dtype = int32)
    s_coeffs        = cuda.shared.array(shape = (SPB,), dtype = float32)
    ngtf = coeffs.shape[0]
    igtf = cuda.threadIdx.x * cuda.blockDim.y * cuda.blockDim.z + cuda.threadIdx.y * cuda.blockDim.z + cuda.threadIdx.z
    for ispb in range((ngtf + SPB - 1) // SPB):    # the assign of shared array is independent with the assign of volume datai
        tmp = igtf + ispb * SPB
        if igtf < SPB and tmp < ngtf:                          # each thread will assign one value into the shared array. 为节省空间shared数组会重复赋值
            s_normcoeffs[igtf] = normcoeffs[tmp]
            s_contracts[igtf] = contracts[tmp]
            s_positions[igtf,0] = positions[tmp,0]
            s_positions[igtf,1] = positions[tmp,1]
            s_positions[igtf,2] = positions[tmp,2]
            s_powers[igtf,0] = powers[tmp,0]
            s_powers[igtf,1] = powers[tmp,1]
            s_powers[igtf,2] = powers[tmp,2]
            s_atomidxs[igtf] = atomidxs[tmp]
            s_coeffs[igtf] = coeffs[tmp]
        cuda.syncthreads()
    
        if i < V.shape[0] and j < V.shape[1] and k < V.shape[2]:    # another boundary check
            Vijk = float32(0.0) #reduce the number of global memory writting 
            aid = -1
            for n in range(0, min(SPB, ngtf-tmp)):    
                if aid != s_atomidxs[n]:      # This if never generate branches because the atomidx for every threads is the same!
                    aid = s_atomidxs[n]
                    x = x0 - s_positions[n,0]
                    y = y0 - s_positions[n,1]
                    z = z0 - s_positions[n,2]
                    r2 = x*x+y*y+z*z
                ar2 = s_contracts[n] * r2
                if ar2 >= float32(25.0):         # This may branch. But it depends on the spacial position of the grid point of this thread so mostly it will not branch
                    continue
                Vijk += s_coeffs[n] * s_normcoeffs[n] * x**s_powers[n,0] * y**s_powers[n,1] * z**s_powers[n,2] * exp(-ar2)    # The most time comsuming part
            V[i,j,k] += Vijk
        cuda.syncthreads()
               
@njit
def orbital_value_kernel_cpu(
    V,               # The volume should be filled
    xmin, ymin, zmin, dx, dy, dz,                           # The parameters of cube
    normcoeffs, contracts, positions, powers, atomidxs,     # The parameters of every GTFs. it is an readonly device array
    coeffs                                                  # coefficient of GTFs of the current orbital
    ):
    # Very slow, do not use
    # Slower than multiwfn
    # Why multiwfn is so fast???
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            for k in range(V.shape[2]):
                x0, y0, z0 = xmin + dx * i, ymin + dy * j, zmin + dz * k
                V[i,j,k] = 0
                aid = -1
                for n in range(len(normcoeffs)):    
                    if aid != atomidxs[n]:      # This if never generate branches because the atomidx for every threads is the same!
                        aid = atomidxs[n]
                        x = x0 - positions[n,0]
                        y = y0 - positions[n,1]
                        z = z0 - positions[n,2]
                        r2 = x*x+y*y+z*z
                    ar2 = contracts[n] * r2
                    if ar2 >= 25.0:         # This may branch. But it depends on the spacial position of the grid point of this thread so mostly it will not branch
                        continue
                    V[i,j,k] += coeffs[n] * normcoeffs[n] * x**powers[n,0] * y**powers[n,1] * z**powers[n,2] * exp(-ar2)    # The most time comsuming part


@cuda.jit
def density_kernel(
    V:cuda.cudadrv.devicearray.DeviceNDArray,               # The volume should be filled
    xmin, ymin, zmin, dx, dy, dz,                           # The parameters of cube
    normcoeffs, contracts, positions, powers, atomidxs,     # The parameters of every GTFs. it is an readonly device array
    C_raveled,                                              # coefficient matrix of each GTF. also readonly device array
    iket, ibra, coeff                                       # sparse matrix representation of the density matrix on the mo representation. the row, col index and the corresponding non-zero value
    ):
    # the parameter was set in this form because every density, like transition density, pure density, electrom-hole map, their matrix on the MO basis are very sparse. 
    # the density at point r always has the form p(r) = sum_uv(fu(r) * D_ao[u,v] * fv(r)) = sum_ij(phi_i(r) * D_mo[i,j] * phi_j(r))
    i, j, k = cuda.grid(3)
    x0, y0, z0 = xmin + dx * i, ymin + dy * j, zmin + dz * k
    if i < V.shape[0] and j < V.shape[1] and k < V.shape[2]:
        for m in range(len(coeff)):
            vket, vbra = 0.0, 0.0
            aid = -1
            for n in range(len(normcoeffs)):
                if aid != atomidxs[n]:
                    aid = atomidxs[n]
                    x = x0 - positions[n,0]
                    y = y0 - positions[n,1]
                    z = z0 - positions[n,2]
                    r2 = x*x+y*y+z*z
                ar2 = contracts[n] * r2
                if ar2 >= 25.0:
                    continue
                v0 = normcoeffs[n] * x**powers[n,0] * y**powers[n,1] * z**powers[n,2] * exp(-ar2)
                vket += C_raveled[n,iket[m]] * v0
                vbra += C_raveled[n,ibra[m]] * v0
            V[i,j,k] += vbra * vket * coeff[m]

@cuda.jit
def density_kernel_v2(
    V:cuda.cudadrv.devicearray.DeviceNDArray,               # The volume should be filled
    xmin, ymin, zmin, dx, dy, dz,                           # The parameters of cube
    normcoeffs, contracts, positions, powers, atomidxs,     # The parameters of every GTFs. it is an readonly device array
    C_raveled,                                              # coefficient matrix of each GTF. also readonly device array
    iket, ibra, coeff                                       # sparse matrix representation of the density matrix on the mo representation. the row, col index and the corresponding non-zero value
    ):
    # the parameter was set in this form because every density, like transition density, pure density, electrom-hole map, their matrix on the MO basis are very sparse. 
    # the density at point r always has the form p(r) = sum_uv(fu(r) * D_ao[u,v] * fv(r)) = sum_ij(phi_i(r) * D_mo[i,j] * phi_j(r))
    i, j, k = cuda.grid(3)
    x0, y0, z0 = xmin + dx * i, ymin + dy * j, zmin + dz * k
    if i < V.shape[0] and j < V.shape[1] and k < V.shape[2]:
        for m in range(len(coeff)):
            vket, vbra = 0.0, 0.0
            aid = -1
            for n in range(len(normcoeffs)):
                if aid != atomidxs[n]:
                    aid = atomidxs[n]
                    x = x0 - positions[n,0]
                    y = y0 - positions[n,1]
                    z = z0 - positions[n,2]
                    r2 = x*x+y*y+z*z
                ar2 = contracts[n] * r2
                if ar2 >= 25.0:
                    continue
                v0 = normcoeffs[n] * x**powers[n,0] * y**powers[n,1] * z**powers[n,2] * exp(-ar2)
                vket += C_raveled[n,iket[m]] * v0
                vbra += C_raveled[n,ibra[m]] * v0
            V[i,j,k] += vbra * vket * coeff[m]


def get_plane_values(p0, pv, plen, ngrid, V:np.ndarray, xmin, ymin, zmin, dx, dy, dz):
    pv = pv / np.linalg.norm(pv)
    a, b, c = pv
    if a == b == 0:
        vx = np.array([1,0,0])
        vy = np.array([0,1,0])
    else:
        vx = np.array([b/sqrt(a*a+b*b),-a/sqrt(a*a+b*b), 0])
        vy = np.cross(vx, pv)

    p00 = np.linspace(-plen/2, plen/2, ngrid)
    p1s = np.stack((p00,p00,p00), axis = 1) * vx.reshape((1, 3))
    p2s = np.stack((p00,p00,p00), axis = 1) * vy.reshape((1, 3))
    p0 = np.array(p0)
    mp = np.empty((p1s.shape[0], p2s.shape[0]), dtype=np.float32)
    get_plane_values_kernel(mp, p0, p1s, p2s, V, xmin, ymin, zmin, dx, dy, dz)
    return mp

@njit
def get_plane_values_kernel(mp:np.ndarray, p0:np.ndarray, p1s:np.ndarray, p2s:np.ndarray, V:np.ndarray, xmin, ymin, zmin, dx, dy, dz):
    nx, ny, nz = V.shape
    for i, v1 in enumerate(p1s):
        for j, v2 in enumerate(p2s):
            v = v1 + v2 + p0
            xv, yv, zv = v
            nxv, nyv, nzv = np.int64(np.round((xv - xmin) // dx)), np.int64(np.round((yv - ymin) // dy)), np.int64(np.round((zv - zmin) // dz))
            nx0, nx1, tx  = min(nx - 1, max(0, nxv)), min(max(0, nxv + 1), nx - 1), min(1, max(0, ((xv - xmin) / dx - nxv)))
            ny0, ny1, ty  = min(ny - 1, max(0, nyv)), min(max(0, nyv + 1), ny - 1), min(1, max(0, ((yv - ymin) / dy - nyv)))
            nz0, nz1, tz  = min(nz - 1, max(0, nzv)), min(max(0, nzv + 1), nz - 1), min(1, max(0, ((zv - zmin) / dz - nzv)))
            v000, v001, v010, v011 = V[nx0,ny0,nz0], V[nx0,ny0,nz1], V[nx0,ny1,nz0], V[nx0,ny1,nz1]
            v100, v101, v110, v111 = V[nx1,ny0,nz0], V[nx1,ny0,nz1], V[nx1,ny1,nz0], V[nx1,ny1,nz1]
            fz00 = v000 * (1 - tz) + v001 * tz
            fz01 = v010 * (1 - tz) + v011 * tz
            fz10 = v100 * (1 - tz) + v101 * tz
            fz11 = v110 * (1 - tz) + v111 * tz
            fy0 = fz00 * (1 - ty) + fz01 * ty
            fy1 = fz10 * (1 - ty) + fz11 * ty
            mp[i,j] = fy0 * (1 - tx) + fy1 * tx



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    V = np.random.randint(-3, 5, size = (8,8,8))
    # V[:,:,:25] = -V[:,:,25:]
    xmin, ymin, zmin = -2, -2, -2
    dx, dy, dz = 0.9, 0.9, 0.9
    p0 = np.array([0,0,0])
    pv = np.array([0.0, 0, 1])
    plen = 8
    ngrid = 100

    mp = get_plane_values(p0, pv, plen, ngrid, V, xmin, ymin, zmin, dx, dy, dz)
    plt.imshow(mp)
    plt.colorbar()
    plt.show()
    
