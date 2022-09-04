from numba import cuda, njit
import numpy as np
from math import exp, sqrt

class MyLogger:
    def __init__(self):
        pass

    def log(self, s):
        print("\n" + s, end = "")

    def log_add(self, s):
        print(s, end = "")

elemlabel = {1: 'H', 30: 'Zn', 63: 'Eu', 2: 'He', 31: 'Ga', 64: 'Gd', 3: 'Li', 32: 'Ge', 65: 'Tb', 4: 'Be', 33: 'As', 66: 'Dy', 5: 'B', 34: 'Se', 67: 'Ho', 6: 'C', 35: 'Br', 68: 'Er', 36: 'Kr', 69: 'Tm', 37: 'Rb', 70: 'Yb', 7: 'N', 38: 'Sr', 71: 'Lu', 8: 'O', 39: 'Y', 72: 'Hf', 9: 'F', 40: 'Zr', 73: 'Ta', 10: 'Ne', 41: 'Nb', 74: 'W', 11: 'Na', 42: 'Mo', 75: 'Re', 12: 'Mg', 43: 'Tc', 76: 'Os', 13: 'Al', 44: 'Ru', 77: 'Ir', 14: 'Si', 45: 'Rh', 78: 'Pt', 15: 'P', 46: 'Pd', 79: 'Au', 16: 'S', 47: 'Ag', 80: 'Hg', 17: 'Cl', 48: 'Cd', 81: 'Tl', 18: 'Ar', 49: 'In', 82: 'Pb', 19: 'K', 50: 'Sn', 83: 'Bi', 20: 'Ca', 51: 'Sb', 84: 'Po', 21: 'Sc', 52: 'Te', 85: 'At', 22: 'Ti', 53: 'I', 86: 'Rn', 23: 'V', 54: 'Xe', 87: 'Fr', 24: 'Cr', 55: 'Cs', 88: 'Ra', 25: 'Mn', 56: 'Ba', 89: 'Ac', 57: 'La', 90: 'Th', 26: 'Fe', 58: 'Ce', 91: 'Pa', 59: 'Pr', 92: 'U', 27: 'Co', 60: 'Nd', 93: 'Np', 61: 'Pm', 94: 'Pu', 28: 'Ni', 62: 'Sm', 95: 'Am', 29: 'Cu', 96: 'Cm'}
elemradius = {1: 0.50, 30: 1.22, 63: 1.98, 2: 0.28, 31: 1.22, 64: 1.96, 3: 1.28, 32: 1.2, 65: 1.94, 4: 0.96, 33: 1.19, 66: 1.92, 5: 0.84, 34: 1.2, 67: 1.92, 6: 0.76, 35: 1.2, 68: 1.89, 36: 1.16, 69: 1.9, 37: 2.2, 70: 1.87, 7: 0.71, 38: 1.95, 71: 1.87, 8: 0.66, 39: 1.9, 72: 1.75, 9: 0.57, 40: 1.75, 73: 1.7, 10: 0.58, 41: 1.64, 74: 1.62, 11: 1.66, 42: 1.54, 75: 1.51, 12: 1.41, 43: 1.47, 76: 1.44, 13: 1.21, 44: 1.46, 77: 1.41, 14: 1.11, 45: 1.42, 78: 1.36, 15: 1.07, 46: 1.39, 79: 1.36, 16: 1.05, 47: 1.45, 80: 1.32, 17: 1.02, 48: 1.44, 81: 1.45, 18: 1.06, 49: 1.42, 82: 1.46, 19: 2.03, 50: 1.39, 83: 1.48, 20: 1.76, 51: 1.39, 84: 1.4, 21: 1.7, 52: 1.38, 85: 1.5, 22: 1.6, 53: 1.39, 86: 1.5, 23: 1.53, 54: 1.4, 87: 2.6, 24: 1.39, 55: 2.44, 88: 2.21, 25: 1.39, 56: 2.15, 89: 2.15, 57: 2.07, 90: 2.06, 26: 1.32, 58: 2.04, 91: 2.0, 59: 2.03, 92: 1.96, 27: 1.26, 60: 2.01, 93: 1.9, 61: 1.99, 94: 1.87, 28: 1.24, 62: 1.98, 95: 1.8, 29: 1.32, 96: 1.69}
elemradius = {elemlabel[i]: elemradius[i] for i in elemlabel}

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


@cuda.jit
def orbital_value_kernel(
    V:cuda.cudadrv.devicearray.DeviceNDArray,               # The volume should be filled
    xmin, ymin, zmin, dx, dy, dz,                           # The parameters of cube
    normcoeffs, contracts, positions, powers, atomidxs,     # The parameters of every GTFs. it is an readonly device array
    coeffs                                                  # coefficient of GTFs of the current orbital
    ):
    # A faster version of calculating cube files by reducing the call of exp(). Has 1/3 time cost of version 1
    # basis contribution smaller than 1.0e-9 are neglected because of the accuracy of float32
    i, j, k = cuda.grid(3)
    x0, y0, z0 = xmin + dx * i, ymin + dy * j, zmin + dz * k
    if i < V.shape[0] and j < V.shape[1] and k < V.shape[2]:
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


def get_plane_values(p0, pv, plen, ngrid, V:np.ndarray, xmin, ymin, zmin, dx, dy, dz):
    xmax, ymax, zmax = xmin+dx*V.shape[0], ymin+dy*V.shape[1], zmin+dz*V.shape[2]
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
    p1s[:,0] = np.clip(p1s[:,0], xmin, xmax - 2*dx)
    p1s[:,1] = np.clip(p1s[:,1], ymin, ymax - 2*dy)
    p1s[:,2] = np.clip(p1s[:,2], zmin, zmax - 2*dz)
    p2s[:,0] = np.clip(p2s[:,0], xmin, xmax - 2*dx)
    p2s[:,1] = np.clip(p2s[:,1], ymin, ymax - 2*dy)
    p2s[:,2] = np.clip(p2s[:,2], zmin, zmax - 2*dz)
    p0 = np.array((np.clip(p0[0], xmin, xmax-2*dx), np.clip(p0[1], ymin, ymax-2*dy), np.clip(p0[2], zmin, zmax-2*dz)))
    mp = np.empty((p1s.shape[0], p2s.shape[0]), dtype=np.float32)
    get_plane_values_kernel(mp, p0, p1s, p2s, V, xmin, ymin, zmin, dx, dy, dz)
    return mp

@njit
def get_plane_values_kernel(mp, p0, p1s, p2s, V, xmin, ymin, zmin, dx, dy, dz):
    for i, v1 in enumerate(p1s):
        for j, v2 in enumerate(p2s):
            v = v1 + v2 + p0
            x, y, z = v
            iv, jv, kv = int((x-xmin)//dx), int((y-ymin)//dy), int((z-zmin)//dz)
            tx, ty, tz = x%dx/dx, y%dy/dy, z%dz/dz
            fx1 = (1-tx)*V[iv,jv,kv]+tx*V[iv+1,jv,kv]
            fx2 = (1-tx)*V[iv,jv+1,kv]+tx*V[iv+1,jv+1,kv]
            fx3 = (1-tx)*V[iv,jv,kv+1]+tx*V[iv+1,jv,kv+1]
            fx4 = (1-tx)*V[iv,jv+1,kv+1]+tx*V[iv+1,jv+1,kv+1]
            fy1 = (1-ty)*fx1+ty*fx2
            fy2 = (1-ty)*fx3+ty*fx4
            mp[i,j] = (1-tz)*fy1+tz*fy2

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    V = np.random.randint(3, 5, size = (50, 50, 50))
    V[:,:,:25] = -V[:,:,25:]
    xmin, ymin, zmin = -1, -1, -1
    dx, dy, dz = 0.04, 0.04, 0.04
    p0 = np.array([0,0,0])
    pv = np.array([-0.01, 0.0, 0.99])
    plen = 1
    ngrid = 100

    mp = get_plane_values(p0, pv, plen, ngrid, V, xmin, ymin, zmin, dx, dy, dz)
    plt.imshow(mp)
    plt.colorbar()
    plt.show()
    
