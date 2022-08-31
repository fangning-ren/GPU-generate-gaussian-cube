from numba import cuda
import numpy as np
from read_files import MoldenWavefunction
from read_files import GaussianCube
from utils import *



class DensityManager:
    """This class is for calculating differnet densities. just for calculating densities """
    # 这个类负责从波函数生成cube
    def __init__(self, wf:MoldenWavefunction, cube:GaussianCube):
        self.cube = cube
        self.wf = wf
        self.C = self.wf.C
        self.D = None

        self.C_raveled = self.wf.get_raveled_C()
        self.gtfs = self.wf.get_raveled_gtf()
        self.cuda_block_dim = (32, 32, 32)
    
    def get_density_matrix(self):
        D_mo = np.diag(self.wf.occupys, k = 0)
        D = self.C.T @ D_mo @ self.C
        self.D = D
        return D

    def get_orbital_value(self, orbidx):
        """param overlap: the calculated data will overlap the original grid data without returning"""
        coeffs = np.ascontiguousarray(self.wf.C_raveled[:,orbidx]) # cuda kernel cannot accept non-continueous array slices
        
        dV = cuda.device_array(self.cube.shape, dtype=np.float32)
        griddim = ((self.cube.nx+31)//self.cuda_block_dim[0], (self.cube.ny+31)//self.cuda_block_dim[1], (self.cube.nz+31)//self.cuda_block_dim[2])
        orbital_value_kernel_v2[self.cuda_block_dim, griddim](dV, *self.cube.minpos, *self.cube.grid_size, *self.gtfs, coeffs)
        return dV.copy_to_host()
        # dV = np.empty(self.cube.shape, dtype=np.float32)
        # orbital_value_kernel_cpu(dV, *self.cube.minpos, *self.cube.grid_size, *self.gtfs, coeffs)
        # return dV

    
    def get_electron_density(self):
        """returns the 3d array"""
        occus = self.wf.occupys
        nzeroidxs = np.nonzero(occus)[0].astype(np.int32)
        C_rav = np.ascontiguousarray(self.C_raveled[:,nzeroidxs]).reshape(self.C_raveled.shape[0], nzeroidxs.shape[0])
        occus = np.ascontiguousarray(occus[nzeroidxs])

        dV = cuda.device_array(self.cube.shape, dtype=np.float32)
        griddim = ((self.cube.nx+31)//self.cuda_block_dim[0], (self.cube.ny+31)//self.cuda_block_dim[1], (self.cube.nz+31)//self.cuda_block_dim[2])
        density_kernel[self.cuda_block_dim, griddim](dV, *self.cube.minpos, *self.cube.grid_size, *self.gtfs, C_rav, occus)
        return dV.copy_to_host()


