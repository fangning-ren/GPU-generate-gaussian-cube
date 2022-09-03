import enum
from numba import cuda
import numpy as np
import copy
from read_files import Excitation, MoldenWavefunction
from read_files import GaussianCube
from utils import *

import time
import matplotlib.pyplot as plt


class DensityManager:
    """This class is for calculating cube files and matrixs """
    # 这个类负责从波函数生成cube
    def __init__(self, wf:MoldenWavefunction, cube:GaussianCube):
        self.cube = cube
        self.wf = wf
        self.C = self.wf.C
        self.D = None

        self.C_raveled = self.wf.get_raveled_C()
        self.gtfs = self.wf.get_raveled_gtf()
        self.cuda_block_dim = (32, 32, 32)

        self.C_raveled = cuda.to_device(self.C_raveled)
        self.gtf_N0 = cuda.to_device(self.gtfs[0])  # GTF norm coefficient
        self.gtf_aa = cuda.to_device(self.gtfs[1])  # GTF contract
        self.gtf_ps = cuda.to_device(self.gtfs[2])  # GTF center
        self.gtf_pw = cuda.to_device(self.gtfs[3])  # GTF power, angular momentum
        self.gtf_ai = cuda.to_device(self.gtfs[4])  # GTF atom index
        self.gtfs = (self.gtf_N0, self.gtf_aa, self.gtf_ps, self.gtf_pw, self.gtf_ai)
    
    def get_density_matrix(self):
        D_mo = np.diag(self.wf.occupys, k = 0)
        D = self.C @ D_mo @ self.C.T
        self.D = D
        return D

    def get_orbital_value(self, orbidx):
        """param overlap: the calculated data will overlap the original grid data without returning"""
        coeffs = np.ascontiguousarray(self.wf.C_raveled[:,orbidx]) # cuda kernel cannot accept non-continueous array slices
        coeffs = cuda.to_device(coeffs)
        dV = cuda.device_array(self.cube.shape, dtype=np.float32)
        threadsperblock = (8, 8, 8)
        blockspergrid_x = np.ceil(self.cube.shape[0] / threadsperblock[0]).astype(np.int32)
        blockspergrid_y = np.ceil(self.cube.shape[1] / threadsperblock[1]).astype(np.int32)
        blockspergrid_z = np.ceil(self.cube.shape[2] / threadsperblock[2]).astype(np.int32)
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        orbital_value_kernel[blockspergrid, threadsperblock](dV, *self.cube.minpos, *self.cube.grid_size, *self.gtfs, coeffs)
        return dV.copy_to_host()

    def get_electron_density(self):
        """returns the 3d array"""
        occus = self.wf.occupys
        nzeroidxs = np.nonzero(occus)[0].astype(np.int32)
        iket = cuda.to_device(nzeroidxs)
        ibra = cuda.to_device(nzeroidxs)
        coef = cuda.to_device(np.ascontiguousarray(occus[nzeroidxs]))
        dV = cuda.device_array(self.cube.shape, dtype=np.float32)

        threadsperblock = (8, 8, 8)
        blockspergrid_x = np.ceil(self.cube.shape[0] / threadsperblock[0]).astype(np.int32)
        blockspergrid_y = np.ceil(self.cube.shape[1] / threadsperblock[1]).astype(np.int32)
        blockspergrid_z = np.ceil(self.cube.shape[2] / threadsperblock[2]).astype(np.int32)
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        density_kernel[blockspergrid, threadsperblock](dV, *self.cube.minpos, *self.cube.grid_size, *self.gtfs, self.C_raveled, iket, ibra, coef)
        cuda.synchronize()
        return dV.copy_to_host()

    def get_transition_wf(self, ext:Excitation):
        """This function cannot execute as the excited wavefunction is not the eigenfunction of time independent sch. equation"""
        # 该函数无法执行。因为跃迁以后的波函数被表示为本征态线性组合，而本征态的线性组合不是不含时薛定谔的解。只有含时薛定谔才能描述此过程
        return self.wf

    def get_transition_density_matrix(self, ext:Excitation):
        """the transition density matrix can still be obtained, but the MO coefficients are no longer the eigenvector of the density matrix."""
        tdm = np.zeros_like(self.D)
        for id1, id2, c in zip(ext.orb1, ext.orb2, ext.cisc):
            tdm[id2,id1] += c
        return self.C @ tdm @ self.C.T

    def get_transition_density(self, ext:Excitation):
        iket = np.zeros(len(ext.cisc), dtype = np.int32)
        ibra = np.zeros(len(ext.cisc), dtype = np.int32)
        coef = np.zeros(len(ext.cisc), dtype = np.float32)
        for i, tp in enumerate(zip(ext.orb1, ext.orb2, ext.cisc)):
            id1, id2, c = tp
            iket[i] = id2
            ibra[i] = id1
            coef[i] = c
        dV = cuda.device_array(self.cube.shape, dtype=np.float32)
        iket = cuda.to_device(iket)
        ibra = cuda.to_device(ibra)
        coef = cuda.to_device(coef)

        threadsperblock = (8, 8, 8)
        blockspergrid_x = np.ceil(self.cube.shape[0] / threadsperblock[0]).astype(np.int32)
        blockspergrid_y = np.ceil(self.cube.shape[1] / threadsperblock[1]).astype(np.int32)
        blockspergrid_z = np.ceil(self.cube.shape[2] / threadsperblock[2]).astype(np.int32)
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        density_kernel[blockspergrid, threadsperblock](dV, *self.cube.minpos, *self.cube.grid_size, *self.gtfs, self.C_raveled, iket, ibra, coef)
        cuda.synchronize()
        return dV.copy_to_host()

    def get_transition_diff_density_matrix(self, ext:Excitation):
        """the transition density matrix can still be obtained, but the MO coefficients are no longer the eigenvector of the density matrix."""
        D = np.diag(self.wf.occupys, k = 0)
        tdm = np.copy(D) / 2
        for i, a, wia in zip(ext.orb1, ext.orb2, ext.cisc):
            for j, b, wjb in zip(ext.orb1, ext.orb2, ext.cisc):
                if i == j and a != b:
                    tdm[a,b] += wia*wjb
                if a == b and i != j:
                    tdm[j,i] += wia*wjb
        for i, a, wia in zip(ext.orb1, ext.orb2, ext.cisc):
            tdm[i,i] -= wia*wia
            tdm[a,a] += wia*wia
        tdm -= D
        return self.C @ tdm @ self.C.T

    def get_transition_diff_density(self, ext:Excitation):
        sparse_tdm = {}
        for i, a, wia in zip(ext.orb1, ext.orb2, ext.cisc):
            for j, b, wjb in zip(ext.orb1, ext.orb2, ext.cisc):
                if i == j and a != b:
                    k, v = (a,b), wia*wjb
                elif a == b and i != j:
                    k, v = (j,i), wia*wjb
                else:
                    continue
                if k not in sparse_tdm:
                    sparse_tdm[k] = v
                else:
                    sparse_tdm[k] += v

        for i, a, wia in zip(ext.orb1, ext.orb2, ext.cisc):
            if (i,i) not in sparse_tdm:
                sparse_tdm[(i,i)] = -wia*wia
            else:
                sparse_tdm[(i,i)] -= wia*wia
            if (a,a) not in sparse_tdm:
                sparse_tdm[(a,a)] = +wia*wia
            else:
                sparse_tdm[(a,a)] += wia*wia

        tdm_pos = list(sparse_tdm.keys())
        tdm_val = [sparse_tdm[k] for k in tdm_pos]
        iket = np.zeros(len(tdm_val), dtype = np.int32)
        ibra = np.zeros(len(tdm_val), dtype = np.int32)
        coef = np.zeros(len(tdm_val), dtype = np.float32)
        for i, tp in enumerate(zip(tdm_pos, tdm_val)):
            p, c = tp
            id1, id2 = p
            iket[i] = id2
            ibra[i] = id1
            coef[i] = c
        dV = cuda.device_array(self.cube.shape, dtype=np.float32)
        iket = cuda.to_device(iket)
        ibra = cuda.to_device(ibra)
        coef = cuda.to_device(coef)

        threadsperblock = (8, 8, 8)
        blockspergrid_x = np.ceil(self.cube.shape[0] / threadsperblock[0]).astype(np.int32)
        blockspergrid_y = np.ceil(self.cube.shape[1] / threadsperblock[1]).astype(np.int32)
        blockspergrid_z = np.ceil(self.cube.shape[2] / threadsperblock[2]).astype(np.int32)
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        density_kernel[blockspergrid, threadsperblock](dV, *self.cube.minpos, *self.cube.grid_size, *self.gtfs, self.C_raveled, iket, ibra, coef)
        cuda.synchronize()
        return dV.copy_to_host()

