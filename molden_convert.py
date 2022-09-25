import numpy as np
import sys
import re
from math import sqrt, factorial, pi

class MyLogger:
    def __init__(self):
        pass

    def log(self, s):
        print("\n" + s, end = "")

    def log_add(self, s):
        print(s, end = "")

S_convert = np.matrix([
    [1.00000000,],
], dtype = np.float32)

P_convert = np.matrix([
    [1.00000000, 0.00000000, 0.00000000],
    [0.00000000, 1.00000000, 0.00000000],
    [0.00000000, 0.00000000, 1.00000000],
], dtype = np.float32)

D_convert = np.matrix([
    # D 0, D+1, D-1, D+2, D-2, S
    [-0.50000000, 0.00000000, 0.00000000, 0.86602540, 0.00000000], #xx
    [-0.50000000, 0.00000000, 0.00000000,-0.86602540, 0.00000000], #yy
    [ 1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000], #zz
    [ 0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000], #xy
    [ 0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000], #xz
    [ 0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000], #yz
], dtype = np.float32).T

# This is the conversion matrix for TeraChem to ORCA. Orca's F+3 and G+4 orbitals should be multiplied by -1 compared with the table provided by Multiwfn.
# For other programs it is possible to multiply the last two columns of the conversion matrix by -1
# Don't ask me why i know this
F_convert = np.matrix([
    # F+0, F+1, F-1, F+2, F-2, F+3, F-3, px, py, pz
    [ 0.00000000,-0.61237244, 0.00000000, 0.00000000, 0.00000000,-0.79056942, 0.00000000], #xxx
    [ 0.00000000, 0.00000000,-0.61237244, 0.00000000, 0.00000000, 0.00000000, 0.79056942], #yyy
    [ 1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000], #zzz
    [ 0.00000000,-0.27386127, 0.00000000, 0.00000000, 0.00000000, 1.06066017, 0.00000000], #xyy
    [ 0.00000000, 0.00000000,-0.27386127, 0.00000000, 0.00000000, 0.00000000,-1.06066017], #xxy
    [-0.67082039, 0.00000000, 0.00000000, 0.86602540, 0.00000000, 0.00000000, 0.00000000], #xxz
    [ 0.00000000, 1.09544511, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000], #xzz
    [ 0.00000000, 0.00000000, 1.09544511, 0.00000000, 0.00000000, 0.00000000, 0.00000000], #yzz
    [-0.67082039, 0.00000000, 0.00000000,-0.86602540, 0.00000000, 0.00000000, 0.00000000], #yyz
    [ 0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000], #xyz
], dtype = np.float32).T

G_convert = np.matrix([
    # G+0,G+1,G-1G+2,G-2,G+3,G-3,G+4,G-4, D+0, D+1, D-1, D+2, D-2, S
    [ 1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000], #zzzz
    [ 0.00000000, 0.00000000, 1.19522860, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000], #yzzz
    [-0.87831006, 0.00000000, 0.00000000,-0.98198050, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000], #yyzz
    [ 0.00000000, 0.00000000,-0.89642145, 0.00000000, 0.00000000, 0.00000000,-0.79056941, 0.00000000, 0.00000000], #yyyz
    [ 0.37500000, 0.00000000, 0.00000000, 0.55901699, 0.00000000, 0.00000000, 0.00000000,-0.73950997, 0.00000000], #yyyy
    [ 0.00000000, 1.19522860, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000], #xzzz
    [ 0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.13389341, 0.00000000, 0.00000000, 0.00000000, 0.00000000], #xyzz
    [ 0.00000000,-0.40089186, 0.00000000, 0.00000000, 0.00000000,-1.06066017, 0.00000000, 0.00000000, 0.00000000], #xyyz
    [ 0.00000000, 0.00000000, 0.00000000, 0.00000000,-0.42257712, 0.00000000, 0.00000000, 0.00000000, 1.11803398], #xyyy
    [-0.87831006, 0.00000000, 0.00000000, 0.98198050, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000], #xxzz
    [ 0.00000000, 0.00000000,-0.40089186, 0.00000000, 0.00000000, 0.00000000, 1.06066017, 0.00000000, 0.00000000], #xxyz
    [ 0.21957751, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.29903810, 0.00000000], #xxyy
    [ 0.00000000,-0.89642145, 0.00000000, 0.00000000, 0.00000000, 0.79056941, 0.00000000, 0.00000000, 0.00000000], #xxxz
    [ 0.00000000, 0.00000000, 0.00000000, 0.00000000,-0.42257712, 0.00000000, 0.00000000, 0.00000000,-1.11803398], #xxxy
    [ 0.37500000, 0.00000000, 0.00000000,-0.55901699, 0.00000000, 0.00000000, 0.00000000, 0.73950997, 0.00000000], #xxxx        
], dtype = np.float32).T

H_convert = np.eye(11, 21, dtype = np.float32)

def build_reverse_matrix(M):
    D = np.empty_like(M, dtype = np.float32)
    D[:,:] = M
    n_car, n_sph = D.shape[1], D.shape[0]
    D = np.concatenate((D, np.random.random((n_car-n_sph, n_car)).astype(np.float32)), axis = 0)

    for i in range(0, n_car - n_sph):
        b = np.zeros(n_car, dtype = np.float32)
        b[n_sph+i] = 1
        v = np.linalg.solve(D, b)
        v /= np.linalg.norm(v)
        D[n_sph+i] = v

    return D.I[:,:n_sph]

D_inverse = build_reverse_matrix(D_convert)
F_inverse = build_reverse_matrix(F_convert)
G_inverse = build_reverse_matrix(G_convert)
H_inverse = np.eye(21, 11, dtype = np.float32)

class GTF:
    def __init__(self, p, c, a, i, j, k):
        self.c, self.a, self.p = c, a, p
        self.i, self.j, self.k = i, j, k
        self.__calculate_normalize_coeff()

    def __call__(self, x, y, z):
        dx, dy, dz = x-self.p[0], y-self.p[1], z-self.p[2]
        return self.c * (dx**self.i * dy**self.j * dz**self.k) * np.exp(-self.a * (dx**2 + dy**2 + dz**2))

    def __calculate_normalize_coeff(self):
        L = self.i + self.j + self.k
        i = L // 3
        j = i + (L % 3) // 2
        k = L - i - j
        N0 = (2*self.a/pi)**0.75 * sqrt((8*self.a)**L*factorial(i)*factorial(j)*factorial(k)/(factorial(2*i)*factorial(2*j)*factorial(2*k)))
        N1 = (2*self.a/pi)**0.75 * sqrt((8*self.a)**L*factorial(self.i)*factorial(self.j)*factorial(self.k)/(factorial(2*self.i)*factorial(2*self.j)*factorial(2*self.k)))
        self.c = self.c * N1 / N0


class GTO:
    def __init__(self, position, contracts, coefficients, px, py, pz, atomidx):
        self.c = coefficients
        self.a = contracts
        self.p = position
        self.px = px
        self.py = py
        self.pz = pz
        self.atomidx = atomidx
        self.funcs = [GTF(self.p, c, a, px, py, pz) for c, a in zip(self.c, self.a)]

    def __call__(self, x, y, z):
        a = 0
        for f in self.funcs:
            a += f(x, y, z)
        return a


class GTOShell:
    def __init__(self, orbital_type = "s", contracts = [1.0,], coefficients = [1.0,], position = [0.0, 0.0, 0.0], atomidx = 0, gtotype = "spherical"):
        self.s = orbital_type
        self.c = np.array(coefficients, dtype = np.float32)
        self.a = np.array(contracts, dtype = np.float32)
        self.p = np.array(position, dtype = np.float32)
        self.atomidx = atomidx
        self.type = gtotype
        self.gtos = []
        self.generate_orbitals()

    def generate_orbitals(self):
        if self.s == "s":
            self.gtos.append(GTO(self.p, self.a, self.c, 0, 0, 0, self.atomidx))
        elif self.s == "p":
            self.gtos.append(GTO(self.p, self.a, self.c, 1, 0, 0, self.atomidx))
            self.gtos.append(GTO(self.p, self.a, self.c, 0, 1, 0, self.atomidx))
            self.gtos.append(GTO(self.p, self.a, self.c, 0, 0, 1, self.atomidx))
        elif self.s == "d":
            self.gtos.append(GTO(self.p, self.a, self.c, 2, 0, 0, self.atomidx))
            self.gtos.append(GTO(self.p, self.a, self.c, 0, 2, 0, self.atomidx))
            self.gtos.append(GTO(self.p, self.a, self.c, 0, 0, 2, self.atomidx))
            self.gtos.append(GTO(self.p, self.a, self.c, 1, 1, 0, self.atomidx))
            self.gtos.append(GTO(self.p, self.a, self.c, 1, 0, 1, self.atomidx))
            self.gtos.append(GTO(self.p, self.a, self.c, 0, 1, 1, self.atomidx))
        elif self.s == "f":
            idx = [3,0,0,1,2,2,1,0,0,1]
            idy = [0,3,0,2,1,0,0,1,2,1]
            idz = [0,0,3,0,0,1,2,2,1,1]
            for x, y, z in zip(idx, idy, idz):
                self.gtos.append(GTO(self.p, self.a, self.c, x, y, z, self.atomidx))
        elif self.s == "g":
            idx = [0,0,0,0,0,1,1,1,1,2,2,2,3,3,4]
            idy = [0,1,2,3,4,0,1,2,3,0,1,2,0,1,0]
            idz = [4,3,2,1,0,3,2,1,0,2,1,0,1,0,0]
            for x, y, z in zip(idx, idy, idz):
                self.gtos.append(GTO(self.p, self.a, self.c, x, y, z, self.atomidx))
        elif self.s == "h":
            idx = [0,0,0,0,0,0,1,1,1,1,1,2,2,2,2,3,3,3,4,4,5]
            idy = [0,1,2,3,4,5,0,1,2,3,4,0,1,2,3,0,1,2,0,1,0]
            idz = [5,4,3,2,1,0,4,3,2,1,0,3,2,1,0,2,1,0,1,0,0]
            for x, y, z in zip(idx, idy, idz):
                self.gtos.append(GTO(self.p, self.a, self.c, x, y, z, self.atomidx))


elemlabel = {1: 'H', 30: 'Zn', 63: 'Eu', 2: 'He', 31: 'Ga', 64: 'Gd', 3: 'Li', 32: 'Ge', 65: 'Tb', 4: 'Be', 33: 'As', 66: 'Dy', 5: 'B', 34: 'Se', 67: 'Ho', 6: 'C', 35: 'Br', 68: 'Er', 36: 'Kr', 69: 'Tm', 37: 'Rb', 70: 'Yb', 7: 'N', 38: 'Sr', 71: 'Lu', 8: 'O', 39: 'Y', 72: 'Hf', 9: 'F', 40: 'Zr', 73: 'Ta', 10: 'Ne', 41: 'Nb', 74: 'W', 11: 'Na', 42: 'Mo', 75: 'Re', 12: 'Mg', 43: 'Tc', 76: 'Os', 13: 'Al', 44: 'Ru', 77: 'Ir', 14: 'Si', 45: 'Rh', 78: 'Pt', 15: 'P', 46: 'Pd', 79: 'Au', 16: 'S', 47: 'Ag', 80: 'Hg', 17: 'Cl', 48: 'Cd', 81: 'Tl', 18: 'Ar', 49: 'In', 82: 'Pb', 19: 'K', 50: 'Sn', 83: 'Bi', 20: 'Ca', 51: 'Sb', 84: 'Po', 21: 'Sc', 52: 'Te', 85: 'At', 22: 'Ti', 53: 'I', 86: 'Rn', 23: 'V', 54: 'Xe', 87: 'Fr', 24: 'Cr', 55: 'Cs', 88: 'Ra', 25: 'Mn', 56: 'Ba', 89: 'Ac', 57: 'La', 90: 'Th', 26: 'Fe', 58: 'Ce', 91: 'Pa', 59: 'Pr', 92: 'U', 27: 'Co', 60: 'Nd', 93: 'Np', 61: 'Pm', 94: 'Pu', 28: 'Ni', 62: 'Sm', 95: 'Am', 29: 'Cu', 96: 'Cm'}
elemradius = {1: 0.50, 30: 1.22, 63: 1.98, 2: 0.28, 31: 1.22, 64: 1.96, 3: 1.28, 32: 1.2, 65: 1.94, 4: 0.96, 33: 1.19, 66: 1.92, 5: 0.84, 34: 1.2, 67: 1.92, 6: 0.76, 35: 1.2, 68: 1.89, 36: 1.16, 69: 1.9, 37: 2.2, 70: 1.87, 7: 0.71, 38: 1.95, 71: 1.87, 8: 0.66, 39: 1.9, 72: 1.75, 9: 0.57, 40: 1.75, 73: 1.7, 10: 0.58, 41: 1.64, 74: 1.62, 11: 1.66, 42: 1.54, 75: 1.51, 12: 1.41, 43: 1.47, 76: 1.44, 13: 1.21, 44: 1.46, 77: 1.41, 14: 1.11, 45: 1.42, 78: 1.36, 15: 1.07, 46: 1.39, 79: 1.36, 16: 1.05, 47: 1.45, 80: 1.32, 17: 1.02, 48: 1.44, 81: 1.45, 18: 1.06, 49: 1.42, 82: 1.46, 19: 2.03, 50: 1.39, 83: 1.48, 20: 1.76, 51: 1.39, 84: 1.4, 21: 1.7, 52: 1.38, 85: 1.5, 22: 1.6, 53: 1.39, 86: 1.5, 23: 1.53, 54: 1.4, 87: 2.6, 24: 1.39, 55: 2.44, 88: 2.21, 25: 1.39, 56: 2.15, 89: 2.15, 57: 2.07, 90: 2.06, 26: 1.32, 58: 2.04, 91: 2.0, 59: 2.03, 92: 1.96, 27: 1.26, 60: 2.01, 93: 1.9, 61: 1.99, 94: 1.87, 28: 1.24, 62: 1.98, 95: 1.8, 29: 1.32, 96: 1.69}
elemradius = {elemlabel[i]: elemradius[i] for i in elemlabel}
elemidxs = {elemlabel[k]:k for k in elemlabel}
elemcolors = {"H": (1.0, 1.0, 1.0, 1.0), "C": (0.1, 0.1, 0.1, 1.0), "N": (0.0, 0.0, 1.0, 1.0), "O": (1.0, 0.0, 0.0, 1.0), "S": (1.0, 1.0, 0.0, 1.0)}


class Molecule:
    "Class for store a molecule. Can be loaded from .xyz files."
    def __init__(self, elems, coords, bonds = []):
        self.elems = elems
        self.coords = np.array(coords, dtype = np.float32)
        self.n_atom = len(elems)

        if len(bonds) == 0 and self.n_atom > 1:
            self.form_bonds()
        else:
            self.bonds = bonds
            self.bondpts = np.array([[self.coords[bond[0]], self.coords[bond[1]]] for bond in bonds])

    def form_bonds(self):
        self.bonds = []
        self.bondpts = []
        for i in range(self.n_atom-1):
            for j in range(i+1, self.n_atom):
                l = np.linalg.norm(self.coords[i] - self.coords[j])**0.5
                if l <= 1.2 * (elemradius[self.elems[i]] + elemradius[self.elems[j]]):
                    self.bonds.append((i, j))
                    self.bondpts.append([self.coords[i], self.coords[j]])
        self.bondpts = np.array(self.bondpts)

    def from_xyz(self, fname):
        f = open(fname, "r")
        lines = f.readlines()
        f.close()
        curlineidx = 0
        elements, datas = [], []
        while curlineidx < len(lines) and lines[curlineidx].strip():
            natom = int(lines[curlineidx])
            datastr = lines[curlineidx+2:curlineidx+2+natom]
            data = np.empty((natom, 3), dtype = float)
            element = []
            for i, line in enumerate(datastr):
                temp = line.split()
                if len(temp) != 4:
                    continue
                a, x, y, z = temp
                element.append(a)
                data[i][0] = float(x)
                data[i][1] = float(y)
                data[i][2] = float(z)
            elements.append(element)
            datas.append(data)
            curlineidx = curlineidx+2+natom
        self.elems = elements[-1]
        self.coords = datas[-1]
        self.form_bonds()

    def to_xyz(self, fname):
        element = self.elems
        data = self.coords
        f = open(fname, "w")
        f.write(str(len(element)) + "\n")
        f.write("\n")
        for i in range(len(element)):
            f.write(f"{element[i]}    {data[i][0]:>12.8f}    {data[i][1]:>12.8f}    {data[i][2]:>12.8f}\n")
        f.close()

class MoldenWavefunction:
    "Class for store the wavefunction. Can be loaded from .molden files."
    def __init__(self, fp:str):

        self.logger = MyLogger()
        self.logger.log(f"Load wavefunction {fp}.")

        self.molecule = None
        self.gtoshells = []
        self.gtos = []
        self.n_gtf = 0
        self.C = []
        self.C_raveled = self.C
        self.energys = []
        self.occupys = []
        self.spins = []
        self.homo = -1
        self.lumo = -1
        self.originaltype = "spherical"

        self.temp = None    # temp data for storing the raveled GTF array
        self.convert_mats = {"s":S_convert, "p":P_convert, "d":D_convert, "f":F_convert, "g":G_convert, "h":H_convert}
        self.inverse_mats = {"s":S_convert, "p":P_convert, "d":D_inverse, "f":F_inverse, "g":G_inverse, "h":H_inverse}

        self.read(fp)
        self.convert_to_cartesian()
        self.find_frontier()
        self.ravel_gtoshells()
        self.get_raveled_gtf()
        self.get_raveled_C()

    def find_frontier(self):
        homo, lumo = -1, -1
        for i, occu in enumerate(self.occupys):
            if occu != 0:
                homo = i
            if lumo < 0 and occu == 0:
                lumo = i
        self.homo, self.lumo = homo, lumo
        self.logger.log(f"Orbital {homo} is HOMO, E = {self.energys[homo]:6.6f} Hartree")
        self.logger.log(f"Orbital {lumo} is LUMO, E = {self.energys[lumo]:6.6f} Hartree")

    def ravel_gtoshells(self):
        for gtoshell in self.gtoshells:
            self.gtos.extend(gtoshell.gtos)

    def get_raveled_gtf(self):
        """ravel the coefficients into the gaussian function sequence. basis functions containing multiple gaussian functions are raveled into multiple coeffients"""
        if self.temp:
            return self.temp["c0"], self.temp["a"], self.temp["p"], self.temp["pow"], self.temp["atomidx"]

        n_gtf = sum([len(gto.funcs) for gto in self.gtos])
        self.n_gtf = n_gtf
        coeffs0 = np.empty(n_gtf, dtype = np.float32)
        
        contracts = np.empty(n_gtf, dtype = np.float32)
        positions = np.empty((n_gtf, 3), dtype = np.float32)
        powers = np.empty((n_gtf, 3), dtype = np.int32)
        atomidxs = []
        cn = 0
        for i, gto in enumerate(self.gtos):
            atomidxs.extend([gto.atomidx for gtf in gto.funcs])
            for gtf in gto.funcs:
                coeffs0[cn] = gtf.c
                contracts[cn] = gtf.a
                positions[cn] = gto.p
                powers[cn,0], powers[cn,1], powers[cn,2] = gtf.i, gtf.j, gtf.k
                cn += 1
        atomidxs = np.ascontiguousarray(atomidxs, dtype = np.int32)
        self.temp = {"c0":coeffs0, "a":contracts, "p":positions, "pow":powers, "atomidx":atomidxs}
        self.logger.log(f"Raveled the basis set into {cn} gaussian functions.")
        return coeffs0, contracts, positions, powers, atomidxs

    def get_raveled_C(self):
        """ravel the coefficients into the gaussian function sequence. basis functions containing multiple gaussian functions are raveled into multiple coeffients"""
        if type(self.C) == type(self.C_raveled) == np.ndarray and self.C.shape == self.C_raveled.shape:
            return self.C
        if isinstance(self.C_raveled, np.ndarray) and self.C_raveled.shape[0] == self.n_gtf:
            return self.C_raveled
        n_gtf = sum([len(gto.funcs) for gto in self.gtos])
        self.n_gtf = n_gtf
        self.C_raveled = np.empty((n_gtf, self.C.shape[0]))
        cn = 0
        for i, gto in enumerate(self.gtos):
            for gtf in gto.funcs:
                self.C_raveled[cn] = self.C[i]
                cn += 1
        return self.C_raveled

    def convert_to_cartesian(self):
        """it is more convenient to calculate cartesian basis set"""
        self.C = self.C.T   # 因为一个错误导致这个函数是按照C是行向量来写的。先把列向量的C转成行向量，然后再转回去
        cat_len = {"s":1, "p":3, "d":6, "f":10, "g":15, "h":21}
        sph_len = {"s":1, "p":3, "d":5, "f": 7, "g": 9, "h":11}
        n_cat, n_sph = 0, 0
        for i, shell in enumerate(self.gtoshells):
            n_cat += cat_len[shell.s]
            n_sph += sph_len[shell.s]
        if n_cat == self.C.shape[1]:
            self.logger.log(f"Cartesian basis set detected. Total {n_cat} gtos with {len(self.gtoshells)} shells.")
            self.C = self.C.T
            self.originaltype = "cartesian"
            return
        elif n_sph != self.C.shape[1]:
            raise ValueError(f"The number of MO coefficnents are supposed to be {n_cat} for cartesian basis or {n_sph} for spherical basis, not {self.C.shape[1]}")
        self.logger.log(f"Spherical basis set detected. Total {n_sph} gtos with {len(self.gtoshells)} shells.")
        self.originaltype = "spherical"
        sphb, catb = 0, 0
        newC = np.zeros((self.C.shape[0], n_cat), dtype = np.float32)
        for i, shell in enumerate(self.gtoshells):
            icat, isph = cat_len[shell.s], sph_len[shell.s]
            C_sph = self.C[:,sphb:sphb+isph]
            C_cat = C_sph @ self.convert_mats[shell.s]
            newC[:,catb:catb+icat] = C_cat
            sphb += isph
            catb += icat
        self.C = np.concatenate((newC, np.zeros((n_cat-n_sph, n_cat), dtype = np.float32)), axis = 0).T
        self.energys = np.concatenate((self.energys, np.zeros(n_cat-n_sph, dtype=np.float32)))
        self.occupys = np.concatenate((self.occupys, np.zeros(n_cat-n_sph, dtype=np.float32)))
        self.spins = np.concatenate((self.spins, np.zeros(n_cat-n_sph, dtype=np.float32)))
        self.logger.log(f"Finished converting the basis set to cartesian.")

    def convert_to_spherical(self, output = False):
        """convert the cartesian basis set into spherical to make it readble for multiwfn"""
        self.C = self.C.T   # 因为一个错误导致这个函数是按照C是行向量来写的。先把列向量的C转成行向量，然后再转回去
        cat_len = {"s":1, "p":3, "d":6, "f":10, "g":15, "h":21}
        sph_len = {"s":1, "p":3, "d":5, "f": 7, "g": 9, "h":11}
        n_cat, n_sph = 0, 0
        for i, shell in enumerate(self.gtoshells):
            n_cat += cat_len[shell.s]
            n_sph += sph_len[shell.s]
        if n_sph == self.C.shape[1]:
            self.logger.log(f"Spherical basis set detected. Do not convert.")
            self.C = self.C.T
            return
        elif n_cat != self.C.shape[1]:
            raise ValueError(f"The number of MO coefficnents are supposed to be {n_cat} for cartesian basis or {n_sph} for spherical basis, not {self.C.shape[1]}")
        self.logger.log(f"Cartesian basis set detected. Total {n_sph} gtos with {len(self.gtoshells)} shells.")
        sphb, catb = 0, 0
        newC = np.zeros((self.C.shape[0], n_sph), dtype = np.float32)
        for i, shell in enumerate(self.gtoshells):
            icat, isph = cat_len[shell.s], sph_len[shell.s]
            C_cat = self.C[:,catb:catb+icat]
            C_sph = C_cat @ self.inverse_mats[shell.s]
            newC[:,sphb:sphb+isph] = C_sph
            sphb += isph
            catb += icat
        self.C = newC[:n_sph].T
        self.energys = self.energys[:n_sph]
        self.occupys = self.occupys[:n_sph]
        self.spins = self.spins[:n_sph]
        self.logger.log(f"Finished converting the basis set to spherical.")

    def write(self, fp:str):
        self.logger.log(fp)
        with open(fp, "w", errors="ignore") as f:
            f.write("[Molden Format]\n")
            f.write("[Title]\n")
            f.write("Written by molden_convert.py. The label 'orca' is added to let multiwfn can correctly read it\n")
            f.write("[Atoms] AU\n")
            for i in range(self.molecule.n_atom):
                mf = self.molecule
                f.write(f"{mf.elems[i]:6s}{i+1:<6d}{elemidxs[mf.elems[i]]:<6d}{mf.coords[i,0]:16.6f}{mf.coords[i,1]:16.6f}{mf.coords[i,2]:16.6f}\n")
            f.write("[GTO]")
            atomidx = -1
            for i, gtoshell in enumerate(self.gtoshells):
                gtoshell:GTOShell
                if gtoshell.atomidx != atomidx:
                    atomidx = gtoshell.atomidx
                    f.write(f"\n    {atomidx} 0\n")
                f.write(f" {gtoshell.s:4s}{len(gtoshell.a)} 1.00\n")
                for j in range(len(gtoshell.a)):
                    f.write(f" {gtoshell.a[j]:24.7f}{gtoshell.c[j]:24.7f}\n")
            f.write("\n\n[MO]\n")
            for i in range(len(self.energys)):
                f.write(f"Ene= {self.energys[i]:<10.4f}\n")
                f.write(f"Spin= Alpha\n")
                f.write(f"Occup= {self.occupys[i]:<10.4f}\n")
                for j in range(self.C.shape[0]):
                    f.write(f"{j+1:6d}{self.C[j,i]:16.8f}\n")

    def read(self, fp:str):
        self.logger.log(fp)
        elems, coords = [], []
        with open(fp, "r", errors="ignore") as f:
            n_orb, n_coeff, N_coeff = 0, 0, 0
            section, i, crdunit = "", 0, "AU"
            sections = set(["[Title]", "[Atoms]", "[GTO]", "[MO]"])
            while True:
                line = f.readline()
                if not line:
                    break
                data = line.split()
                if not data:
                    continue
                if data[0] in sections:
                    section = data[0]
                    if data[0] == "[Atoms]" and len(data) >= 2:
                        crdunit = data[1]
                elif section == "[Atoms]":
                    elem, atmid, atmwt = data[0], int(data[1]), int(data[2])
                    x, y, z = float(data[3]), float(data[4]), float(data[5])
                    if crdunit == "Angs":
                        x, y, z = x/0.529177249, y/0.529177249, z/0.529177249
                    coords.append([x, y, z])
                    elems.append(elem)

                elif section == "[GTO]":
                    if len(data) != 2 or data[0].find(".") != -1:
                        continue
                    tmp = line.split()
                    atmidx, _ = int(tmp[0]), int(tmp[1])
                    j = 1
                    while j < 1000:
                        ndata = f.readline().split()
                        if len(ndata) != 3:
                            break
                        slb, ngto, _ = ndata[0], int(ndata[1]), ndata[2]
                        contracts, coefficients = np.zeros(ngto, dtype = np.float32), np.zeros(ngto, dtype = np.float32)
                        for k in range(0, ngto):
                            tmp = f.readline().split()[:2]
                            contracts[k] = float(tmp[0])
                            coefficients[k] = float(tmp[1])
                        self.gtoshells.append(GTOShell(slb, contracts, coefficients, coords[atmidx-1], atmidx))
                        j += ngto + 1

                if section == "[MO]":
                    if data[0].find("Ene") != -1 and len(self.C) == 0:
                        self.C.append([])
                        self.energys.append(float(data[1]))
                        for iii in range(10):
                            line = f.readline()
                            data = line.split()
                            if data[0][-1] != "=":
                                self.C[-1].append(float(data[1]))
                                break
                            if data[0].find("Spin") != -1:
                                self.spins.append(data[1])
                            if data[0].find("Occup") != -1:
                                self.occupys.append(float(data[1]))
                        while True:
                            line = f.readline()
                            data = line.split()
                            if len(data) != 2 or data[0].find("=") != -1:
                                break
                            iii, ccc = data[0], data[1]
                            spltidx = line.find(ccc)-3
                            self.C[-1].append(float(line[spltidx:]))

                if section == "[MO]" and len(self.C) > 0:
                    if data[0].find("=") != -1:
                        if data[0].find("Ene") != -1:
                            self.energys.append(float(data[1]))
                            n_coeff = 0
                            n_orb += 1
                            if n_orb == 1:
                                Cs = np.empty((len(self.C[-1]), len(self.C[-1])), dtype = np.float32)
                                Cs[0] = np.array(self.C[0], dtype = np.float32)
                                self.C = Cs
                                N_coeff = self.C.shape[0]

                        elif data[0].find("Spin") != -1:
                            self.spins.append(data[1])
                        elif data[0].find("Occup") != -1:
                            self.occupys.append(float(data[1]))
                    else:
                        while True:
                            self.C[n_orb,n_coeff] = float(line[spltidx:])
                            if n_coeff >= N_coeff-1:
                                break
                            line = f.readline()
                            n_coeff += 1


        self.C = np.array(self.C, dtype=np.float32)
        n_energy, n_coeffs = self.C.shape
        if n_energy > n_coeffs:
            self.C = self.C[:n_coeffs,:]
        elif n_energy < n_coeffs:
            self.C = np.concatenate((self.C, np.zeros((n_coeffs - n_energy, n_coeffs))), axis = 0)
        n, n = self.C.shape
        self.C = self.C.T
        self.molecule = Molecule(elems, coords)
        self.energys = np.array(self.energys[:n])
        self.occupys = np.array(self.occupys[:n])
        self.spins = np.array(self.spins[:n])
        self.logger.log(f"Total {self.molecule.n_atom} atom detected.")
        self.logger.log(f"Total {len(self.energys)} orbitals detected.")

class Excitation:
    "Class for store the excitation data"
    def __init__(self):
        self.orb1 = [1,]
        self.orb2 = [2,]
        self.cisc = [1.,]
        self.osci = 1.0
        self.e = 1.0
        self.wlen = 45.5640
        self.Tx, self.Ty, self.Tz, self.T2 = 0.0, 0.0, 0.0, 0.0
        self.vTx, self.vTy, self.vTz, self.vT2 = 0.0, 0.0, 0.0, 0.0

    def check_normalize(self):
        sum2 = sum([c**2 for c in self.cisc])
        return sum2**0.5 > 0.95

    def filter_coefficients(self, threshold = 0.01, normalize = False):
        self.orb1 = np.array(self.orb1, dtype = np.int32)
        self.orb2 = np.array(self.orb2, dtype = np.int32)
        self.cisc = np.array(self.cisc, dtype = np.float32)
        idxs = np.nonzero(self.cisc > threshold)[0]
        self.orb1 = self.orb1[idxs]
        self.orb2 = self.orb2[idxs]
        self.cisc = self.cisc[idxs]
        l = len(self.cisc)
        if normalize:
            sum2 = sum([c**2 for c in self.cisc])
            sum1 = sum2**0.5
            self.cisc = [c / sum1 for c in self.cisc]


def read_cis_output(fp:str):
    "Function for reading TeraChem TDDFT/CIS outputs. Returns a list of excitation objects."
    with open(fp, "r", errors="ignore") as f:
        for i in range(9999999):
            line = f.readline()
            if line.find("Transition dipole moments") != -1:
                curlinestart = f.tell()
                break
        # reading transition dipole
        f.seek(curlinestart)
        dipoledata = []
        patt = r'(?P<ridx>\d+) +(?P<Tx>-?\d+\.\d+) +(?P<Ty>-?\d+\.\d+) +(?P<Tz>-?\d+\.\d+) +(?P<T>-?\d+\.\d+)'
        for i in range(9999999):
            line = f.readline()
            if line.find("Transition dipole moments between excited states:") != -1:
                curlinestart = f.tell()
                break
            result = re.search(patt, line)
            if not result:
                continue
            result = result.groupdict()
            dipoledata.append([float(result["Tx"]), float(result["Ty"]), float(result["Tz"]), float(result["T"])**2])

        # skipping infos
        f.seek(curlinestart)
        for i in range(9999999):
            curlinestart = f.tell()
            line = f.readline()
            if line.find("Velocity transition dipole moments") != -1:
                curlinestart = f.tell()
                break

        # reading velocity dipole
        f.seek(curlinestart)
        vpoledata = []
        patt = r'(?P<ridx>\d+) +(?P<Tx>-?\d+\.\d+) +(?P<Ty>-?\d+\.\d+) +(?P<Tz>-?\d+\.\d+) +(?P<T>-?\d+\.\d+)'
        for i in range(9999999):
            line = f.readline()
            if line.find("Magnetic transition dipole moments and rotational strengths") != -1:
                curlinestart = f.tell()
                break
            result = re.search(patt, line)
            if not result:
                continue
            result = result.groupdict()
            vpoledata.append([float(result["Tx"]), float(result["Ty"]), float(result["Tz"]), float(result["T"])**2])

        # skipping infos
        f.seek(curlinestart)
        for i in range(9999999):
            curlinestart = f.tell()
            line = f.readline()
            if line.find("Largest CI coefficients") != -1:
                break

        # reading CI coefficient
        f.seek(curlinestart)
        occus, virts, coeffs = [], [], []
        patt = r"(?P<occu>\d+) +-> +(?P<virt>\d+) +:.+ +(?P<coeff>-?\d+\.\d+)"
        for i in range(9999999):
            curlinestart = f.tell()
            line = f.readline()
            if line.find("Final Excited State Results:") != -1:
                break
            if line.find("Largest CI coefficients") != -1:
                occus.append([])
                virts.append([])
                coeffs.append([])
            result = re.search(patt, line)
            if not result:
                continue
            result = result.groupdict()
            occus[-1].append(int(result["occu"])-1)
            virts[-1].append(int(result["virt"])-1)
            coeffs[-1].append(float(result["coeff"]))
        f.seek(curlinestart)
        eexts, oscis = [], []
        for i in range(1000):
            line = f.readline()
            if not line:
                break
            patt = r"(?P<ridx>\d+) +(?P<tene>-?\d+\.\d+) +(?P<eext>-?\d+\.\d+) +(?P<osci>-?\d+\.\d+) +(?P<s2>-?\d+\.\d+)"
            result = re.search(patt, line)
            if not result:
                continue
            eexts.append(float(result["eext"]) / 27.21139664130791)
            oscis.append(float(result["osci"]))
    
    if not (len(dipoledata) == len(occus) == len(virts) == len(coeffs) == len(eexts) == len(oscis)):
        raise ValueError(f"Errors found in reading excitation info: {len(occus)} single excitation determinants, {len(coeffs)} CIS coefficients, {len(eexts)} excitation energy, {len(oscis)} oscillator strenth, and {len(dipoledata)} transition dipole moments.")

    excitations = []
    for i in range(len(eexts)):
        e = Excitation()
        e.cisc = coeffs[i]
        e.e = eexts[i]
        if not (len(coeffs[i]) == len(occus[i]) == len(virts[i])):
            raise ValueError(f"Errors found in reading excitation {i}. It contains {len(coeffs[i])} CIS coefficients and {len(occus[i])} transition orbitals.")
        e.osci = oscis[i]
        e.orb1 = occus[i]
        e.orb2 = virts[i]
        e.Tx, e.Ty, e.Tz, e.T2 = dipoledata[i]
        e.vTx, e.vTy, e.vTz, e.vT2 = vpoledata[i]
        e.wlen = 45.56337117 / e.e
        excitations.append(e)
    return excitations

def write_multiwfn_readable_orca_output_file(fname:str, extlist):
    """Generate a ORCA TDDFT/CIS type output file from TeraChem TDDFT/CIS output. So that can be readed by Multiwfn"""
    occus, virts = [], []
    for ext in extlist:
        occus += list(ext.orb1)
        virts += list(ext.orb2)
    
    f = open(fname, "w")
    f.write("""
    
                                 *****************
                                 * O   R   C   A *
                                 *****************

                                            #,                                       
                                            ###                                      
                                            ####                                     
                                            #####                                    
                                            ######                                   
                                           ########,                                 
                                     ,,################,,,,,                         
                               ,,#################################,,                 
                          ,,##########################################,,             
                       ,#########################################, ''#####,          
                    ,#############################################,,   '####,        
                  ,##################################################,,,,####,       
                ,###########''''           ''''###############################       
              ,#####''   ,,,,##########,,,,          '''####'''          '####       
            ,##' ,,,,###########################,,,                        '##       
           ' ,,###''''                  '''############,,,                           
         ,,##''                                '''############,,,,        ,,,,,,###''
      ,#''                                            '''#######################'''  
     '                                                          ''''####''''         
             ,#######,   #######,   ,#######,      ##                                
            ,#'     '#,  ##    ##  ,#'     '#,    #''#        ######   ,####,        
            ##       ##  ##   ,#'  ##            #'  '#       #        #'  '#        
            ##       ##  #######   ##           ,######,      #####,   #    #        
            '#,     ,#'  ##    ##  '#,     ,#' ,#      #,         ##   #,  ,#        
             '#######'   ##     ##  '#######'  #'      '#     #####' # '####'        

    \n""")
    f.write(""" 
       ***********************************************************
       *                    TeraChem v1.9-2021.10-dev            *
       *                 Hg Version:                             *
       *                   Development Version                   *
       *           Chemistry at the Speed of Graphics!           *
       ***********************************************************
       *        The original data is generated by TeraChem       *
       *             Converted by read_excitation.py             *
       ***********************************************************

------------------------------------------------------------------------------
            THIS OUTPUT ONLY CONTAINS EXCITATION CI COEFFICIENTS
                        OTHER INFORMATION ARE INVALID
------------------------------------------------------------------------------
    \n""")

    f.write(f"""
------------------------------------------------------------------------------
                        ORCA TD-DFT/TDA CALCULATION
------------------------------------------------------------------------------

Input orbitals are from        ... {fname}.gbw
CI-vector output               ... {fname}.cis
Tamm-Dancoff approximation     ... operative
CIS-Integral strategy          ... AO-integrals
Integral handling              ... AO integral Direct
Max. core memory used          ... 2048 MB
Reference state                ... RHF
Generation of triplets         ... off
Follow IRoot                   ... off
Number of operators            ... 1
Orbital ranges used for CIS calculation:
 Operator 0:  Orbitals  0...{max(occus)}  to {min(virts)}...1631
XAS localization array:
 Operator 0:  Orbitals  -1... -1
    \n""")

    f.write("""
     *** TD-DFT CALCULATION INITIALIZED ***

------------------------
DAVIDSON-DIAGONALIZATION
------------------------

Dimension of the eigenvalue problem            ... 131072
Number of roots to be determined               ...      5
Maximum size of the expansion space            ...     50
Maximum number of iterations                   ...    100
Convergence tolerance for the residual         ...    2.500e-07
Convergence tolerance for the energies         ...    2.500e-07
Orthogonality tolerance                        ...    1.000e-14
Level Shift                                    ...    0.000e+00
Constructing the preconditioner                ... o.k.
Building the initial guess                     ... o.k.
Number of trial vectors determined             ...     50


                       ****Iteration    0****

   Memory handling for direct AO based CIS:
   Memory per vector needed      ...   128 MB
   Memory needed                 ...  2048 MB
   Memory available              ...  2048 MB
   Number of vectors per batch   ...    16
   Number of batches             ...     2
   Time for densities:            0.000
   Time for RI-J (Direct):        0.000
   Time for K (COSX):             0.000
   Time for XC-Integration:       0.000
   Time for Sigma-Completion:     0.000
   Time for densities:            0.000
   Time for RI-J (Direct):        0.000
   Time for K (COSX):             0.000
   Time for XC-Integration:       0.000
   Time for Sigma-Completion:     0.000
   Size of expansion space: 15
   Lowest Energy          :     0.000000000000
   Maximum Energy change  :     0.000000000000 (vector 1)
   Maximum residual norm  :     0.000000000000


      *** CONVERGENCE OF RESIDUAL NORM REACHED ***

Storing the converged CI vectors               ... hello_world.cis

                 *** DAVIDSON DONE ***

Total time for solving the CIS problem:   702.861sec
    \n""")
    f.write("""
------------------------------------
TD-DFT/TDA EXCITED STATES (SINGLETS)
------------------------------------

the weight of the individual excitations are printed if larger than 1.0e-02""")
    for i, ext in enumerate(extlist):
        ext:Excitation
        f.write(f"\nSTATE  {i+1}:  E=   {ext.e:.6f} au      {ext.e*27.211:>.3f} eV    {ext.e*219474.6:>6.1f} cm**-1 <S**2> =   0.000000\n")
        for orb1, orb2, coeff in zip(ext.orb1, ext.orb2, ext.cisc):
            f.write(f"   {orb1}a -> {orb2}a  :   {coeff**2:>3.6f} (c={coeff:>3.8f})\n")

    f.write("""
-----------------------------
TD-DFT/TDA-EXCITATION SPECTRA
-----------------------------

Center of mass = (  0.0000,  0.0000,  3.4488)
Calculating the Dipole integrals                  ... done
Transforming integrals                            ... done
Calculating the Linear Momentum integrals         ... done
Transforming integrals                            ... done
Calculating angular momentum integrals            ... done
Transforming integrals                            ... done
    """)
    f.write("""
-----------------------------------------------------------------------------
         ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
-----------------------------------------------------------------------------
State   Energy    Wavelength  fosc         T2        TX        TY        TZ  
        (cm-1)      (nm)                 (au**2)    (au)      (au)      (au) 
-----------------------------------------------------------------------------
""")
    for i, ext in enumerate(extlist):
        ext:Excitation
        f.write(f"   {i+1:<2d}   {ext.e*219474.6: <6.1f}    {ext.wlen: <3.1f}   {ext.osci: <2.9f}   {ext.T2: <2.5f}  {ext.Tx: <2.5f}   {ext.Ty: <2.5f}   {ext.Tz :<2.5f}\n")

    f.write("""
-----------------------------------------------------------------------------
         ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS
-----------------------------------------------------------------------------
State   Energy    Wavelength   fosc        P2         PX        PY        PZ  
        (cm-1)      (nm)                 (au**2)     (au)      (au)      (au) 
-----------------------------------------------------------------------------
""")
    for i, ext in enumerate(extlist):
        ext:Excitation
        f.write(f"   {i+1:>2d}   {ext.e*219474.6:>6.1f}    {ext.wlen:>3.1f}   {ext.osci:>2.9f}   {ext.vT2:>2.5f}  {ext.vTx:>2.5f}   {ext.vTy:>2.5f}   {ext.vTz:>2.5f}\n")
    f.write("""
Total run time:      0.000 sec

           *** ORCA-CIS/TD-DFT FINISHED WITHOUT ERROR ***

Maximum memory used throughout the entire CIS-calculation: 1024.0 MB

-----------------------
CIS/TD-DFT TOTAL ENERGY
-----------------------

    E(SCF)  =      0.000000000 Eh
    DE(CIS) =      0.000000000 Eh (Root  1)
    ----------------------------- ---------
    E(tot)  =      0.000000000 Eh



-------------------------   ----------------
Dispersion correction           -0.000000000
-------------------------   ----------------


-------------------------   --------------------
FINAL SINGLE POINT ENERGY         0.000000000000
-------------------------   --------------------



                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 0 hours 0 minutes 0 seconds 0 msec
    """)


if __name__ == "__main__":
    fname = sys.argv[1]
    if fname.find(".molden") != -1:
        wf = MoldenWavefunction(fname)
        if wf.originaltype == "cartesian":
            wf.convert_to_spherical()
            wf.write(fname.replace(".molden", "-spherical.molden"))
        elif wf.originaltype == "spherical":
            wf.convert_to_cartesian()
            wf.write(fname.replace(".molden", "-cartesian.molden"))
    elif fname.find(".out") != -1:
        exts = read_cis_output(sys.argv[1])
        write_multiwfn_readable_orca_output_file(sys.argv[1].replace(".out", "-orca.out"), exts)
