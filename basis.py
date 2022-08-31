import numpy as np
from math import sqrt, factorial, pi

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
], dtype = np.float32)

# 这是专用orca的转换矩阵。相对于multiwfn提供的表格，orca的F+3和G+4轨道的相位是反着的。因此对于其他程序可能要把转移矩阵的后两列乘以-1
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
], dtype = np.float32)
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
], dtype = np.float32)

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
