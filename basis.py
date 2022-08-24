import numpy as np
from math import sqrt, factorial, pi

class GaussianFunction:
    def __init__(self, p, c, a, i, j, k):
        self.c, self.a, self.p = c, a, p
        self.i, self.j, self.k = i, j, k
        self.A = (2*a/pi)**0.75 * sqrt((8*a)**(i+j+k)*factorial(i)*factorial(j)*factorial(k)/(factorial(2*i)*factorial(2*j)*factorial(2*k)))

    def __call__(self, x, y, z):
        dx, dy, dz = x-self.p[0], y-self.p[1], z-self.p[2]
        return self.A * self.c * (dx**self.i * dy**self.j * dz**self.k) * np.exp(-self.a * (dx**2 + dy**2 + dz**2))

class GTO:
    def __init__(self, position, contracts, coefficients, px, py, pz, atomidx):
        self.c = coefficients
        self.a = contracts
        self.p = position
        self.px = px
        self.py = py
        self.pz = pz
        self.atomidx = atomidx
        self.funcs = [GaussianFunction(self.p, c, a, px, py, pz) for c, a in zip(self.c, self.a)]
        self.A = self.funcs[0].A

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
