from utils import *
from basis import *

class MoldenWavefunction:
    def __init__(self, fp:str):
        self.n_atom = 0
        self.elems = []
        self.bonds = []
        self.bondpts = []
        self.coords = []
        self.elemlabel = {1: 'H', 30: 'Zn', 63: 'Eu', 2: 'He', 31: 'Ga', 64: 'Gd', 3: 'Li', 32: 'Ge', 65: 'Tb', 4: 'Be', 33: 'As', 66: 'Dy', 5: 'B', 34: 'Se', 67: 'Ho', 6: 'C', 35: 'Br', 68: 'Er', 36: 'Kr', 69: 'Tm', 37: 'Rb', 70: 'Yb', 7: 'N', 38: 'Sr', 71: 'Lu', 8: 'O', 39: 'Y', 72: 'Hf', 9: 'F', 40: 'Zr', 73: 'Ta', 10: 'Ne', 41: 'Nb', 74: 'W', 11: 'Na', 42: 'Mo', 75: 'Re', 12: 'Mg', 43: 'Tc', 76: 'Os', 13: 'Al', 44: 'Ru', 77: 'Ir', 14: 'Si', 45: 'Rh', 78: 'Pt', 15: 'P', 46: 'Pd', 79: 'Au', 16: 'S', 47: 'Ag', 80: 'Hg', 17: 'Cl', 48: 'Cd', 81: 'Tl', 18: 'Ar', 49: 'In', 82: 'Pb', 19: 'K', 50: 'Sn', 83: 'Bi', 20: 'Ca', 51: 'Sb', 84: 'Po', 21: 'Sc', 52: 'Te', 85: 'At', 22: 'Ti', 53: 'I', 86: 'Rn', 23: 'V', 54: 'Xe', 87: 'Fr', 24: 'Cr', 55: 'Cs', 88: 'Ra', 25: 'Mn', 56: 'Ba', 89: 'Ac', 57: 'La', 90: 'Th', 26: 'Fe', 58: 'Ce', 91: 'Pa', 59: 'Pr', 92: 'U', 27: 'Co', 60: 'Nd', 93: 'Np', 61: 'Pm', 94: 'Pu', 28: 'Ni', 62: 'Sm', 95: 'Am', 29: 'Cu', 96: 'Cm'}
        self.elemradius = {1: 0.50, 30: 1.22, 63: 1.98, 2: 0.28, 31: 1.22, 64: 1.96, 3: 1.28, 32: 1.2, 65: 1.94, 4: 0.96, 33: 1.19, 66: 1.92, 5: 0.84, 34: 1.2, 67: 1.92, 6: 0.76, 35: 1.2, 68: 1.89, 36: 1.16, 69: 1.9, 37: 2.2, 70: 1.87, 7: 0.71, 38: 1.95, 71: 1.87, 8: 0.66, 39: 1.9, 72: 1.75, 9: 0.57, 40: 1.75, 73: 1.7, 10: 0.58, 41: 1.64, 74: 1.62, 11: 1.66, 42: 1.54, 75: 1.51, 12: 1.41, 43: 1.47, 76: 1.44, 13: 1.21, 44: 1.46, 77: 1.41, 14: 1.11, 45: 1.42, 78: 1.36, 15: 1.07, 46: 1.39, 79: 1.36, 16: 1.05, 47: 1.45, 80: 1.32, 17: 1.02, 48: 1.44, 81: 1.45, 18: 1.06, 49: 1.42, 82: 1.46, 19: 2.03, 50: 1.39, 83: 1.48, 20: 1.76, 51: 1.39, 84: 1.4, 21: 1.7, 52: 1.38, 85: 1.5, 22: 1.6, 53: 1.39, 86: 1.5, 23: 1.53, 54: 1.4, 87: 2.6, 24: 1.39, 55: 2.44, 88: 2.21, 25: 1.39, 56: 2.15, 89: 2.15, 57: 2.07, 90: 2.06, 26: 1.32, 58: 2.04, 91: 2.0, 59: 2.03, 92: 1.96, 27: 1.26, 60: 2.01, 93: 1.9, 61: 1.99, 94: 1.87, 28: 1.24, 62: 1.98, 95: 1.8, 29: 1.32, 96: 1.69}
        self.elemradius = {self.elemlabel[i]: self.elemradius[i] for i in self.elemlabel}
        self.logger = MyLogger()
        self.logger.log(f"Load wavefunction {fp}.")

        self.gtoshells = []
        self.gtos = []
        self.C = []
        self.energys = []
        self.occupys = []
        self.spins = []
        self.homo = -1
        self.lumo = -1

        self.temp = None
        self.convert_mats = {"s":S_convert, "p":P_convert, "d":D_convert, "f":F_convert, "g":G_convert, "h":np.eye(21, dtype = np.float32)}

        self.read(fp)
        self.form_bonds()
        self.convert_to_cartesian()
        self.ravel_gtoshells()

    def form_bonds(self):
        for i in range(self.n_atom-1):
            for j in range(i+1, self.n_atom):
                l = np.linalg.norm(self.coords[i] - self.coords[j])**0.5
                if l <= 1.0 * (self.elemradius[self.elems[i]] + self.elemradius[self.elems[j]]):
                    self.bonds.append((i, j))
                    self.bondpts.append([self.coords[i], self.coords[j]])
        self.bondpts = np.array(self.bondpts)

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

    def get_raveled_atomidx(self):
        if not self.temp:
            self.get_raveled_gtf()
        if "atomidx" in self.temp:
            return self.temp["atomidx"]
        self.temp["atomidx"] = []
        for i, gto in enumerate(self.gtos):
            gto:GTO
            self.temp["atomidx"].extend([gto.atomidx for gtf in gto.funcs])
        self.temp["atomidx"] = np.array(self.temp["atomidx"], dtype = np.int32)
        return self.temp["atomidx"]

    def get_raveled_gtf(self, coeffs = []):
        if self.temp:
            coeffs = np.ones_like(self.temp["c0"]) if len(coeffs) == 0 else coeffs
            cus = np.empty_like(self.temp["c0"])
            cn = 0
            for i, gto in enumerate(self.gtos):
                for gtf in gto.funcs:
                    cus[cn] = coeffs[i]
                    cn += 1
            return self.temp["c0"] * cus, self.temp["a"], self.temp["p"], self.temp["pow"]

        n_gtf = sum([len(gto.funcs) for gto in self.gtos])
        coeffs0 = np.empty(n_gtf, dtype = np.float32)
        cus = np.empty(n_gtf, dtype = np.float32)
        coeffs = np.ones_like(coeffs0) if len(coeffs) == 0 else coeffs
        
        contracts = np.empty(n_gtf, dtype = np.float32)
        positions = np.empty((n_gtf, 3), dtype = np.float32)
        powers = np.empty((n_gtf, 3), dtype = np.int32)
        cn = 0
        
        for i, gto in enumerate(self.gtos):
            for gtf in gto.funcs:
                coeffs0[cn] = gtf.A * gtf.c
                contracts[cn] = gtf.a
                positions[cn] = gto.p
                powers[cn,0], powers[cn,1], powers[cn,2] = gtf.i, gtf.j, gtf.k
                cus[cn] = coeffs[i]
                cn += 1
        self.temp = {"c0":coeffs0, "a":contracts, "p":positions, "pow":powers}
        self.logger.log(f"Raveled the basis set into {cn} gaussian functions.")
        return coeffs0 * cus, contracts, positions, powers

    def convert_to_cartesian(self):
        """it is more convenient to calculate cartesian basis set"""
        cat_len = {"s":1, "p":3, "d":6, "f":10, "g":15, "h":21}
        sph_len = {"s":1, "p":3, "d":5, "f": 7, "g": 9, "h":11}
        n_cat, n_sph = 0, 0
        for i, shell in enumerate(self.gtoshells):
            n_cat += cat_len[shell.s]
            n_sph += sph_len[shell.s]
        if n_cat == self.C.shape[1]:
            self.logger.log(f"Cartesian basis set detected. Total {n_cat} gtos with {len(self.gtoshells)} shells.")
            return
        elif n_sph != self.C.shape[1]:
            raise ValueError(f"The number of MO coefficnents are supposed to be {n_cat} for cartesian basis or {n_sph} for spherical basis, not {self.C.shape[1]}")
        self.logger.log(f"Spherical basis set detected. Total {n_sph} gtos with {len(self.gtoshells)} shells.")
        sphb, catb = 0, 0
        newC = np.zeros((self.C.shape[0], n_cat), dtype = np.float32)
        for i, shell in enumerate(self.gtoshells):
            icat, isph = cat_len[shell.s], sph_len[shell.s]
            C_sph = np.concatenate((self.C[:,sphb:sphb+isph], np.zeros((self.C.shape[0], icat-isph), dtype = np.float32)), axis = 1)
            C_cat = C_sph @ self.convert_mats[shell.s]
            newC[:,catb:catb+icat] = C_cat
            sphb += isph
            catb += icat
        self.C = np.concatenate((newC, np.zeros((n_cat-n_sph, n_cat), dtype = np.float32)), axis = 0)
        self.energys = np.concatenate((self.energys, np.zeros(n_cat-n_sph, dtype=np.float32)))
        self.occupys = np.concatenate((self.occupys, np.zeros(n_cat-n_sph, dtype=np.float32)))
        self.spins = np.concatenate((self.spins, np.zeros(n_cat-n_sph, dtype=np.float32)))
        self.logger.log(f"Finished converting the basis set to cartesian.")

    def read(self, fp:str):
        self.logger.log(fp)
        with open(fp, "r", errors="ignore") as f:
            n_orb, n_coeff, N_coeff = 0, 0, 0
            section, i = "", 0
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
                    continue

                elif section == "[Atoms]":
                    elem, atmid, atmwt = data[0], int(data[1]), int(data[2])
                    x, y, z = float(data[3]), float(data[4]), float(data[5])
                    self.coords.append([x, y, z])
                    self.elems.append(elem)

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
                        self.gtoshells.append(GTOShell(slb, contracts, coefficients, self.coords[atmidx-1], atmidx))
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
                            spltidx = line.find(ccc)-1
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
        self.n_atom = len(self.coords)
        self.coords = np.array(self.coords)
        self.energys = np.array(self.energys[:n])
        self.occupys = np.array(self.occupys[:n])
        self.spins = np.array(self.spins[:n])
        self.logger.log(f"Total {len(self.elems)} atom detected.")
        self.logger.log(f"Total {len(self.energys)} orbitals detected.")


class GaussianCube:
    def __init__(self, fp = None):
        self.n_atom = 0
        self.elems = []
        self.bonds = []
        self.bondpts = []
        self.coords = None
        self.elemlabel = {1: 'H', 30: 'Zn', 63: 'Eu', 2: 'He', 31: 'Ga', 64: 'Gd', 3: 'Li', 32: 'Ge', 65: 'Tb', 4: 'Be', 33: 'As', 66: 'Dy', 5: 'B', 34: 'Se', 67: 'Ho', 6: 'C', 35: 'Br', 68: 'Er', 36: 'Kr', 69: 'Tm', 37: 'Rb', 70: 'Yb', 7: 'N', 38: 'Sr', 71: 'Lu', 8: 'O', 39: 'Y', 72: 'Hf', 9: 'F', 40: 'Zr', 73: 'Ta', 10: 'Ne', 41: 'Nb', 74: 'W', 11: 'Na', 42: 'Mo', 75: 'Re', 12: 'Mg', 43: 'Tc', 76: 'Os', 13: 'Al', 44: 'Ru', 77: 'Ir', 14: 'Si', 45: 'Rh', 78: 'Pt', 15: 'P', 46: 'Pd', 79: 'Au', 16: 'S', 47: 'Ag', 80: 'Hg', 17: 'Cl', 48: 'Cd', 81: 'Tl', 18: 'Ar', 49: 'In', 82: 'Pb', 19: 'K', 50: 'Sn', 83: 'Bi', 20: 'Ca', 51: 'Sb', 84: 'Po', 21: 'Sc', 52: 'Te', 85: 'At', 22: 'Ti', 53: 'I', 86: 'Rn', 23: 'V', 54: 'Xe', 87: 'Fr', 24: 'Cr', 55: 'Cs', 88: 'Ra', 25: 'Mn', 56: 'Ba', 89: 'Ac', 57: 'La', 90: 'Th', 26: 'Fe', 58: 'Ce', 91: 'Pa', 59: 'Pr', 92: 'U', 27: 'Co', 60: 'Nd', 93: 'Np', 61: 'Pm', 94: 'Pu', 28: 'Ni', 62: 'Sm', 95: 'Am', 29: 'Cu', 96: 'Cm'}
        self.elemradius = {1: 0.50, 30: 1.22, 63: 1.98, 2: 0.28, 31: 1.22, 64: 1.96, 3: 1.28, 32: 1.2, 65: 1.94, 4: 0.96, 33: 1.19, 66: 1.92, 5: 0.84, 34: 1.2, 67: 1.92, 6: 0.76, 35: 1.2, 68: 1.89, 36: 1.16, 69: 1.9, 37: 2.2, 70: 1.87, 7: 0.71, 38: 1.95, 71: 1.87, 8: 0.66, 39: 1.9, 72: 1.75, 9: 0.57, 40: 1.75, 73: 1.7, 10: 0.58, 41: 1.64, 74: 1.62, 11: 1.66, 42: 1.54, 75: 1.51, 12: 1.41, 43: 1.47, 76: 1.44, 13: 1.21, 44: 1.46, 77: 1.41, 14: 1.11, 45: 1.42, 78: 1.36, 15: 1.07, 46: 1.39, 79: 1.36, 16: 1.05, 47: 1.45, 80: 1.32, 17: 1.02, 48: 1.44, 81: 1.45, 18: 1.06, 49: 1.42, 82: 1.46, 19: 2.03, 50: 1.39, 83: 1.48, 20: 1.76, 51: 1.39, 84: 1.4, 21: 1.7, 52: 1.38, 85: 1.5, 22: 1.6, 53: 1.39, 86: 1.5, 23: 1.53, 54: 1.4, 87: 2.6, 24: 1.39, 55: 2.44, 88: 2.21, 25: 1.39, 56: 2.15, 89: 2.15, 57: 2.07, 90: 2.06, 26: 1.32, 58: 2.04, 91: 2.0, 59: 2.03, 92: 1.96, 27: 1.26, 60: 2.01, 93: 1.9, 61: 1.99, 94: 1.87, 28: 1.24, 62: 1.98, 95: 1.8, 29: 1.32, 96: 1.69}
        self.elemradius = {self.elemlabel[i]: self.elemradius[i] for i in self.elemlabel}
        self.logger = MyLogger()
        self.logger.log(f"Load gaussian cube file {fp}.")

        self.xmin = 0.0
        self.ymin = 0.0
        self.zmin = 0.0
        self.nx = 1
        self.ny = 1
        self.nz = 1
        self.dx = 1.0
        self.dy = 1.0
        self.dz = 1.0
        self.data = None
        self.xmax, self.ymax, self.zmax = self.xmin+self.nx*self.dx, self.ymin+self.ny*self.dy, self.zmin+self.nz*self.dz

        if isinstance(fp, str):
            self.read_cube(fp)

    def form_bonds(self):
        for i in range(self.n_atom-1):
            for j in range(i+1, self.n_atom):
                l = np.linalg.norm(self.coords[i] - self.coords[j])**0.5
                if l <= 1.2 * (self.elemradius[self.elems[i]] + self.elemradius[self.elems[j]]):
                    self.bonds.append((i, j))
                    self.bondpts.append([self.coords[i], self.coords[j]])
        self.bondpts = np.array(self.bondpts)

    def read_block(self, f, bsize:int):
        while True:
            chunk = f.read(bsize)
            if not chunk:
                break
            yield chunk
    
    def read_cube_data(self, f, fpos:int):
        self.data = np.zeros((self.nx, self.ny, self.nz), dtype = np.float32)
        n_linebreak = (self.nz + 5) // 6
        bsize = 14 * self.nz + n_linebreak
        f.seek(fpos, 0)
        nchk = self.nx * self.ny
        nnchk = nchk // 10
        for idx, chk in enumerate(self.read_block(f, bsize)):
            x = idx // self.ny
            y = idx % self.ny
            data = ",".join(chk.split())
            if not data:
                break
            self.data[x,y] = np.fromstring(data, dtype = np.float32, sep = ",")
            if idx % nnchk == 0:
                self.logger.log(f"Reading data {idx / nchk * 100:>3.2f}%")
        self.logger.log(f"Reading finished. Total {self.data.size:d} data points loaded.")

    def read_cube(self, filename):
        with open(filename, "r", errors = "ignore") as f:
            iline = 1
            line = f.readline()
            while line:
                if iline < 3:
                    pass
                elif iline == 3:
                    n_atom, xmin, ymin, zmin = line.split()
                    self.n_atom = int(n_atom)
                    self.xmin = float(xmin)
                    self.ymin = float(ymin)
                    self.zmin = float(zmin)
                    self.coords = np.zeros((self.n_atom, 3), dtype = np.float32)
                    self.logger.log(f"There are {self.n_atom} atoms in this molecule.")
                elif iline == 4:
                    xdim, dx, _, _ = line.split()
                    self.nx, self.dx = int(xdim), float(dx)
                    self.logger.log(f"X-axis: {self.nx} grids with interval {self.dx:>1.6f} Angstrom")
                elif iline == 5:
                    ydim, _, dy, _ = line.split()
                    self.ny, self.dy = int(ydim), float(dy)
                    self.logger.log(f"Y-axis: {self.ny} grids with interval {self.dy:>1.6f} Angstrom")
                elif iline == 6:
                    zdim, _, _, dz = line.split()
                    self.nz, self.dz = int(zdim), float(dz)
                    self.logger.log(f"Z-axis: {self.nz} grids with interval {self.dz:>1.6f} Angstrom")
                    self.logger.log(f"Grid point number: {self.nx*self.ny*self.nz}")
                    self.logger.log(f"Space range: ({self.xmin:>3.3f},{self.ymin:>3.3f},{self.zmin:>3.3f})->({self.xmin+self.nx*self.dx:>3.3f},{self.ymin+self.ny*self.dy:>3.3f},{self.zmin+self.nz*self.dz:>3.3f})")
                    self.xmax, self.ymax, self.zmax = self.xmin+self.nx*self.dx, self.ymin+self.ny*self.dy, self.zmin+self.nz*self.dz
                else:
                    tmp = line.split()
                    if len(tmp) == 5:
                        self.elems.append(self.elemlabel[int(tmp[0])])
                        self.coords[iline-7] = np.array((float(tmp[2]), float(tmp[3]), float(tmp[4])), dtype = np.float32)
                    elif len(tmp) == 6:
                        self.form_bonds()
                        self.read_cube_data(f, curlinestart)
                        break
                    else:
                        pass      
                curlinestart = f.tell()
                line = f.readline()
                iline += 1
