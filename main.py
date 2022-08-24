from io import FileIO
import numpy as np
from math import sqrt, factorial, pi, exp
import time
import matplotlib.pyplot as plt

from vispy import scene, app
from vispy.visuals.filters import Alpha
from vispy.visuals.filters.clipping_planes import PlanesClipper
from vispy.color import Colormap

from utils import *
from basis import *
from read_files import *

class DensityViewer():
    def __init__(self, data):
        self.logger = MyLogger()
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1280, 960), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.bgcolor = '#444444'
        self.view.camera = 'turntable'
        self.canvas.events["key_press"].connect(self.on_key_press)

        self.planefig, self.planeax = plt.subplots(1, 1)
        self.colorbar = None
        plt.ion()
        plt.pause(0.01)

        self.current_orbital_idx = -1
        self.current_delta_var = -1
        self.current_isosurface = -1
        self.grid_dims = (128, 128, 128)

        self.volume = None
        self.isosurface_1 = None
        self.isosurface_2 = None
        self.mode = "volume"

        n_color = 75
        rs = np.concatenate([np.zeros(n_color // 2), np.ones(n_color - n_color // 2)]) * 0.90
        gs = np.ones(n_color) * 0.10
        bs = np.concatenate([np.ones(n_color // 2), np.zeros(n_color - n_color // 2)]) * 0.90
        alphas = np.abs(np.linspace(-1, 1, n_color))

        alphas[n_color // 2 - 1] = alphas[n_color // 2] = alphas[n_color // 2 + 1] = 1 / 256
        rs[n_color // 2 - 1] = rs[n_color // 2] = rs[n_color // 2 + 1] = 1 / 256
        gs[n_color // 2 - 1] = gs[n_color // 2] = gs[n_color // 2 + 1] = 1 / 256
        bs[n_color // 2 - 1] = bs[n_color // 2] = bs[n_color // 2 + 1] = 1 / 256
        colors = np.stack((rs, gs, bs, alphas), axis = 1)
        self.cm = Colormap(colors)
        
        if isinstance(data, GaussianCube):
            self.cube = data
            self.wavefunction = None
        elif isinstance(data, MoldenWavefunction):
            self.wavefunction = data
            self.cube = None
        elif isinstance(data, str):
            if data.endswith(".cub") or data.endswith(".cube"):
                self.cube = GaussianCube(data)
                self.wavefunction = None
            elif data.endswith("molden"):
                self.wavefunction = MoldenWavefunction(data)
                self.cube = None

    def set_molecule(self):
        a = self.cube if self.cube else self.wavefunction
        edgs = a.bondpts
        atoms = scene.visuals.Markers(pos=a.coords, size = 10)
        edges = scene.visuals.Line(edgs, color = (0, 1, 0, 1), width = 5, connect = "segments")
        texts = scene.visuals.Text(text = a.elems, pos = a.coords, method = "gpu")
        self.view.add(atoms)
        self.view.add(edges)
        self.view.add(texts)

    def build_initial_volume(self):
        cm = self.cm
        data = self.cube.data
        scaletp = (self.cube.dx, self.cube.dy, self.cube.dz)
        transtp = (self.cube.xmin, self.cube.ymin, self.cube.zmin)
        var = (data.max() - data.min()) / 2
        self.current_delta_var = var * 0.05
        self.volume = scene.visuals.Volume(data.transpose(2,1,0).copy(), interpolation="nearest", cmap = cm, method = "translucent", clim = (-var, var), threshold=0)
        self.volume.transform = scene.STTransform(scale = scaletp, translate=transtp)
        self.alphafilter = Alpha(0.8)
        self.volume.attach(self.alphafilter)
        self.volume.clipping_planes = np.array([[[self.cube.xmax, 0, 0], [-1, 0, 0]], [[0, self.cube.ymax, 0], [0, -1, 0]], [[0, 0, self.cube.zmax//2], [0, 0, -1]]])
        self.logger.log(f"Set volume clip to ({-var:1.3f}, {var:1.3f})")

    def modify_clipplane(self, ax=0, ay=0, az=0, dx=0, dy=0, dz=0):
        cp = self.clipplane
        cp[-1,-1] = np.array(rot(*self.clipplane[-1,-1], ax, ay, az), dtype = np.float32)
        nx, ny, nz = cp[-1,0][0]+dx, cp[-1,0][1]+dy, cp[-1,0][2]+dz
        nx = min(max(nx, self.cube.xmin), self.cube.xmax)
        ny = min(max(ny, self.cube.ymin), self.cube.ymax)
        nz = min(max(nz, self.cube.zmin), self.cube.zmax)
        cp[-1,0][0], cp[-1,0][1], cp[-1,0][2] = nx, ny, nz
        self.clipplane = cp
        self.volume.clipping_planes = cp
        self.isoplanefilter.clipping_planes = cp

    def build_initial_isosurface(self):
        data = self.cube.data
        scaletp = (self.cube.dx, self.cube.dy, self.cube.dz)
        transtp = (self.cube.xmin, self.cube.ymin, self.cube.zmin)
        var = (data.max() - data.min()) / 2
        self.current_isosurface = var / 3

        self.isosurface_1 = scene.visuals.Isosurface(data, level = var / 3, color = (1, 0.5, 0, 1))
        self.isosurface_2 = scene.visuals.Isosurface(data, level =-var / 3, color = (0, 0.5, 1, 1))
        self.isosurface_1.transform = scene.STTransform(scale = scaletp, translate=transtp)
        self.isosurface_2.transform = scene.STTransform(scale = scaletp, translate=transtp)

        self.alphafilter = Alpha(0.8)
        self.isosurface_1.attach(self.alphafilter)
        self.isosurface_2.attach(self.alphafilter)
        
        self.clipsurfacethetax = 0
        self.clipsurfacethetay = 0
        vx, vy, vz = np.array(rot(0, 0,-1, 0, 0, 0), dtype = np.float32)
        self.clipplane = np.array([[[self.cube.xmax,0,0], [-1,0,0]], [[0,self.cube.ymax,0], [0,-1,0]], [[0,0,self.cube.zmax], [vx, vy, vz]]])

        self.isoplanefilter = PlanesClipper(self.clipplane)
        self.isosurface_1.attach(self.isoplanefilter)
        self.isosurface_2.attach(self.isoplanefilter)
        self.logger.log(f"Set isosurface to ({self.current_isosurface:1.3f}, {self.current_isosurface:1.3f})")

    def set_density(self):
        self.build_initial_isosurface()
        self.build_initial_volume()
        if self.mode == "volume":
            self.view.add(self.volume)
        elif self.mode == "isosurface":
            self.view.add(self.isosurface_1)
            self.view.add(self.isosurface_2)
        self.logger.log(f"Current display mode: {self.mode}.")
    
    def switch_visual(self):
        if self.mode == "volume":
            self.volume.parent = None
            self.view.add(self.isosurface_1)
            self.view.add(self.isosurface_2)
            self.mode = "isosurface"
        elif self.mode == "isosurface":
            self.isosurface_1.parent = None
            self.isosurface_2.parent = None
            self.view.add(self.volume)
            self.mode = "volume"
        self.logger.log(f"Current display mode: {self.mode}.")

    def update_density(self):
        data = self.cube.data
        var = self.volume.clim[1]
        self.volume.set_data(data.transpose(2,1,0).copy(), clim = (-var, var))
        self.isosurface_1.set_data(data, color = self.isosurface_1.color)
        self.isosurface_2.set_data(data, color = self.isosurface_2.color)

    def update_clim(self, clim):
        _, var = clim
        self.volume.set_data(self.cube.data.transpose(2,1,0).copy(), clim = (-var, var))
        self.logger.log(f"Set volume clip to ({-var:1.3f}, {var:1.3f})")

    def update_isolevel(self, level):
        self.isosurface_1.level = level[0]
        self.isosurface_2.level = level[1]

    def gen_cube_from_wf(self, data:np.ndarray, minpos = (0, 0, 0), grid_size = (1, 1, 1)):
        if not self.cube:
            self.cube = GaussianCube()
            wf = self.wavefunction
            self.cube.elems = wf.elems
            self.cube.coords = wf.coords
            self.cube.bonds = wf.bonds
            self.cube.n_atom = len(wf.elems)
        self.cube.xmin, self.cube.ymin, self.cube.zmin = minpos
        self.cube.nx, self.cube.ny, self.cube.nz = data.shape
        self.cube.dx, self.cube.dy, self.cube.dz = grid_size
        self.cube.xmax, self.cube.ymax, self.cube.zmax = self.cube.xmin+self.cube.nx*self.cube.dx, self.cube.ymin+self.cube.ny*self.cube.dy, self.cube.zmin+self.cube.nz*self.cube.dz
        self.cube.data = data

    def get_orbital_grid(self, orbitalid):
        if not self.wavefunction:
            return
        grid_dims = self.grid_dims
        if self.current_orbital_idx != orbitalid:
            self.current_orbital_idx = orbitalid
        cs, cas, pos, pws = self.wavefunction.get_raveled_gtf(self.wavefunction.C[self.current_orbital_idx])
        atomidxs = self.wavefunction.get_raveled_atomidx()

        dist0 = 6
        a = self.cube if self.cube else self.wavefunction
        xmin, ymin, zmin = a.coords[:,0].min()-dist0, a.coords[:,1].min()-dist0, a.coords[:,2].min()-dist0
        xmax, ymax, zmax = a.coords[:,0].max()+dist0, a.coords[:,1].max()+dist0, a.coords[:,2].max()+dist0
        dx, dy, dz = (xmax - xmin) / grid_dims[0], (ymax - ymin) / grid_dims[1], (zmax - zmin) / grid_dims[2]
        self.logger.log(f"Orbital {orbitalid}. Energy = {self.wavefunction.energys[orbitalid]:>5.6f} Hartree, Occupy = {self.wavefunction.occupys[orbitalid]:>1.3f}")

        t = time.time()
        V1 = cuda.to_device(np.empty(grid_dims, dtype=np.float32))
        blockdim = (32, 32, 32)
        griddim = (4,4,4)
        cube_kernel_v2[blockdim, griddim](V1, xmin, ymin, zmin, dx, dy, dz, cs, cas, pos, pws, atomidxs)
        V1 = np.array(V1)
        self.logger.log(f"Time for cube file generation: {time.time()-t:.3f} second.")
        self.gen_cube_from_wf(V1, (xmin, ymin, zmin), (dx, dy, dz))

    def show_current_plane(self, ngrid = 256):
        p0 = self.clipplane[-1][0]
        pv = self.clipplane[-1][1]
        plen = min(self.cube.dx*self.cube.nx, self.cube.dy*self.cube.ny,self.cube.dz*self.cube.nz)
        V = self.cube.data
        xmin, ymin, zmin = self.cube.xmin, self.cube.ymin, self.cube.zmin
        dx, dy, dz = self.cube.dx, self.cube.dy, self.cube.dz
        mp = get_plane_values(p0, pv, plen, ngrid, V, xmin, ymin, zmin, dx, dy, dz)
        _, var = self.volume.clim
        var = var / 2.5

    
        plt.clf()
        x0 = np.linspace(-plen/2, plen/2, ngrid)
        levels = np.linspace(-var, var, 100)
        cnt = plt.contourf(*np.meshgrid(x0, x0), mp, levels, vmin = -var, vmax = var, cmap = "seismic")
        plt.colorbar(cnt)
        plt.pause(0.1)

    def on_key_press(self, event):
        a = event._key._names[0]
        if a == "W":
            self.modify_clipplane(0,0,0,0,0,0.1)
        elif a == "S":
            self.modify_clipplane(0,0,0,0,0,-0.1)
        elif a == "A":
            self.modify_clipplane(1/180*pi,0,0,0,0,0)
        elif a == "D":
            self.modify_clipplane(-1/180*pi,0,0,0,0,0)
        elif a == "Up" and self.wavefunction != None:
            orbitalidx = min(self.wavefunction.C.shape[1] - 1, self.current_orbital_idx + 1)
            self.get_orbital_grid(orbitalidx)
            self.update_density()
            self.canvas.update()
        elif a == "Down" and self.wavefunction != None:
            orbitalidx = max(0, self.current_orbital_idx - 1)
            self.get_orbital_grid(orbitalidx)
            self.update_density()
            self.canvas.update()
        elif a == "Left" and self.cube != None:
            if self.mode == "volume":
                _, var = self.volume.clim
                var = var * 0.78125
                self.update_clim((-var, var))
            elif self.mode == "isosurface":
                level1 = self.isosurface_1.level
                level2 = self.isosurface_2.level
                self.update_isolevel((level1*0.78125, level2*0.78125))
            self.canvas.update()
        elif a == "Right" and self.cube != None:
            if self.mode == "volume":
                _, var = self.volume.clim
                var = var * 1.28000
                self.update_clim((-var, var))
            elif self.mode == "isosurface":
                level1 = self.isosurface_1.level
                level2 = self.isosurface_2.level
                self.update_isolevel((level1*1.28000, level2*1.28000))
            self.canvas.update()
        elif (a == "Q" or a == "E") and self.cube != None:
            self.switch_visual()
            self.canvas.update()

        elif a == "P":
            self.show_current_plane()

    def run(self):
        self.canvas.app.run()


if __name__ == "__main__":
    filename1 = 'moldenfiles/1-benzene.molden'
    # filename1 = 'cubefiles/7-coronene-hh-tddft.cisdp01.cube'
    viewer = DensityViewer(filename1)

    # viewer.set_molecule()
    viewer.get_orbital_grid(10)
    viewer.set_density()
    viewer.run()