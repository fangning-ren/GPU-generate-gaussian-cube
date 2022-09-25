#python

import numpy as np
import time

import vispy.plot as vplt
from vispy import scene
from vispy.visuals.filters import Alpha
from vispy.color import Colormap

from utils import *
from basis import *
from read_files import *
from convert_to_orca import *
from density import *


class EnergyViewer():
    def __init__(self, fig:vplt.PlotWidget):
        self.fig = fig
        self.fig._configure_2d()
        self.view = self.fig.view
        self.view:scene.ViewBox
        self.e = None
        self.elines = None
        self.eplot = None
        self.etext = None
        
        self.avg_de = 1.0
        self.dl = 1.0

        self.highlight_level = scene.visuals.Line(np.array([[0, 0.5],[0, 0]]), color = (1.0, 0.0, 0.0, 0.75), width = 10)
        self.currentidx = 0

    def plot_energy(self, energys):
        self.e = np.array(energys)
        self.e = self.e[self.e != 0.0]

        if self.e.shape[0] >= 20:
            n = self.e.shape[0]
            emid = self.e[int(n*0.25):int(n*0.75)]
            avg_de = (emid.max() - emid.min()) // emid.shape[0]
        else:
            avg_de = (self.e.max() - self.e.min()) // self.e.shape[0]
        degenerate_threshold = 5.0e-4
        dl = 0.05
        cur_xleft = 0.0

        xs = [[cur_xleft, cur_xleft+dl],]
        ys = [[self.e[0], self.e[0]],]
        connect = [[0, 1],]
        texts = [f"{self.e[0]:>.3f}",]
        for i in range(1, self.e.shape[0]):
            if abs(self.e[i] - self.e[i-1]) <= degenerate_threshold:
                cur_xleft += dl * 1.5
            else:
                cur_xleft = 0
            xs.append([cur_xleft, cur_xleft + dl])
            ys.append([self.e[i], self.e[i]])
            connect.append([i*2, i*2+1])
            texts.append(f"{self.e[i]:>.3f}")

            
        xs, ys, connect = np.array(xs), np.array(ys), np.array(connect)
        self.elines = np.stack((xs, ys), axis = -1)
        self.eplot = scene.visuals.LinePlot((xs.ravel(), ys.ravel()), color = "blue", symbol = "-", width = 2, connect = connect)
        self.etext = scene.visuals.Text(texts, color = "blue", pos = [(-dl*0.1, e) for e in self.e], anchor_x = "right", anchor_y = "center")

        self.view.add(self.eplot)
        self.view.add(self.etext)
        self.view.add(self.highlight_level)
        self.view.camera.set_range(x = (-dl*1.5, dl*3.2), y = (-avg_de, avg_de))

        self.avg_de = avg_de
        self.dl = dl

    def move_to_energy(self, eidx:int):
        self.highlight_level:scene.visuals.visuals.LineVisual
        self.highlight_level.set_data(self.elines[eidx])
        xl = self.elines[eidx][0][0]
        xr = self.elines[eidx][1][0]
        ys = self.elines[eidx][0][1]
        self.view.camera.center = (xl - self.dl*0.5, ys)
        self.currentidx = eidx

    def recover_everything(self):
        self.view.add(self.eplot)
        self.view.add(self.etext)
        self.view.add(self.highlight_level)
        self.view.camera.set_range(x = (-self.dl*1.5, self.dl*3.2), y = (-self.avg_de, self.avg_de))
        self.move_to_energy(self.currentidx)

    def detach_everything(self):
        self.eplot:scene.visuals.visuals.LinePlotVisual
        self.eplot.parent = None
        self.etext.parent = None
        self.highlight_level.parent = None

class SpectrumViewer():
    def __init__(self, fig:vplt.PlotWidget):
        self.fig = fig
        self.fig._configure_2d()
        self.view = self.fig.view
        self.view:scene.ViewBox

        self.uvspec = None
        self.uvpeak = None
        self.peaks = None
        self.highlight_peak = scene.visuals.Line(np.array([[0, 0.5],[0, 0]]), color = (1.0, 0.0, 0.0, 0.75), width = 2.5)
        self.currentidx = 0

        self.wwlstart = 200
        self.wwlend = 500

        self.camxrange = (self.wwlend - self.wwlstart) / 10
        self.camyrange = 10

    def _spectrum(self, es, oscs, sigma, x):
        y = np.zeros_like(x)
        for e, osc in zip(es, oscs):
            y += osc * np.exp(-(x - e)**2 / (sigma**2))
        return y

    def plot_spectrum(self, exts):
        xs_wl = np.linspace(self.wwlstart, self.wwlend, 500)
        xs_ev = 1239.8424121 / xs_wl
        es_ev = [ext.e*27.2113845 for ext in exts]
        oscs = [ext.osci for ext in exts]
        uvspec = self._spectrum(es_ev, oscs, 0.25 / 2.355, xs_ev)

        ptx, pty, connect, i = [], [], [], 0
        for e, osc in zip(es_ev, oscs):
            ptx.extend([1239.8424121/e, 1239.8424121/e])
            pty.extend([0, osc])
            connect.append((i*2, i*2+1))
            i += 1
        connect = np.array(connect)

        ptx, pty = np.array(ptx, dtype = np.float32), np.array(pty, dtype = np.float32)
        self.uvspec = scene.visuals.LinePlot((xs_wl, uvspec), color = "blue", symbol = "o", width = 2)
        self.uvpeak = scene.visuals.LinePlot((ptx, pty), color = "blue", width = 5, marker_size = 0, connect = connect)
        self.peaks = np.stack((ptx[1::2], pty[1::2]), axis = 1)

        self.camxrange = (self.wwlend - self.wwlstart) / 10
        self.camyrange = max(oscs)
        self.view.camera.set_range(x = (-self.camxrange, self.camxrange), y = (-self.camyrange, self.camyrange))

        self.view.add(self.uvspec)
        self.view.add(self.uvpeak)
        self.view.add(self.highlight_peak)

    def move_to_peak(self, pidx:int):
        xs = self.peaks[pidx][0]
        yl = 0
        yr = self.peaks[pidx][1]
        self.highlight_peak.set_data(np.array([[xs, yl],[xs, self.peaks[:,1].max()]]))
        self.view.camera.center = (xs, (yr - yl) / 2)
        self.currentidx = pidx

    def recover_everything(self):
        self.view.add(self.uvpeak)
        self.view.add(self.uvspec)
        self.view.add(self.highlight_peak)
        self.view.camera.set_range(x = (-self.camxrange, self.camxrange), y = (-self.camyrange, self.camyrange))
        self.move_to_peak(self.currentidx)

    def detach_everything(self):
        self.uvpeak.parent = None
        self.uvspec.parent = None
        self.highlight_peak.parent = None

class DensityViewer():
    """This is the frond end"""
    def __init__(self):
        self.logger = MyLogger()
        # self.canvas = scene.SceneCanvas(keys='interactive', size=(1280, 960), show=True)
        # self.canvasgrid = self.canvas.central_widget.add_grid()
        self.figure = vplt.Fig(bgcolor = "white", size = (1280, 960), show=False)
        self.figmain = self.figure[0:2,0:6]
        self.figclip = self.figure[0:1,6:8]
        self.figspec = self.figure[1:2,6:8]
        self.figmain:vplt.PlotWidget
        self.figclip:vplt.PlotWidget
        self.figspec:vplt.PlotWidget
        self.figmain._configure_3d()
        self.figclip._configure_2d()
        self.figspec._configure_2d()

        self.clipcolorbar = self.figclip.colorbar(
            position="right",
            clim= (-1, 1),
            cmap="seismic",
            border_width=0,
            border_color="#000000")

        self.energyviewer = EnergyViewer(self.figspec)
        self.uvspecviewer = SpectrumViewer(self.figspec)
        
        # self.figclipcolorbar:vplt.ColorBarWidget
        self.viewmain = self.figmain.view
        self.viewmain:scene.ViewBox
        self.viewmain.camera = 'turntable'
        self.viewmain.bgcolor = "#444444"
        self.viewclip = self.figclip.view
        self.viewclip.bgcolor = "#444444"

        self.figure.events["key_press"].connect(self.on_key_press)
        self.cm = self._transparent_seismic_colormap()

        self.current_orbital_idx = -1
        self.current_delta_var = -1
        self.current_isosurface = -1
        self.grid_dims = (128, 128, 128)
        self.grid_padding = 6.0
        self.volume = None
        self.isosurface_1 = None
        self.isosurface_2 = None
        self.mode = "orbital"

        self.wf = None      # loaded wave function. 
        self.cube = None    # Current cube data
        self.dm = None      # density manager object

        self.current_excitation_index = 0
        self.excitations = []
        
        self.keyops = {
            "W":self.move_plane_up,
            "S":self.move_plane_down,
            "A":self.rotate_plane_yminus,
            "D":self.rotate_plane_yplus,
            "Left": self.decrease_orbital,
            "Right": self.increase_orbital,
            "Up": self.increase_isovalue,
            "Down": self.decrease_isovalue,
            "Q": self.switch_to_orbital_viewer,
            "E": self.switch_to_excitation_viewer,
        }
        self.last_operation_time = time.time()

    def _transparent_seismic_colormap(self, n_color = 75):
        """The colormap for coloring the wavefunction volume"""
        rs = np.concatenate([np.zeros(n_color // 2), np.ones(n_color - n_color // 2)]) * 0.90
        gs = np.ones(n_color) * 0.10
        bs = np.concatenate([np.ones(n_color // 2), np.zeros(n_color - n_color // 2)]) * 0.90
        alphas = np.abs(np.linspace(-1, 1, n_color))
        alphas[n_color // 2 - 1] = alphas[n_color // 2] = alphas[n_color // 2 + 1] = 1 / 256
        rs[n_color // 2 - 1] = rs[n_color // 2] = rs[n_color // 2 + 1] = 1 / 256
        gs[n_color // 2 - 1] = gs[n_color // 2] = gs[n_color // 2 + 1] = 1 / 256
        bs[n_color // 2 - 1] = bs[n_color // 2] = bs[n_color // 2 + 1] = 1 / 256
        colors = np.stack((rs, gs, bs, alphas), axis = 1)
        return Colormap(colors)

    # Calculation functions
    def get_plane_data(self, ngrid = 512, clipplane = None):
        clipplane = self.clipplane if not isinstance(clipplane, np.ndarray) else clipplane
        p0, pv = self.clipplane[-1]
        plen = max(self.cube.dx*self.cube.nx, self.cube.dy*self.cube.ny,self.cube.dz*self.cube.nz)
        mp = get_plane_values(p0, pv, plen, ngrid, self.cube.data, *self.cube.minpos, *self.cube.grid_size)
        return mp

    def get_orbital_cube(self, orbidx = 0):
        """generate the cube data for a certain orbital. cuda kernel function required."""
        if not self.wf:
            return 
        grid_dims = self.grid_dims
        if self.current_orbital_idx != orbidx:
            self.current_orbital_idx = orbidx

        dist0 = self.grid_padding
        mol = self.wf.molecule
        xmin, ymin, zmin = mol.coords[:,0].min()-dist0, mol.coords[:,1].min()-dist0, mol.coords[:,2].min()-dist0
        xmax, ymax, zmax = mol.coords[:,0].max()+dist0, mol.coords[:,1].max()+dist0, mol.coords[:,2].max()+dist0
        dx, dy, dz = (xmax - xmin) / (grid_dims[0] - 1), (ymax - ymin) / (grid_dims[1] - 1), (zmax - zmin) / (grid_dims[2] - 1)
        V = np.zeros(grid_dims, dtype = np.float32)
        if not isinstance(self.cube, GaussianCube) or self.cube.shape != V.shape:
            del self.cube
            self.cube = GaussianCube(
                data = V,
                molecule = mol,
                grid_size=(dx, dy, dz),
                minpos = (xmin, ymin, zmin))
            if isinstance(self.dm, DensityManager):
                self.dm.cube = self.cube
        else:
            self.cube.data = V
        if not isinstance(self.dm, DensityManager):
            self.dm = DensityManager(self.wf, self.cube)

        self.logger.log(f"Orbital {orbidx}. Energy = {self.wf.energys[orbidx]:>5.6f} Hartree, Occupy = {self.wf.occupys[orbidx]:>1.3f}")
        t = time.time()
        self.cube.data = self.dm.get_orbital_value(orbidx)
        self.logger.log(f"Time for cube file generation: {time.time()-t:.3f} second.")
        return self.cube.data

    def get_electron_density(self):
        "Calculate the electron density"
        if not self.dm:
            self.logger.log("No wavefunction loaded. Cannot calculate electron density.")
            return 
        t = time.time()
        self.logger.log("Calculating electron density")
        V = self.dm.get_electron_density()
        a = np.sum(V)
        a *= self.cube.dx*self.cube.dy*self.cube.dz
        self.logger.log(f"Summing up all value and multiply differential element: {a}")
        self.logger.log(f"Time cost: {time.time()-t}")
        return V

    def get_transition_density(self, ext:Excitation = None):
        """Calculate the transition density"""
        if not self.dm:
            self.logger.log("No wavefunction loaded. Cannot calculate transition density.")
            return 
        if not ext:
            self.logger.log("No excitation loaded. Cannot calculate transition density.")
            return 
        if not ext.check_normalize():
            self.logger.log("warning: the square norm of this cis vector is smaller than 0.95, result may be inaccurate.")
        t = time.time()
        self.logger.log("Calculating transition density")
        V = self.dm.get_transition_density(ext=ext)
        self.logger.log(f"Excitation {self.current_excitation_index}. ΔE = {ext.e:1.3f}, fosc = {ext.osci:1.3f}, |T| = {ext.T2:1.3f}")
        self.logger.log(f"Time cost: {time.time()-t}")
        return V

    def get_difference_density(self, ext:Excitation = None):
        """Calculate the density difference between ground state and excited state, also known as the particle-hole density"""
        if not self.dm:
            self.logger.log("No wavefunction loaded. Cannot calculate transition difference density.")
            return 
        if not ext:
            self.logger.log("No excitation loaded. Cannot calculate transition difference density.")
            return 
        if not ext.check_normalize():
            self.logger.log("warning: the square norm of this cis vector is smaller than 0.95, result may be inaccurate.")
        t = time.time()
        self.logger.log("Calculating difference density")
        V = self.dm.get_difference_density(ext=ext)
        self.logger.log(f"Excitation {self.current_excitation_index}. ΔE = {ext.e:1.3f}, fosc = {ext.osci:1.3f}, |T| = {ext.T2:1.3f}")
        self.logger.log(f"Time cost: {time.time()-t}")
        return V

    def get_hole(self, ext:Excitation = None):
        """Calculate the hole distribution"""
        if not self.dm:
            self.logger.log("No wavefunction loaded. Cannot calculate.")
            return 
        if not ext:
            self.logger.log("No excitation loaded. Cannot calculate.")
            return 
        if not ext.check_normalize():
            self.logger.log("warning: the square norm of this cis vector is smaller than 0.95, result may be inaccurate.")
        t = time.time()
        self.logger.log("Calculating hole distribution")
        V = self.dm.get_hole(ext=ext)
        self.logger.log(f"Excitation {self.current_excitation_index}. ΔE = {ext.e:1.3f}, fosc = {ext.osci:1.3f}, |T| = {ext.T2:1.3f}")
        self.logger.log(f"Time cost: {time.time()-t}")
        return -V

    def get_electron(self, ext:Excitation = None):
        """Calculate the electron distribution"""
        if not self.dm:
            self.logger.log("No wavefunction loaded. Cannot calculate.")
            return 
        if not ext:
            self.logger.log("No excitation loaded. Cannot calculate.")
            return 
        if not ext.check_normalize():
            self.logger.log("warning: the square norm of this cis vector is smaller than 0.95, result may be inaccurate.")
        t = time.time()
        self.logger.log("Calculating electron distribution")
        V = self.dm.get_electron(ext=ext)
        self.logger.log(f"Excitation {self.current_excitation_index}. ΔE = {ext.e:1.3f}, fosc = {ext.osci:1.3f}, |T| = {ext.T2:1.3f}")
        self.logger.log(f"Time cost: {time.time()-t}")
        return +V

    def get_hole_and_electron(self, ext:Excitation):
        """This function is mainly used for adjusting visulization patterns"""
        V1 = self.get_hole(ext)
        V2 = self.get_electron(ext)
        Vid = np.random.randint(2, size = V1.shape, dtype = np.uint8)
        V1[Vid==1] = V2[Vid==1]
        return V1

    # Initialization functions       
    def read(self, data):
        if isinstance(data, GaussianCube):
            self.cube = data
            self.wf = None
        elif isinstance(data, np.ndarray) and data.ndim == 3:
            self.cube = GaussianCube(data = data)
            self.wf = None
        elif isinstance(data, MoldenWavefunction):
            self.wf = data
            self.get_orbital_cube(self.wf.homo)
        elif isinstance(data, str) and (data.endswith(".cube") or data.endswith(".cub")):
            self.cube = GaussianCube(fp = data)
            self.wf = None
        elif isinstance(data, str) and (data.endswith(".molden") or data.endswith(".molden.input")):
            self.wf = MoldenWavefunction(data)
            self.get_orbital_cube(0)
        else:
            raise ValueError("No corrected data loaded. Expect cube file or molden file.")
        

    def build_initial_isosurface(self):
        """build the isosurface visual before add it to the scene"""
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
        self.logger.log(f"Set isosurface to ({self.current_isosurface:1.3f}, {self.current_isosurface:1.3f})")

    def build_initial_volume(self):
        """build the volume visual before add it to the scene"""
        cm = self.cm
        data = self.cube.data
        scaletp = (self.cube.dx, self.cube.dy, self.cube.dz)
        transtp = (self.cube.xmin, self.cube.ymin, self.cube.zmin)
        var = max(1e-4, (data.max() - data.min()) / 2)
        self.current_delta_var = var * 0.05
        var = var * 0.25
        self.volume = scene.visuals.Volume(data.transpose(2,1,0).copy(), interpolation="linear", cmap = cm, method = "translucent", clim = (-var, var), threshold=0)
        self.volume.transform = scene.STTransform(scale = scaletp, translate=transtp)
        self.alphafilter = Alpha(0.8)
        self.volume.attach(self.alphafilter)
        self.volume.clipping_planes = self.clipplane
        self.logger.log(f"Set volume clip to ({-var:1.3f}, {var:1.3f})")

    def draw_initial_clip_image(self):
        plen = min(self.cube.dx*self.cube.nx, self.cube.dy*self.cube.ny,self.cube.dz*self.cube.nz)
        mp = self.get_plane_data()
        _, var = self.volume.clim
        var /= 2.5
        self.clipimage = scene.visuals.Image(mp, cmap = "seismic", clim = (-var, var))
        nx, ny = mp.shape
        self.viewclip.add(self.clipimage)
        self.figclip.view.camera.aspect = 1
        self.figclip.view.camera.set_range()

    def draw_initial_energy_level(self):
        if self.mode == "orbital":
            self.energyviewer.plot_energy(self.wf.energys)
            self.energyviewer.move_to_energy(self.current_orbital_idx)
    
    def draw_initial_spectrum(self):
        if len(self.excitations) > 0:
            self.uvspecviewer.plot_spectrum(self.excitations)
            self.uvspecviewer.move_to_peak(self.current_excitation_index)
  
    def draw_molecule(self):
        """draw the molecule"""
        a = self.cube.molecule if self.cube else self.wf.molecule
        edgs = a.bondpts
        if edgs.shape[-1] not in [2, 3]:
            edgs = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype = np.float32)
        
        edges = scene.visuals.Line(edgs, color = (0, 1, 0, 1), width = 5, connect = "segments")
        atoms = scene.visuals.Markers(pos=a.coords, size = 10)
        colors = [elemcolors[c] if c in elemcolors else (0.5, 0.5, 0.0, 1.0) for c in a.elems]
        texts = scene.visuals.Text(text = a.elems, pos = a.coords, color = colors, method = "gpu", font_size=500)
        self.viewmain.add(atoms)
        self.viewmain.add(edges)
        self.viewmain.add(texts)

    def draw_density(self):
        self.viewmain.add(self.volume)
        
    def initialize_scene(self):
        self.build_initial_isosurface()
        self.build_initial_volume()
        self.draw_initial_clip_image()
        self.draw_initial_energy_level()
        self.draw_molecule()
        self.draw_density()
        self.logger.log("Scene initialized!")

    def update_clip_image(self):
        mp = self.get_plane_data()
        _, var = self.volume.clim
        var /= 2.5
        self.clipimage.set_data(mp)
        self.clipimage.clim = (-var, var)
        self.clipcolorbar:vplt.ColorBarWidget
        self.clipcolorbar.clim = (-var, var)
        self.figure.update()

    # Interactive functions
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

    def move_plane_up(self):
        self.modify_clipplane(0,0,0,0,0,0.1)

    def move_plane_down(self):
        self.modify_clipplane(0,0,0,0,0,-0.1)

    def rotate_plane_yplus(self):
        self.modify_clipplane(1/180*pi,0,0,0,0,0)

    def rotate_plane_yminus(self):
        self.modify_clipplane(-1/180*pi,0,0,0,0,0)

    def increase_isovalue(self):
        _, var = self.volume.clim
        self.volume.set_data(self.cube.data.transpose(2,1,0).copy(), clim = (-var*1.28, var*1.28))
        self.logger.log(f"Set volume clip to ({-var:1.3f}, {var:1.3f})")
        self.isosurface_1.level = self.isosurface_1.level*1.28000
        self.isosurface_2.level = self.isosurface_2.level*1.28000
        self.logger.log(f"Set isosurface to ({-self.isosurface_1.level:1.3f}, {self.isosurface_2.level:1.3f})")
        self.figure.update()

    def decrease_isovalue(self):
        _, var = self.volume.clim
        self.volume.set_data(self.cube.data.transpose(2,1,0).copy(), clim = (-var*0.78125, var*0.78125))
        self.logger.log(f"Set volume clip to ({-var:1.3f}, {var:1.3f})")
        self.isosurface_1.level = self.isosurface_1.level*0.78125
        self.isosurface_2.level = self.isosurface_2.level*0.78125
        self.logger.log(f"Set isosurface to ({-self.isosurface_1.level:1.3f}, {self.isosurface_2.level:1.3f})")
        self.figure.update()

    def switch_visual(self):
        return 
        if self.mode == "volume":
            self.volume.parent = None
            self.viewmain.add(self.isosurface_1)
            self.viewmain.add(self.isosurface_2)
            self.mode = "isosurface"
        elif self.mode == "isosurface":
            self.isosurface_1.parent = None
            self.isosurface_2.parent = None
            self.viewmain.add(self.volume)
            self.mode = "volume"
        self.logger.log(f"Current display mode: {self.mode}")

    def increase_orbital(self):
        if self.mode == "orbital":
            self.current_orbital_idx = min(self.wf.C.shape[1] - 1, self.current_orbital_idx + 1)
            self.cube.data = self.get_orbital_cube(self.current_orbital_idx)
            self.energyviewer.move_to_energy(self.current_orbital_idx)
        elif self.mode == "excitation":
            self.current_excitation_index = min(len(self.excitations)-1, self.current_excitation_index + 1)
            self.cube.data = self.get_difference_density(self.excitations[self.current_excitation_index])
            self.uvspecviewer.move_to_peak(self.current_excitation_index)
        var = self.volume.clim[1]
        self.volume.set_data(self.cube.data.transpose(2,1,0).copy(), clim = (-var, var))
        self.isosurface_1.set_data(self.cube.data, color = self.isosurface_1.color)
        self.isosurface_2.set_data(self.cube.data, color = self.isosurface_2.color)
        self.figure.update()

    def decrease_orbital(self):
        if self.mode == "orbital":
            self.current_orbital_idx = max(0, self.current_orbital_idx - 1)
            self.cube.data = self.get_orbital_cube(self.current_orbital_idx)
            self.energyviewer.move_to_energy(self.current_orbital_idx)
        elif self.mode == "excitation":
            self.current_excitation_index = max(0, self.current_excitation_index - 1)
            self.cube.data = self.get_difference_density(self.excitations[self.current_excitation_index])
            self.uvspecviewer.move_to_peak(self.current_excitation_index)
        var = self.volume.clim[1]
        self.volume.set_data(self.cube.data.transpose(2,1,0).copy(), clim = (-var, var))
        self.isosurface_1.set_data(self.cube.data, color = self.isosurface_1.color)
        self.isosurface_2.set_data(self.cube.data, color = self.isosurface_2.color)
        self.figure.update()

    def switch_to_excitation_viewer(self):
        if len(self.excitations) == 0:
            self.logger.log("No excitations loaded!")
            return
        self.mode = "excitation"
        self.cube.data = self.get_difference_density(self.excitations[self.current_excitation_index])
        var = max(abs(self.cube.data.min()), self.cube.data.max())
        self.volume.set_data(self.cube.data.transpose(2,1,0), clim = (-var, var))
        self.energyviewer.detach_everything()
        if self.uvspecviewer.uvspec == None:
            self.uvspecviewer.plot_spectrum(self.excitations)
        self.uvspecviewer.recover_everything()

    def switch_to_orbital_viewer(self):
        self.mode = "orbital"
        self.cube.data = self.get_orbital_cube(self.current_orbital_idx)
        var = max(abs(self.cube.data.min()), self.cube.data.max())
        self.volume.set_data(self.cube.data.transpose(2,1,0), clim = (-var, var))
        self.uvspecviewer.detach_everything()
        self.energyviewer.recover_everything()

    def on_key_press(self, event):
        s = event._key._names[0]
        if s == "L":
            oid = self.current_orbital_idx
            self.substract(oid, GaussianCube(f"cubefiles\\orb0000{str(oid+1).zfill(2)}.cub"))
        if s not in self.keyops:
            return 
        self.keyops[s]()
        t = time.time()
        if t - self.last_operation_time >= 0.25 or s in ["Left", "Right"]:
            self.update_clip_image()
            self.last_operation_time = t

    def run(self):
        self.figure.show(visible=True, run = True)

    def substract(self, orbidx, c2:GaussianCube):
        
        self.get_orbital_cube(orbidx)
        a = self.cube.dx * self.cube.dy * self.cube.dz
        b = c2.dx * c2.dy * c2.dz

        A = self.cube.data
        B = c2.data
        print(np.sum(A)*a, np.sum(B)*b)


        dV = A - B
        var = self.volume.clim[1]
        self.volume.set_data(dV.transpose(2,1,0).copy(), clim = (-var, var))
        self.figure.update()

