
import pytest
from viewer import *

if 1 == 2:
    print("askdlfjskdlfjaklsdajfklsd")
    filename1 = r'C:\Users\fangn\Desktop\research\asphaltene\moldens\struct04_s.molden'
    filename2 = r'C:\Users\fangn\Desktop\research\asphaltene\outs\struct04_s.out'
    viewer = DensityViewer()
    viewer.read(filename1)
    viewer.excitations = read_cis_output(filename2)
    for ext in viewer.excitations:
        ext.filter_coefficients(threshold=0.05)
    viewer.initialize_scene()
    viewer.run()

if 1==1:
    viewer1 = DensityViewer()

    filename1 = r'C:\Users\fangn\Desktop\research\zwit-result-corrected\gly-blyp-80.0-cartesian.molden'
    # filename1 = r'shit.molden'
    viewer1.read(filename1)
    data1 = viewer1.get_electron_density()

    viewer2 = DensityViewer()
    filename2 = r'C:\Users\fangn\Desktop\research\zwit-result-corrected\gly-blyp-1.0-cartesian.molden'
    viewer2.read(filename2)
    data2 = viewer2.get_electron_density()

    viewer1.cube.data = data1 - data2

    print("aaa", np.sum(np.abs(data1 - data2)) / data1.size)

    viewer1.initialize_scene()
    viewer1.run()
    exit()

viewer1 = DensityViewer()

filename1 = r'C:\Users\fangn\Desktop\research\zwit-result-corrected\gly-blyp-1.0-cartesian.molden'
filename1 = r'shit.molden'
viewer1.read(filename1)
data1 = viewer1.get_electron_density()