import numpy as np
import calc_adj as ca
import matplotlib.pyplot as plt

if 'ifuncs_x' not in globals():
    ifuncs_ry = ca.load_ifuncs('RY', 'p', case='7+2')
    ifuncs_x = ca.load_ifuncs('X', 'p', case='7+2')

# displ_x = ca.load_displ_legendre(ifuncs_x, 8, 4, rms=5.0)
# displ_ry, _ = np.gradient(displ_x, 0.5)  # 0.5 mm
displ_x = ca.load_displ_grav(axis='X')
displ_ry = ca.load_displ_grav(axis='RY')
displ_rz = ca.load_displ_grav(axis='RZ')
displ_ry_calc, _ = np.gradient(displ_x, 0.5)  # 0.5 mm

if 'clip' not in globals():
    clip = 20

if 'n_ss' not in globals():
    n_ss = 5

# Optimize on X

print
print "** Optimize on X **"
print

coeffs_x, adj_x, M_2d_x, displ_x_clip = ca.calc_adj(
    ifuncs_x, displ_x, n_ss, clip)

# print results
# slope errors in az and ax from displ_x solution
resid_x = displ_x_clip - adj_x
print "Inputs"
print "Displ_x stddev, mean: {:.4f},{:.4f}".format(
    displ_x.std(), displ_x.mean())
print "Displ_ry stddev, mean: {:.4f},{:.4f}".format(
    displ_ry.std(), displ_ry.mean())
print "Displ_rz stddev, mean: {:.4f},{:.4f}".format(
    displ_rz.std(), displ_rz.mean())
print "Displ_ry_calc stddev, mean: {:.4f},{:.4f}".format(
    displ_ry_calc.std(), displ_ry_calc.mean())

print
print "Output residuals"
resid_x_ry, resid_x_rz = np.gradient(resid_x, 0.5)
print "Displ_x stddev, mean: {:.4f},{:.4f}".format(
    resid_x.std(), resid_x.mean())
print "Displ_x calculated RY (axial) stddev, mean: {:.4f},{:.4f}".format(
    resid_x_ry.std(), resid_x_ry.mean())
print "Displ_x calculated RZ (azimuthal) stddev, mean: {:.4f},{:.4f}".format(
    resid_x_rz.std(), resid_x_rz.mean())

# Optimize on RY

print
print "** Optimize on RY **"
print 

coeffs_ry, adj_ry, M_2d_ry, displ_ry_clip = ca.calc_adj(
    ifuncs_ry, displ_ry, n_ss, clip)

resid_ry = displ_ry_clip - adj_ry
print "Inputs"
print "Displ_ry stddev, mean: {:.4f},{:.4f}".format(
    displ_ry.std(), displ_ry.mean())
print
print "Output residuals"
print "Displ_ry stddev, mean: {:.4f},{:.4f}".format(
    resid_ry.std(), resid_ry.mean())
