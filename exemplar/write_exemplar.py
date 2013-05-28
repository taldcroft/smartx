"""Run calc_adj for the exemplar data
"""
import numpy as np
import matplotlib.pyplot as plt

import calc_adj

RAD2ARCSEC = 206000.  # convert to arcsec for better scale
clip = 20
scale_ry = RAD2ARCSEC / 1000.  # ampl in microns, slope in arcsec

if 'ifuncs_x' not in globals() or 'ifuncs_ry' not in globals():
    ifuncs_x = calc_adj.load_ifuncs('X', case='10+0_half')
    ifuncs_ry = calc_adj.load_ifuncs('RY', case='10+0_half')

displ_x = calc_adj.load_file_legendre(
    ifuncs_x, slope=False, filename='data/exemplar_021312.dat')

out = open('exemplar_displ_X.dat', 'w')
for i in range(displ_x.shape[0]):
    for j in range(displ_x.shape[1]):
        print >>out, i, j, '%.4f' % displ_x[i, j]
out.close()
