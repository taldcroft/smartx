"""Run calc_adj for the exemplar data
"""
import numpy as np
import matplotlib.pyplot as plt

import calc_adj

RAD2ARCSEC = 206000.  # convert to arcsec for better scale
clip = 20
scale_ry = RAD2ARCSEC / 1000.  # ampl in microns, slope in arcsec

if 'save' not in globals():
    save = None
if 'corr_using' not in globals():
    corr_using = 'x'

if 'ifuncs_x' not in globals() or 'ifuncs_ry' not in globals():
    ifuncs_x = calc_adj.load_ifuncs('X', case='10+0_half')
    ifuncs_ry = calc_adj.load_ifuncs('RY', case='10+0_half')

if 1:
    displ_x = calc_adj.load_file_legendre(
        ifuncs_x, slope=False, filename='data/exemplar_021312.dat')
    save = 'exemplar2_'
else:
    displ_x = calc_adj.load_displ_legendre(ifuncs_x, 8, 4, 0.5)
    save = 'leg84_'

# displ_ry = np.gradient(displ_x, 0.5 * 1000)[0] * RAD2ARCSEC
# radians (for exemplar down by factor of 1000)
displ_ry = np.gradient(displ_x, 0.5)[0]

if corr_using == 'x':
    # coeffs from optimizing on amplitude
    coeffs, adj_2d, M_2d_all, displ_clip = calc_adj.calc_adj(ifuncs_x, displ_x,
                                                             n_ss=5, clip=clip)
else:
    # coeffs from optimizing on slope
    coeffs, adj_2d, M_2d_all, displ_clip = calc_adj.calc_adj(ifuncs_ry,
                                                             displ_ry,
                                                             n_ss=5, clip=clip)


adj_x = calc_adj.calc_adj_coeffs(ifuncs_x, coeffs)
adj_ry = calc_adj.calc_adj_coeffs(ifuncs_ry, coeffs)
# adj_ry_dxdz = np.gradient(adj_x, 0.5)[0]

fig1 = plt.figure(1, figsize=(6, 8))
fig2 = plt.figure(2, figsize=(6, 8))
calc_adj.make_plots(displ_x, adj_x, fig1=fig1, fig2=fig2, clip=clip,
                    save=save + corr_using + '_X')

fig1 = plt.figure(3, figsize=(6, 8))
fig2 = plt.figure(4, figsize=(6, 8))
calc_adj.make_plots(displ_ry * scale_ry, adj_ry * scale_ry,
                    fig1=fig1, fig2=fig2, clip=clip,
                    save=save + corr_using + '_RY')

# fig1 = plt.figure(5, figsize=(6, 8))
# fig2 = plt.figure(6, figsize=(6, 8))
# calc_adj.make_plots(displ_ry, adj_ry_dxdz, fig1=fig1, fig2=fig2, clip=clip)

cols = np.linspace(0, displ_x.shape[1], 10).astype(int)
cols = (cols[1:] + cols[:-1]) // 2

if save:
    resid = displ_x - adj_x
    np.savetxt(save + corr_using + '_resid_X.dat', resid[::2, cols],
               fmt='%8.5f')

    resid = displ_ry - adj_ry
    np.savetxt(save + corr_using + '_resid_RY.dat', resid[::2, cols],
               fmt='%9.6f')

    np.savetxt(save + 'uncorr_X.dat', displ_x[::2, cols], fmt='%8.5f')
