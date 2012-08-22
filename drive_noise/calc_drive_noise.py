"""Run calc_adj for the exemplar data
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '..')
import calc_adj
import calc_scatter
import adj_opt_case

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
    save = 'exemplar_'
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

fig1 = plt.figure(1, figsize=(6, 8))
fig2 = plt.figure(2, figsize=(6, 8))
calc_adj.make_plots(displ_x, adj_x, fig1=fig1, fig2=fig2, clip=clip)
# save=save + corr_using + '_X')

plt.figure(5, figsize=(8, 5))
plt.subplot(1, 2, 1)
coeffs2d = coeffs.reshape(22, 22)
plt.imshow(coeffs2d, interpolation='nearest')
plt.colorbar(fraction=0.07)
plt.subplot(1, 2, 2)
plt.hist(coeffs, bins=20)

reference = 'median'  # or 'peak'
drive_ref = (np.median(abs(coeffs))
             if reference == 'median'
             else np.max(abs(coeffs)))

for noise in (0.00001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2):
    print 'Drive noise = {} (fraction of {} drive voltage)'.format(
        noise, reference)
    drive_noise = noise * drive_ref

    coeffs_noisy = coeffs + np.random.normal(0.0, scale=drive_noise,
                                             size=len(coeffs))

    adj_x = calc_adj.calc_adj_coeffs(ifuncs_x, coeffs_noisy)
    fig1 = plt.figure(3, figsize=(6, 8))
    fig2 = plt.figure(4, figsize=(6, 8))
    calc_adj.make_plots(displ_x, adj_x, fig1=fig1, fig2=fig2, clip=clip,
                        save='drive_noise_{}'.format(noise),
                        vmin_arg=-0.15, vmax_arg=0.15)
    resid_x = displ_x - adj_x
    thetas

#In [64]: calc_adj.make_plots(displ_x, adj_x, fig1=fig1, fig2=fig2, clip=clip)
#Input stddev, mean: 0.2538,-0.0406
#Resid stddev, mean: 0.2842,-0.0303
