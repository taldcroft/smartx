import os
import time
import numpy as np
from scipy.interpolate import griddata

root = 'data/10+0_5x10mm-0.2gap'

if 'in_grid' not in globals():
    print 'Reading grid'
    in_grid = np.loadtxt(os.path.join(root, 'P1000_Case1_nodes.txt'))

n_az = 821
n_ax = 411

az = np.radians(in_grid[:, 2]) * np.mean(in_grid[:, 1])
ax = in_grid[:, 3]

ok = np.abs(ax - np.min(ax)) < 0.002
print 'min ax', np.sum(ok)
ax0 = np.max(ax[ok]) + 1e-5

ok = np.abs(ax - np.max(ax)) < 0.002
print 'max ax', np.sum(ok)
ax1 = np.min(ax[ok]) - 1e-5

ok = np.abs(az - np.min(az)) < 0.002
print 'min az', np.sum(ok)
az0 = np.max(az[ok]) + 1e-5

ok = np.abs(az - np.max(az)) < 0.002
print 'max az', np.sum(ok)
az1 = np.min(az[ok]) - 1e-5

in_ax = ax.reshape(n_az, n_ax)
in_az = az.reshape(n_az, n_ax).transpose()

out_ax = np.ones([n_az, n_ax]) * np.linspace(ax0, ax1, n_ax)
out_az = np.ones([n_ax, n_az]) * np.linspace(az0, az1, n_az)

ins = np.vstack([in_ax.ravel(), in_az.ravel()]).transpose()
outs = np.vstack([out_ax.ravel(), out_az.ravel()]).transpose()

axes = ('RY',)
for axis in axes:
    infile = os.path.join(root, '{}_ifuncs_irreg.npy'.format(axis))
    print 'Reading', infile
    ifuncs = np.load(infile)
    ifuncs_out = np.empty_like(ifuncs)

    n_row, n_col = ifuncs.shape[:2]
    for row in range(n_row):
        for col in range(n_col):
            print 'Regridding', row, col, time.ctime()
            ifunc_in = ifuncs[row, col].ravel()
            ifunc_out = griddata(ins, ifunc_in, outs, method='linear')
            ifuncs_out[row, col] = ifunc_out.reshape(n_ax, n_az)
            tmp_outfile = os.path.join(root, '{}_ifuncs_{},{}.npy'.format(axis, row, col))
            np.save(tmp_outfile, ifunc_out)

    outfile = os.path.join(root, '{}_ifuncs.npy'.format(axis))
    np.save(outfile, ifuncs_out)
