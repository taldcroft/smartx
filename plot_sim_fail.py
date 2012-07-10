import glob
import numpy as np
import matplotlib.pyplot as plt
import asciitable

N_ROW_COL = 22
lines = []
for filename in glob.glob('reports/10+0_half_exemplar/1/act_fail-*.dat'):
    lines.extend(open(filename, 'r').readlines())

vals = asciitable.read(lines, Reader=asciitable.NoHeader, guess=False,
                       names=['n_fail', 'hpd', 'rmsd', 'i_fail'])
n_vals = len(vals)
i_fails_list = [np.array([int(strval) for strval in x.split(',')])
                for x in vals['i_fail']]

n_edge = np.zeros(n_vals)
n_edge2 = np.zeros(n_vals)
nrc1 = N_ROW_COL - 1
for i, i_fails in enumerate(i_fails_list):
    row = i_fails / N_ROW_COL
    col = i_fails % N_ROW_COL
    edge = (row < 1) | (row > nrc1 - 1) | (col < 1) | (col > nrc1 - 1)
    edge2 = (row < 2) | (row > nrc1 - 2) | (col < 2) | (col > nrc1 - 2)
    n_edge[i] = np.sum(edge)
    n_edge2[i] = np.sum(edge2)

xr = (np.random.random(n_vals) - 0.5) * 1
yr = (np.random.random(n_vals) - 0.5) * 0.02
xvals = vals['n_fail'] + xr
yvals = vals['hpd'] + yr

plt.clf()
plt.plot(xvals, yvals, ',', mec='b', mfc='b')
plt.plot([0.0], [0.74], 'xb', mew=3, ms=10)

ok = n_edge >= np.sqrt(vals['n_fail'])
plt.plot(xvals[ok], yvals[ok], '.', mfc='r', mec='r',
         label='Number edge failures >= sqrt(n_fail)')
plt.xlim(-1, 22)
plt.grid()
plt.xlabel("Actuator failures")
plt.ylabel('HPD (arcsec)')
plt.title("HPD vs. actuator failures in manufacturing")
plt.legend(loc='best')

plt.savefig('reports/10+0_half_exemplar/1/act_fail_manuf.png')

