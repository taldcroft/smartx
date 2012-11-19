import glob
import numpy as np
import matplotlib.pyplot as plt
import asciitable

N_ROW_COL = 22
lines = []
for filename in glob.glob('reports/10+0_half_exemplar/21/act_fail-*.dat'):
    lines.extend(open(filename, 'r').readlines())

vals = asciitable.read(lines, Reader=asciitable.NoHeader, guess=False,
                       names=['n_fail', 'hpd', 'rmsd', 'a', 'b', 'i_fail'])
n_vals = len(vals)
i_fails_list = [np.array([int(strval) for strval in x.split(',')])
                for x in vals['i_fail']]

xr = (np.random.random(n_vals) - 0.5) * 1
yr = (np.random.random(n_vals) - 0.5) * 0.02
xvals = (vals['n_fail'] + xr) / 441. * 100.0
yvals = (vals['hpd'] + yr) / 0.61

plt.figure(1, figsize=(5, 3.5))
plt.clf()
plt.plot(xvals, yvals, ',', mec='b', mfc='b')
# plt.plot([0.0], [1.22], 'xb', mew=3, ms=10)

plt.xlim(-0.05, 5.0)
plt.ylim(-0.05, 4.0)
plt.grid()
plt.xlabel("Actuator failure rate (percent)")
plt.ylabel('HPD relative to zero-failure case')
plt.tight_layout()

plt.savefig('reports/10+0_half_exemplar/21/act_fail_manuf.png')
