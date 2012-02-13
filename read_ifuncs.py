import numpy as np
import asciitable

n_fea_azim = 821
n_fea_axial = 411
n_if_azim = 10
n_if_axial = 10

out = {}
cols = 'X   Y   Z   RX  RY  RZ'.split()
for col in cols:
    out[col] = np.empty([n_if_azim, n_if_axial, n_fea_azim, n_fea_axial],
                        dtype=np.float64)

for mir in ('p', 's'):
    for i in range(n_if_azim):
        for j in range(n_if_axial):
            fname = '{}1000/ifuncs/Gen-X_{}-1000_1umPiezo_{},{}.dat'.format(
                mir, mir.upper(), j + 1, i + 1)
            print 'Reading', fname
            dat = asciitable.read(fname, guess=False, header_start=14,
                                  data_start=10, data_end=-5)
            print '  processing'
            for col in cols:
                vals = dat['Displa-{}'.format(col)]
                out[col][i, j] = vals.reshape(n_fea_azim, n_fea_axial)
            print '  done'

    for col in cols:
        np.save('{}1000/{}_ifuncs.npy'.format(mir, col), out[col])
