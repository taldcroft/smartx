import numpy as np
import asciitable

n_fea_azim = 821
n_fea_axial = 411

out = {}
cols = 'X   Y   Z   RX  RY  RZ'.split()
for col in cols:
    out[col] = np.empty([n_fea_axial, n_fea_azim], dtype=np.float64)

for mir in ('p', 's'):
    for displ in ('GRAV-Z',):
        fname = '{}1000/Gen-X_{}-1000_1umPiezo_{}.dat'.format(
            mir, mir.upper(), displ)
        print 'Reading', fname
        dat = asciitable.read(fname, guess=False, header_start=14,
                              data_start=10, data_end=-5)
        print '  processing'
        for col in cols:
            vals = dat['Displa-{}'.format(col)]
            out[col] = vals.reshape(n_fea_azim, n_fea_axial).transpose()
        print '  done'

        for col in cols:
            filename = '{}1000/{}_{}.npy'.format(mir, col, displ.lower())
            np.save(filename, out[col])
