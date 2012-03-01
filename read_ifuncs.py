from glob import glob
import numpy as np
import asciitable

n_fea_azim = 821
n_fea_axial = 411
n_if_azim = 10
n_if_axial = 10

out = {}
cols = 'X   Y   Z   RX  RY  RZ'.split()
for col in cols:
    out[col] = np.empty([n_if_azim, n_if_axial, n_fea_axial, n_fea_azim],
                        dtype=np.float64)
colnames = ("Node  Displa-X   Displa-Y   Displa-Z   Displa-RX  "
            "Displa-RY  Displa-RZ".split())
for mir in ('p', 's'):
    for i in range(n_if_azim):
        for j in range(n_if_axial):
            fnames = glob('{}1000/ifuncs/*{}*1000*_{},{}.*'.format(
                    mir, mir.upper(), j + 1, i + 1))
            if len(fnames) != 1:
                raise ValueError('Got {}'.format(fnames))

            fname = fnames[0]
            print 'Reading', fname
            dat = asciitable.read(fname, guess=False,
                                  Reader=asciitable.NoHeader,
                                  names=colnames)
            print '  processing'
            for col in cols:
                vals = dat['Displa-{}'.format(col)]
                out[col][i, j] = vals.reshape(n_fea_azim,
                                              n_fea_axial).transpose()
            print '  done'

    for col in cols:
        np.save('{}1000/{}_ifuncs.npy'.format(mir, col), out[col])
