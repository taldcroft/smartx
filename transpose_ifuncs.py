import numpy as np
import shutil


cols = 'X   Y   Z   RX  RY  RZ'.split()
out = np.empty([10, 10, 411, 821], dtype=np.float64)

for mir in ('p', 's'):
    for col in cols:
        name = '{}1000/{}_ifuncs.npy'.format(mir, col)
        print name
        # shutil.copy2(name, name + '.bak')
        dat = np.load(name + '.bak')
        for i in range(dat.shape[0]):
            for j in range(dat.shape[1]):
                out[i, j] = dat[i, j].transpose()
        np.save(name, out)
