"""Run calc_adj for the exemplar data
"""
import matplotlib.pyplot as plt

import calc_adj


ifuncs = calc_adj.load_ifuncs('X')
displ = calc_adj.load_file_legendre(
    ifuncs, slope=False, filename='data/exemplar_021312.dat')
cols = np.linspace(0, displ.shape[1], 10).astype(int)
cols = (cols[1:] + cols[:-1]) // 2
print 'cols =', cols

coeffs, adj_2d, M_2d_all, displ_clip = calc_adj.calc_adj(ifuncs, displ,
                                                         n_ss=5, clip=20)
calc_adj.make_plots(displ_clip, adj_2d)
plt.figure(1)
plt.savefig('exemplar_X.png')

resid = displ_clip - adj_2d
np.savetxt('exemplar_resid_X.dat', resid[::2, cols], fmt='%8.5f')

ifuncs = calc_adj.load_ifuncs('RY')
displ = calc_adj.load_file_legendre(
    ifuncs, slope=True, filename='data/exemplar_021312.dat')

coeffs, adj_2d, M_2d_all, displ_clip = calc_adj.calc_adj(ifuncs, displ,
                                                         n_ss=5, clip=20)
calc_adj.make_plots(displ_clip, adj_2d)
plt.figure(1)
plt.savefig('exemplar_RY.png')

resid = displ_clip - adj_2d
np.savetxt('exemplar_resid_RY.dat', resid[::2, cols], fmt='%9.6f')
