import numpy as np
import scipy.linalg
from scipy.special import legendre
import matplotlib.pyplot as plt


RAD2ARCSEC = 206000.  # convert to arcsec for better scale


def calc_adj(ifuncs, displ, n_ss=10, clip=None):
    """Calculate the best (least-squared) set of coefficients to
    adjust for a displacement ``displ`` given influence functions
    ``ifuncs`` and sub-sampling ``n_ss``.  If ``clip`` is supplied
    then clip the specified number of pixels from each boundary.

    Returns
    -------
    coeffs: driving coefficients corresponding to ``ifuncs``
    adj: best adjustment (same size as ``displ``)
    """

    # Clip boundaries
    if clip:
        displ = displ[clip:-clip, clip:-clip]
        ifuncs = ifuncs[:, :, clip:-clip, clip:-clip]

    # Squash first two dimensions (20x20) of ifuncs into one (400)
    n_ax, n_az = ifuncs.shape[2:4]
    M_3d_all = ifuncs.reshape(-1, n_ax, n_az)
    M_2d_all = M_3d_all.reshape(M_3d_all.shape[0], -1).transpose()

    # Sub-sample by n_ss along axial and aximuthal axes.  This uses
    # the numpy mgrid convenience routine:
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.mgrid.html
    i_ss, j_ss = np.mgrid[0:n_ax:n_ss, 0:n_az:n_ss]
    M_3d = M_3d_all[:, i_ss, j_ss]

    # Now reshape to final 2d matrix (e.g. 3486 rows x 400 cols for
    # n_ss = 10)
    M = M_3d.reshape(M_3d.shape[0], -1).transpose()

    # Subsample displacement matrix and then flatten to 1d
    d_2d = displ[i_ss, j_ss]
    d = d_2d.flatten()

    # Compute SVD and then the pseudo-inverse of M.
    # Note that .dot is the generalized array dot product and
    # in this case is matrix multiplication.
    U, s, Vh = scipy.linalg.svd(M, full_matrices=False)
    Minv = Vh.transpose() .dot (np.diag(1 / s)) .dot (U.transpose())

    # Finally compute the piezo driving coefficients
    coeffs = Minv .dot (d)

    # Compute the actual adjustment (1d and 2d) given the coefficients
    adj = M_2d_all .dot (coeffs)
    adj_2d = adj.reshape(*displ.shape)

    return coeffs, adj_2d, M_2d_all, displ


def calc_adj_coeffs(ifuncs, coeffs):
    n_ax, n_az = ifuncs.shape[2:4]
    M_3d_all = ifuncs.reshape(-1, n_ax, n_az)
    M_2d_all = M_3d_all.reshape(M_3d_all.shape[0], -1).transpose()
    adj = M_2d_all.dot(coeffs)
    adj_2d = adj.reshape(n_ax, n_az)

    return adj_2d


def make_plots(displ, adj, clip=None, col0=150, col1=160,
               fig1=None, fig2=None,
               save=None, vmin_arg=None, vmax_arg=None):
    if clip:
        displ = displ[clip:-clip, clip:-clip]
        adj = adj[clip:-clip, clip:-clip]

    vals = np.hstack([displ, adj]).flatten()
    vals.sort()
    vmin = vals[int(len(vals) * 0.005)]
    vmax = vals[int(len(vals) * 0.995)]
    if fig1 is None:
        fig1 = plt.figure(1, figsize=(6, 8))
    plt.figure(fig1.number)
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.imshow(displ, vmin=vmin, vmax=vmax)
    plt.gca().axison = False
    plt.title('Input distortion')
    plt.colorbar(fraction=0.07)
    plt.subplot(3, 1, 2)
    plt.imshow(adj, vmin=vmin, vmax=vmax)
    plt.title('Best-fit adjustment')
    plt.gca().axison = False
    plt.colorbar(fraction=0.07)

    resid = displ - adj
    residf = np.sort(resid.flatten())
    vmin = vmin_arg or residf[int(len(residf) * 0.01)]
    vmax = vmax_arg or residf[int(len(residf) * 0.99)]
    plt.subplot(3, 1, 3)
    plt.title('Residual')
    plt.imshow(resid, vmin=vmin, vmax=vmax)
    plt.gca().axison = False
    plt.colorbar(fraction=0.07)
    if save:
        plt.savefig(save + '_map.png')

    if fig2 is None:
        fig2 = plt.figure(2)
    plt.figure(fig2.number)
    plt.clf()
    cols = slice(col0, col1)
    plt.plot(displ[:, cols].mean(axis=1) / 10., label='Input / 10')
    plt.plot(adj[:, cols].mean(axis=1) / 10., label='Adjust / 10')
    plt.plot(resid[:, cols].mean(axis=1), label='Resid')
    plt.title('Slice on mean of cols {}:{}'.format(col0, col1))
    plt.legend(loc='best')
    if save:
        plt.savefig(save + '_col.png')

    # Also show the RMS and mean
    print "Input stddev, mean: {:.4f},{:.4f}".format(displ.std(), displ.mean())
    print "Resid stddev, mean: {:.4f},{:.4f}".format(resid.std(), resid.mean())


def load_displ_grav(axis='RY', mirror='p', rms=None, case='7+2'):
    displ = np.load('data/{}/{}1000/{}_grav-z.npy'
                    .format(case, mirror, axis)) * RAD2ARCSEC
    if rms:
        displ = displ / np.std(displ) * rms

    return displ


def load_ifuncs(axis='RY', mirror='p', case='10+2'):
    filename = 'data/{}/{}1000/{}_ifuncs.npy'.format(case, mirror, axis)
    if10 = np.load(filename) * RAD2ARCSEC
    nr, nc, n_ax, n_az = if10.shape
    nr2 = nr * 2
    nc2 = nc * 2
    if axis == 'RY':
        symmfac = -1
    elif axis in 'XYZ':
        symmfac = 1
    else:
        raise ValueError('Which symmfac??')
    ifuncs = np.empty([nr2, nc2, n_ax, n_az])
    ifuncs[0:nr, 0:nc] = if10
    ifuncs[nr:nr2, 0:nc] = symmfac * if10[::-1, :, ::-1, :]
    ifuncs[0:nr, nc:nc2] = if10[:, ::-1, :, ::-1]
    ifuncs[nr:nr2, nc:nc2] = symmfac * if10[::-1, ::-1, ::-1, ::-1]
    return ifuncs


def load_file_legendre(ifuncs, filename='data/exemplar_021312.dat',
                       slope=False, rms=None):
    n_ax, n_az = ifuncs.shape[2:4]
    lines = (line.strip() for line in open(filename, 'rb')
             if not line.startswith('#'))
    D = np.array([[float(val) for val in line.split()]
                  for line in lines])
    nD_ax, nD_az = D.shape  # nD_m, nD_n

    x = np.linspace(-1, 1, n_az).reshape(1, n_az)
    y = np.linspace(-1, 1, n_ax).reshape(n_ax, 1)
    Pm_x = np.vstack([legendre(i)(x) for i in range(nD_ax)])
    Pn_y = np.hstack([legendre(i)(y) for i in range(nD_az)])
    Y_az_ax = np.zeros((n_ax, n_az), dtype=np.float)
    for n in range(nD_az):
        sum_Pm = np.zeros_like(x)
        for m in range(nD_ax):
            sum_Pm += D[m, n] * Pm_x[m, :]
        Y_az_ax += Pn_y[:, n].reshape(-1, 1) * sum_Pm

    # Unvectorized version for reference.
    #
    # xs = np.linspace(-1, 1, n_az)
    # ys = np.linspace(-1, 1, n_ax)
    # Pm_x = np.vstack([legendre(i)(xs) for i in range(nD_ax)])
    # Pn_y = np.vstack([legendre(i)(ys) for i in range(nD_az)])
    # Y_az_ax = np.zeros((n_ax, n_az), dtype=np.float)
    # for ix, x in enumerate(xs):
    #     for iy, y in enumerate(ys):
    #         for n in range(nD_az):
    #             sum_Pm = 0.0
    #             for m in range(nD_ax):
    #                 sum_Pm += D[m, n] * Pm_x[m, ix]
    #             Y_az_ax[iy, ix] += Pn_y[n, iy] * sum_Pm

    if slope:
        # 0.5 mm spacing * 1000 um / mm, then convert radians to arcsec
        displ = np.gradient(Y_az_ax, 0.5 * 1000)[0] * RAD2ARCSEC
    else:
        displ = Y_az_ax

    if rms:
        displ = displ / np.std(displ) * rms

    return displ


def load_displ_legendre(ifuncs, ord_ax=2, ord_az=0, rms=None):
    n_ax, n_az = ifuncs.shape[2:4]
    x = np.linspace(-1, 1, n_az).reshape(1, n_az)
    y = np.linspace(-1, 1, n_ax).reshape(n_ax, 1)
    displ = (1 - legendre(ord_ax)(y)) * (1 - legendre(ord_az)(x))
    if rms:
        displ = displ / np.std(displ) * rms

    return displ


def do_calc():
    # Some ugliness to initialize global vars so this can be used
    # interactively in IPython.
    global ifuncs, displ, clip, n_ss
    global coeffs, adj, M_2d
    if 'ifuncs' not in globals():
        ifuncs = load_ifuncs('RY', 'p')
    if 'displ' not in globals():
        displ = load_displ_legendre(ifuncs, 8, 4, rms=5.0)
        # OR displ = load_displ_grav('RY', 'p')
    if 'clip' not in globals():
        clip = 20
    if 'n_ss' not in globals():
        n_ss = 5

    coeffs, adj, M_2d, displ_clip = calc_adj(ifuncs, displ, n_ss, clip)
    make_plots(displ, adj, clip)
