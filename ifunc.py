import os
import numpy as np
import scipy.linalg
from scipy.special import legendre

RAD2ARCSEC = 206000.  # convert to arcsec for better scale


def get_slice(clip, axis_len, n_ss):
    if isinstance(clip, (tuple, list)):
        out_slice = slice(clip[0], clip[1], n_ss)
    else:
        out_slice = slice(clip, axis_len - clip, n_ss)
    return out_slice


def get_ifuncs_displ(case='10+0_baseline/'):
    ifuncs = load_ifuncs(case=case)
    n_rows, n_cols, n_ax, n_az = ifuncs.shape
    displ_x_all, displ_ry_all = load_displ_legendre(n_ax, n_az, offset_az=2)
    return ifuncs, displ_x_all


def clip_ifuncs_displ(ifuncs, displ_x_all, n_ss=5, ax_clip=50, az_clip=75,
                      row_slice=slice(None), col_slice=slice(None)):
    # Use only selected actuators and regenerate n_nows, n_cols
    ifuncs = ifuncs[row_slice, col_slice, :, :]
    n_rows, n_cols, n_ax, n_az = ifuncs.shape

    ax_slice = get_slice(ax_clip, n_ax, n_ss)
    az_slice = get_slice(az_clip, n_az, n_ss)

    i_ss, j_ss = np.mgrid[ax_slice, az_slice]
    ifuncs_clip_ss = ifuncs[:, :, i_ss, j_ss]
    M_3d_all = ifuncs.reshape(-1, n_ax, n_az)
    M_3d = M_3d_all[:, i_ss, j_ss]
    M_2d = M_3d.reshape(M_3d.shape[0], -1).transpose().copy()

    displ_x = displ_x_all[i_ss, j_ss].flatten().copy()

    return ifuncs_clip_ss, displ_x, M_2d


def get_ax_az_clip(clip):
    """
    Returns ax_clip, az_clip for either a scalar or tuple value of ``clip``.
    """
    if isinstance(clip, tuple):
        return clip
    else:
        return clip, clip


def get_ax_az_slice(clip):
    """
    Returns ax_slice, az_slice for either a scalar or tuple value of ``clip``.
    """
    ax_clip, az_clip = get_ax_az_clip(clip)
    ax_slice = slice(ax_clip, -ax_clip)
    az_slice = slice(az_clip, -az_clip)
    return ax_slice, az_slice


def calc_coeffs(ifuncs, displ, n_ss=10, clip=None, adj_clip=None):
    """Calculate the best (least-squared) set of coefficients to
    adjust for a displacement ``displ`` given influence functions
    ``ifuncs`` and sub-sampling ``n_ss``.  If ``clip`` is supplied
    then clip the specified number of pixels from each boundary.

    Returns
    -------
    coeffs: driving coefficients corresponding to ``ifuncs``
    """

    if adj_clip:
        ifuncs = ifuncs[adj_clip:-adj_clip, adj_clip:-adj_clip, :, :]

    # Clip boundaries
    if clip:
        ax_slice, az_slice = get_ax_az_slice(clip)
        displ = displ[ax_slice, az_slice]
        print 'SHAPE', ifuncs.shape
        ifuncs = ifuncs[..., ax_slice, az_slice]

    # Squash first two dimensions (20x20) of ifuncs into one (400)
    n_ax, n_az = ifuncs.shape[-2:]
    M_3d_all = ifuncs.reshape(-1, n_ax, n_az)

    # Sub-sample by n_ss along axial and aximuthal axes.  This uses
    # the numpy mgrid convenience routine:
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.mgrid.html
    if n_ss:
        i_ss, j_ss = np.mgrid[0:n_ax:n_ss, 0:n_az:n_ss]
        M_3d = M_3d_all[:, i_ss, j_ss]
        d_2d = displ[i_ss, j_ss]
    else:
        M_3d = M_3d_all
        d_2d = displ

    # Now reshape to final 2d matrix (e.g. 3486 rows x 400 cols for
    # n_ss = 10)
    M = M_3d.reshape(M_3d.shape[0], -1).transpose()

    # Flatten displacement to 1d
    d = d_2d.flatten()

    # Compute SVD and then the pseudo-inverse of M.
    # Note that .dot is the generalized array dot product and
    # in this case is matrix multiplication.
    U, s, Vh = scipy.linalg.svd(M, full_matrices=False)
    Minv = Vh.transpose() .dot (np.diag(1 / s)) .dot (U.transpose())

    # Finally compute the piezo driving coefficients
    coeffs = Minv .dot (d)

    return coeffs


def calc_adj_displ(ifuncs, coeffs, clip_adj=None):
    """Return the adjusted displacement as a 2-d array for the given ``ifuncs``
    and ``coeffs``.
    """
    n_ax, n_az = ifuncs.shape[-2:]
    M_3d_all = ifuncs.reshape(-1, n_ax, n_az)
    M_2d_all = M_3d_all.reshape(M_3d_all.shape[0], -1).transpose()
    if clip_adj:
        n_rows, n_cols = 0  # FAIL
        clip_coeffs_2d = coeffs.reshape(n_rows - 2 * clip_adj, n_cols - 2 * clip_adj)
        coeffs_2d = np.zeros((n_rows, n_cols), dtype=np.float)
        coeffs_2d[clip_adj:-clip_adj, clip_adj:-clip_adj] = clip_coeffs_2d
        coeffs = coeffs_2d.ravel()
    adj = M_2d_all.dot(coeffs)
    adj_2d = adj.reshape(n_ax, n_az)

    return adj_2d


def load_ifuncs(axis='X', case='10+2/p1000'):
    filename = 'data/{}/{}_ifuncs.npy'.format(case, axis)
    if10 = np.load(filename)
    nr, nc, n_ax, n_az = if10.shape
    nr2 = nr * 2
    nc2 = nc * 2
    if axis == 'RY':
        symmfac = -1
    elif axis in 'XYZ':
        symmfac = 1
        if10 = if10 * 1000  # convert linear displacement mm to microns
    else:
        raise ValueError('Which symmfac??')
    ifuncs = np.empty([nr2, nc2, n_ax, n_az])
    ifuncs[0:nr, 0:nc] = if10
    ifuncs[nr:nr2, 0:nc] = symmfac * if10[::-1, :, ::-1, :]
    ifuncs[0:nr, nc:nc2] = if10[:, ::-1, :, ::-1]
    ifuncs[nr:nr2, nc:nc2] = symmfac * if10[::-1, ::-1, ::-1, ::-1]
    return ifuncs


def slice_ifuncs(axis='X', case='10+0_baseline', nslice=3):
    filename = 'data/{}/{}_ifuncs.npy'.format(case, axis)
    if10 = np.load(filename)
    nr, nc, n_ax, n_az = if10.shape
    rows_per_act = n_ax // (nr * 2)  # only half the actuator ifuncs are stored
    cols_per_act = n_az // (nc * 2)
    ax_center = n_ax // 2
    az_center = n_az // 2
    ax0 = ax_center - nslice * rows_per_act
    ax1 = ax_center + nslice * rows_per_act
    az0 = az_center - nslice * cols_per_act
    az1 = az_center + nslice * cols_per_act
    out = if10[-nslice:, -nslice:, ax0:ax1, az0:az1]
    outdir = 'data/{}/{}x{}'.format(case, nslice, nslice)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    filename = '{}/{}_ifuncs.npy'.format(outdir, axis)
    np.save(filename, out)


def load_displ_grav(n_ax, n_az, case='10+2/p1000'):
    """"Return displ_x in mm for 0.5 mm spaced nodes, displ_ry in radians.
    """
    displ_x = np.load('data/{}/X_grav-z.npy'.format(case)) * 1000  # convert um
    displ_ry = np.load('data/{}/RY_grav-z.npy'.format(case))
    # displ_ry = np.gradient(displ_x, 0.5)[0]  # Radians

    return displ_x, displ_ry


def get_mount_map(n_ax, n_az):
    phase_az = np.linspace(0.0, 5 * 2 * np.pi, n_az)
    ampl_az = np.cos(phase_az)
    ampl_ax = np.zeros(n_ax, dtype=np.float)
    i_trans = int(n_ax * 0.15)
    phase_ax = np.linspace(0, np.pi, i_trans)
    ampl_ax[:i_trans] = (np.cos(phase_ax) + 1) / 2.0
    ampl_ax[-i_trans:] = ampl_ax[i_trans - 1::-1]

    mount_map = np.empty((n_ax, n_az), dtype=np.float)
    for i in range(n_ax):
        mount_map[i, :] = (1 + ampl_az * ampl_ax[i]) / (1 + ampl_ax[i])

    return mount_map


def load_file_legendre(n_ax, n_az, filename='data/exemplar_021312.dat',
                       apply_10_0=True):
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

    displ_x = Y_az_ax  # microns

    if apply_10_0:
        mount_map = get_mount_map(n_ax, n_az)
        displ_x = displ_x * mount_map

    # 0.5 mm spacing * 1000 um / mm
    displ_ry = np.gradient(displ_x, 0.5 * 1000)[0]  # radians

    return displ_x, displ_ry


def load_displ_legendre(n_ax, n_az, ord_ax=2, ord_az=0, offset_ax=1, offset_az=1, norm=1.0):
    x = np.linspace(-1, 1, n_az).reshape(1, n_az)
    y = np.linspace(-1, 1, n_ax).reshape(n_ax, 1)
    displ_x = norm * (offset_ax - legendre(ord_ax)(y)) * (offset_az - legendre(ord_az)(x))  # um
    displ_ry = np.gradient(displ_x, 0.5 * 1000)[0]  # radians

    return displ_x, displ_ry


def load_displ_10_0(n_ax, n_az, ord_ax=2, ord_az=1):
    pass
