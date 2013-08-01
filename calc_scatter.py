#!/usr/bin/env python

import numpy as np
from numpy import pi, sin, cos, exp, abs
from scipy.interpolate import interp1d

RAD2ARCSEC = 180. * 3600. / pi  # convert to arcsec for better scale


class CalcScatter(object):

    def __init__(self, alpha, x, y, scale, lam):
        self.alpha = alpha
        self.x = x
        self.y = y
        self.scale = scale
        self.lam = lam

    def __call__(self, theta):
        fac = 2 * pi * 1j / self.lam
        d_sin = fac * (sin(self.alpha) - sin(theta))
        d_cos = fac * (cos(self.alpha) + cos(theta)) * self.scale

        ampl = np.sum(exp(self.x * d_sin - self.y * d_cos),
                      axis=0)
        ampl_sum = np.sum(abs(ampl) ** 2)

        return ampl_sum


def calc_scatter(displ, dx=500, graze_angle=2.0, scale=np.sqrt(2), lam=1.24e-3,
                 thetas=None, d_theta=0.01, theta_max=20, n_x=2000, n_proc=4):
    """
    lam = 1.24e-3 um <=> 1 keV
    """
    alpha = (90 - graze_angle) * pi / 180
    theta_max /= RAD2ARCSEC
    d_theta /= RAD2ARCSEC
    n_ax, n_az = displ.shape
    x = np.arange(n_ax) * dx
    x = x - x.mean()
    if thetas is None:
        thetas = np.arange(alpha - theta_max, alpha + theta_max, d_theta)
    else:
        thetas = thetas + alpha
    I_scatter = []

    interp = interp1d(x, displ, kind='cubic', axis=0)
    x_int = np.linspace(x[0], x[-1], n_x)
    y = interp(x_int)
    x = x_int.reshape(-1, 1)

    calc_func = CalcScatter(alpha, x, y, scale, lam)
    if n_proc:
        from multiprocessing import Pool
        pool = Pool(processes=n_proc)
        I_scatter = pool.map(calc_func, list(thetas))
        pool.close()
    else:
        I_scatter = [calc_func(x) for x in thetas]

    I_scatter = np.array(I_scatter) / np.sum(I_scatter)

    return (thetas - thetas.mean()) * RAD2ARCSEC, np.array(I_scatter)


def calc_scatter_stats(theta, scatter):
    """Return stats about scatter distribution.
    """
    out = {}
    i_mid = len(theta) // 2
    i1 = 2 * i_mid - 1
    angle = theta[i_mid:i1]

    sym_scatter = scatter[i_mid:i1] + scatter[i_mid - 1:0:-1]
    sym_scatter /= sym_scatter.sum()

    ee = np.cumsum(sym_scatter)
    i_hpr = np.searchsorted(ee, 0.5)
    out['hpd'] = angle[i_hpr] * 2

    i99 = np.searchsorted(ee, 0.99)
    out['rmsd'] = 2 * np.sqrt(np.sum(angle ** 2 * sym_scatter)
                              / np.sum(sym_scatter))
    out['ee_angle'] = angle
    out['ee_val'] = ee
    out['ee_d50'] = out['hpd']
    out['ee_d90'] = angle[np.searchsorted(ee, 0.9)] * 2
    out['ee_d99'] = angle[i99] * 2
    return out  # angle_hpd, angle_rmsd, angle, ee


def main():
    """
    Command line script to computer scatter intensity curve.
    """
    import os
    import astropy.io.fits as pyfits
    from astropy.table import Table
    opt = get_opt()

    hdus = pyfits.open(opt.resid_file)
    displ = hdus[0].data
    hdus.close()

    print 'Read in FITS image {!r}'.format(opt.resid_file)
    print 'Image size: {} rows x {} cols'.format(displ.shape[0], displ.shape[1])
    print 'Image row 0 (cols 0..3) : {}'.format(displ[0][0:4])
    print
    theta_max = 2.55e-4
    thetas_in = np.linspace(-theta_max, theta_max, 10001)  # 10001

    # Compute the column position of axial strips
    cols = np.linspace(0, displ.shape[1], opt.n_strips + 1).astype(int)
    cols = (cols[1:] + cols[:-1]) // 2

    print 'Calculating scatter intensity using {} processors'.format(opt.n_proc)
    thetas, scatter = calc_scatter(displ[:, cols], dx=opt.dx, graze_angle=opt.graze_angle,
                                   lam=opt.lam, thetas=thetas_in, d_theta=opt.d_theta,
                                   n_x=opt.n_x, n_proc=opt.n_proc)

    outroot, ext = os.path.splitext(opt.resid_file)
    print 'Writing theta, scatter values to {}'.format(outroot + '.dat')
    dat = Table([thetas, scatter], names=['theta', 'scatter'])
    dat.write(outroot + '.dat', format='ascii')

    print 'Scatter statistics:'
    out = calc_scatter_stats(thetas, scatter)
    for key in ('ee_d50', 'ee_d90', 'ee_d99', 'hpd', 'rmsd'):
        print '  {} = {:.4f}'.format(key, out[key])


def get_opt():
    """
    dx=500, graze_angle=1.428, scale=np.sqrt(2), lam=1.24e-3,
    thetas=None, d_theta=0.01, theta_max=20, n_x=2000, n_proc=4
    """
    import argparse
    parser = argparse.ArgumentParser(description='Compute scatter intensity curve')
    parser.add_argument('--resid-file',
                        type=str,
                        help='Input residual image file')
    parser.add_argument('--graze-angle',
                        type=float,
                        default=1.428,
                        help='Graze angle (deg, default=1.428)')
    parser.add_argument('--dx',
                        type=float,
                        default=500,
                        help='Pixel size (microns, default=500)')
    parser.add_argument('--lam',
                        type=float,
                        default=1.24e-3,
                        help='Wavelength (microns, default=1.24e-3 == 1 keV)')
    parser.add_argument('--d-theta',
                        type=float,
                        default=0.01,
                        help='Theta bin size (arcsec, default=0.01)')
    parser.add_argument('--n-strips',
                        type=int,
                        default=20,
                        help='Number of axial strips (default=20)')
    parser.add_argument('--n-x',
                        type=int,
                        default=2000,
                        help='Number of points in axial direction (default=2000)')
    parser.add_argument('--n-proc',
                        type=int,
                        default=4,
                        help='Number of processors to use (default=4)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
