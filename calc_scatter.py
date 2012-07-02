import numpy as np
from numpy import pi, sin, cos, exp, abs
from scipy.interpolate import interp1d
from multiprocessing import Pool

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
    pool = Pool(processes=n_proc)
    I_scatter = pool.map(calc_func, list(thetas))
    I_scatter /= np.sum(I_scatter)

    return (thetas - thetas.mean()) * RAD2ARCSEC, np.array(I_scatter)
