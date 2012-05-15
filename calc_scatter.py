import numpy as np
from numpy import pi, sin, cos, exp
from scipy.interpolate import interp1d

RAD2ARCSEC = 180. * 3600. / pi  # convert to arcsec for better scale

def calc_scatter(displ, dx=500, graze_angle=2.0, scale=np.sqrt(2), lam=1.24e-3,
                 thetas=None, n_theta=1000, theta_max=60, n_x=1000):
    """
    lam = 1.24e-3 um <=> 1 keV
    """
    alpha = (90 - graze_angle) * pi / 180
    # theta_max0 = lam / (2 * dx * (pi / 2 - alpha))
    theta_max /= RAD2ARCSEC
    print 'theta_max0 (arcsec)', theta_max * RAD2ARCSEC
    n_ax, n_az = displ.shape
    x = np.arange(n_ax) * dx
    x = x - x.mean()
    if thetas is None:
        thetas = np.linspace(alpha - theta_max, alpha + theta_max, n_theta)
    I_scatter = []

    print x
    print 'mean(x), std(x)', np.mean(x), np.std(x)
    interp = interp1d(x, displ, kind='cubic', axis=0)
    x_int = np.linspace(x[0], x[-1], n_x)
    displ_int = interp(x_int)

    for i_theta, theta in enumerate(thetas):
        ampl_sum = 0.0
        d_sin = sin(alpha) - sin(theta)
        d_cos = cos(alpha) + cos(theta)
        print 'd_sin', d_sin
        print 'd_cos', d_cos

        for jj in range(n_az):
            y = displ_int[:, jj]
            # print 'mean(y), std(y)', np.mean(y), np.std(y)
            ampl = np.sum(exp(2 * pi * 1j / lam *
                              (x_int * d_sin - y * d_cos * scale)))
            ampl_sum += abs(ampl) ** 2
        I_scatter.append(ampl_sum)

    return (thetas - thetas.mean()) * RAD2ARCSEC, np.array(I_scatter)
