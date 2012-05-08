import numpy as np
from numpy import pi, sin, cos, exp

RAD2ARCSEC = 206000.  # convert to arcsec for better scale

def calc_scatter(displ, dx=500, graze_angle=2.0, scale=np.sqrt(2), E=1.0,
                 n_theta=100,
                 ):

    lam = 1.24e-3 / E   # wavelength in microns for given energy in keV
    alpha = (90 - graze_angle) * pi / 180
    theta_max = lam / (2 * dx * (pi / 2 - alpha))
    print 'theta_max (arcsec)', theta_max * 206000.
    cos_alpha = cos(alpha)
    sin_alpha = sin(alpha)
    n_ax, n_az = displ.shape
    x = np.arange(n_ax) * dx
    x = x - x.mean()
    thetas = np.linspace(alpha - theta_max, alpha + theta_max, n_theta)
    I_scatter = []

    for theta in thetas:
        ampl_sum = 0.0
        d_sin = sin(alpha) - sin(theta)
        d_cos = cos(alpha) + cos(theta)

        for jj in range(n_az):
            y = displ[:, jj]
            ampl = np.sum(exp(2 * pi * 1j / lam *
                              (x * d_sin - y * d_cos * scale))) / n_ax
            ampl_sum += abs(ampl) ** 2
        I_scatter.append(ampl_sum)

    return (thetas - thetas.mean()) * RAD2ARCSEC, np.array(I_scatter)
