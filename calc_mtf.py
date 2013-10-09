"""
Compute "modulation transfer function" in a fashion.
"""

import os
import shelve
import logging

import ifunc
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

RAD2ARCSEC = 206000.
N_AX = N_AZ = None  # For pyflakes

if 'CASE' not in globals():
    CASE = '10+0_baseline'

if 'AXIS' not in globals():
    AXIS = 'X'


def load_ifuncs():
    global ifuncs
    global displ
    global N_AX
    global N_AZ
    print 'Loading influence functions', CASE, 'for axis', AXIS
    ifuncs, displ = ifunc.get_ifuncs_displ(case=CASE, axis=AXIS)
    if AXIS == 'RY':
        ifuncs *= RAD2ARCSEC
    N_AX, N_AZ = ifuncs.shape[-2:]


if 'ifuncs' not in globals():
    load_ifuncs()


class Displ(object):
    pass


def displ_exemplar(ampl=1.0, apply_10_0=True):
    #def load_file_legendre(n_ax, n_az, filename='data/exemplar_021312.dat',
    #                   apply_10_0=True):
    out = Displ()
    displ_x, displ_ry = ifunc.load_file_legendre(N_AX, N_AZ, apply_10_0=apply_10_0)
    if AXIS == 'X':
        out.vals = displ_x * ampl
    else:
        out.vals = displ_ry * ampl * RAD2ARCSEC  # convert displ from radians to arcsec
    out.title = 'Displ-{} exemplar data (ampl={:.2f})'.format(AXIS, ampl)
    return out


def displ_sin_ax(n_cycle=1, ampl=0.5, phase=0.0):
    """
    Sinusoidal oscillation in axial direction.  Phase is specified
    in terms of cycles, so phase=0.25 corresponds to a cosine.

    :param n_cycle: Number of cycles (default = 1)
    :param ampl: Sinusoidal amplitude (microns)
    :param phase: Phase offset in units of cycles

    :returns: N_AX x N_AZ array
    """
    x = (np.arange(N_AX, dtype=float) / N_AX + phase) * 2 * np.pi * n_cycle
    x = np.repeat(x, N_AZ).reshape(-1, N_AZ)
    out = Displ()
    out.vals = np.sin(x) * ampl / n_cycle
    out.title = 'Displ-X (ampl={:.2f} n_cycle={:.1f} phase={})'.format(ampl, n_cycle, phase)
    return out


def displ_sin_az(n_cycle=1, ampl=0.5, phase=0.0):
    """
    Sinusoidal oscillation in azimuthal direction.  Phase is specified
    in terms of cycles, so phase=0.25 corresponds to a cosine.

    :param n_cycle: Number of cycles (default = 1)
    :param ampl: Sinusoidal amplitude (microns)
    :param phase: Phase offset in units of cycles

    :returns: N_AX x N_AZ array
    """
    x = (np.arange(N_AZ, dtype=float) / N_AZ + phase) * 2 * np.pi * n_cycle
    x = np.repeat(x, N_AX).reshape(-1, N_AX).transpose()
    out = Displ()
    out.vals = np.sin(x) * ampl / n_cycle
    out.title = 'Displ-X (ampl={:.2f} n_cycle={:.1f} phase={})'.format(ampl, n_cycle, phase)
    return np.sin(x) * ampl / n_cycle


def displ_flat(ampl=1.0):
    """
    Flat displacement

    :param ampl: Flat displacement value (microns)
    :returns: N_AX x N_AZ array
    """
    out = Displ()
    out.vals = np.ones((N_AX, N_AZ)) * ampl
    out.title = 'Bias flat (ampl={})'.format(ampl)
    return out


def displ_uniform_coeffs(ampl=1.0):
    """
    Response for all actuator coefficients set to a uniform value.

    An ``ampl`` of 1.0 corresponds to a value of 1 micron in a rectangle
    covering the central 50% (in each axis) of the mirror.

    :param ampl: Median response in center 50% of mirror
    :returns: N_AX x N_AZ array
    """
    ifuncs_3d = ifuncs.reshape(-1, N_AX, N_AZ)
    out = Displ()
    out.vals = np.sum(ifuncs_3d, axis=0)
    # Renormalize based on the median in the center 20% portion
    i0 = int(N_AX * 0.25)
    i1 = int(N_AX * 0.75)
    j0 = int(N_AZ * 0.25)
    j1 = int(N_AZ * 0.75)
    median_val = np.median(out.vals[i0:i1, j0:j1])
    out.vals *= ampl / median_val
    out.title = 'Bias uniform coefficients (ampl={})'.format(ampl)
    return out


def calc_plot_adj(row_clip=2, col_clip=2,
                  ax_clip=None, az_clip=None,
                  bias=None, error=None,
                  plot_file=None,
                  max_iter=0,
                  nnls=False):
    """
    Calculate and plot the displacement, residuals, and coeffs for
    an input displacement function.

    Example::

      >>> error = displ_sin_ax(n_cycle=1.0, ampl=0.5, phase=0.0)
      >>> bias = displ_flat(1.0)
      >>> out = calc_plot_adj(row_clip=2, col_clip=2, ax_clip=75, az_clip=150,
                              bias=bias, error=error, plot_file=None)

    :param row_clip: Number of actuator rows near edge to ignore
    :param col_clip: Number of actuator columns near edge to ignore
    :param ax_clip: Number of pixels near edge to clip in axial direction
    :param az_clip: Number of pixels near edge to clip in azimuthal direction
    :param bias: N_AX x N_AZ image of input bias
    :param error: N_AX x N_AZ image of input figure error
    :param max_iter: Maximum iterations for removing negative coefficients (default=0)
    :param plot_file: plot file name (default=None for no saved file)
    :param nnls: Use SciPy non-negative least squares solver instead of SVD

    :returns: dict of results
    """
    row_slice = slice(row_clip, -row_clip) if row_clip else slice(None, None)
    col_slice = slice(col_clip, -col_clip) if col_clip else slice(None, None)
    ax_slice = slice(ax_clip, -ax_clip)
    az_slice = slice(az_clip, -az_clip)

    # Get stddev of input displacement within clipped region
    displ = bias.vals + error.vals
    input_stddev = np.std(error.vals[ax_slice, az_slice])

    # Get ifuncs and displ that are clipped and sub-sampled (at default of n_ss=5)
    ifuncs_clip_4d, displ_clip, M_2d = ifunc.clip_ifuncs_displ(
        ifuncs, displ,
        row_slice=row_slice, col_slice=col_slice,
        ax_clip=ax_clip, az_clip=az_clip)

    ifuncs_clip = ifuncs_clip_4d.reshape(-1, *(ifuncs_clip_4d.shape[-2:]))
    coeffs_clip = np.zeros(len(ifuncs_clip))
    coeffs = ifunc.calc_coeffs(ifuncs_clip, displ_clip, n_ss=None, clip=0, nnls=nnls)
    pos_vals = np.ones(len(coeffs), dtype=bool)
    for ii in range(max_iter):
        pos = coeffs >= 0
        pos_idxs = np.flatnonzero(pos)
        if len(pos_idxs) == len(coeffs):
            break
        ifuncs_clip = ifuncs_clip.take(pos_idxs, axis=0)
        coeffs = ifunc.calc_coeffs(ifuncs_clip, displ_clip, n_ss=None, clip=0)
        pos_val_idxs = np.array([ii for ii, val in enumerate(pos_vals) if val])
        new_neg_idxs = pos_val_idxs[~pos]
        logger.info('Negative indexes: {}'.format(new_neg_idxs))
        pos_vals[pos_val_idxs[~pos]] = False

    coeffs_clip[np.where(pos_vals)] = coeffs
    adj = ifunc.calc_adj_displ(ifuncs[row_slice, col_slice, :, :], coeffs_clip)

    ny, nx = ifuncs.shape[2:4]
    clipbox_x = [az_clip, nx - az_clip, nx - az_clip, az_clip, az_clip]
    clipbox_y = [ax_clip, ax_clip, ny - ax_clip, ny - ax_clip, ax_clip]

    resid = displ - adj
    resid_clip = resid[ax_slice, az_slice]
    resid_min, resid_max = np.percentile(resid[ax_slice, az_slice],
                                         [0.5, 99.5])
    dv = 0.02

    resid_stddev = np.std(resid_clip)
    plt.figure(1, figsize=(14, 7))
    plt.clf()
    plt.subplot(2, 2, 1)
    ax = plt.gca()
    ax.axison = False
    vmin, vmax = np.percentile(bias.vals, [0.5, 99.5])
    plt.imshow(bias.vals, vmin=vmin - dv, vmax=vmax + dv)
    plt.title(bias.title)
    plt.colorbar(fraction=0.07)

    plt.subplot(2, 2, 2)
    ax = plt.gca()
    ax.axison = False
    vmin, vmax = np.percentile(error.vals, [0.5, 99.5])
    plt.imshow(error.vals, vmin=vmin - dv, vmax=vmax + dv)
    plt.title(error.title)
    plt.colorbar(fraction=0.07)

    plt.subplot(2, 2, 3)
    plt.imshow(resid, vmin=resid_min, vmax=resid_max)
    ax = plt.gca()
    ax.axison = False
    ax.autoscale(enable=False)
    plt.title('Resids (min={:.4f} max={:.4f} stddev={:.4f})'
              .format(np.min(resid_clip), np.max(resid_clip), resid_stddev))
    plt.plot(clipbox_x, clipbox_y, '-m')
    plt.colorbar(fraction=0.07)

    plt.subplot(2, 2, 4)
    coeffs_all = np.zeros(ifuncs.shape[:2])
    coeffs_2d = coeffs_clip.reshape(ifuncs_clip_4d.shape[:2])
    coeffs_all[row_slice, col_slice] = coeffs_2d
    cimg = np.dstack([coeffs_all, coeffs_all]).reshape(coeffs_all.shape[0],
                                                       coeffs_all.shape[1] * 2)
    ax = plt.gca()
    ax.axison = False
    plt.imshow(cimg, interpolation='nearest')
    ax.autoscale(enable=False)
    coeffs_min, coeffs_max = np.min(coeffs_2d), np.max(coeffs_2d)
    plt.title('Actuator coeffs (min={:.4f} max={:.4f})'
              .format(coeffs_min, coeffs_max))
    plt.colorbar(fraction=0.07)
    r, c = np.mgrid[0:coeffs_all.shape[0], 0:coeffs_all.shape[1]]
    ok = coeffs_all < 0
    plt.plot(c[ok] * 2 + 0.5, r[ok], '.r', ms=10)

    if plot_file:
        plt.savefig(plot_file)

    # Make resulting statistics available
    names = ('input_stddev resid_stddev resid_min resid_max coeffs_min coeffs_max '
             'ax_clip az_clip'.split())
    _locals = locals()
    results = {name: _locals[name] for name in names}
    results['resid_img'] = resid.copy()
    results['bias_img'] = bias.vals.copy()
    results['error_img'] = error.vals.copy()
    results['coeffs_img'] = coeffs_all.copy()
    for name in names:
        logger.info('{:12s} : {:.3f}'.format(name, results[name]))
    return results


def calc_grid(row_clip=2, col_clip=2,
              ax_clip=40, az_clip=80,
              root_dir='mtf/grid',
              n_cycles=[0.5, 1, 2, 3, 4, 5],
              ampls=[0.25, 0.5, 0.75],
              biases=[1, 1.5],
              phases=[0],
              force=False,
              bias_func=displ_uniform_coeffs,
              error_func=displ_sin_ax):
    import subprocess
    if not os.path.exists(root_dir):
        logger.info('Making directory {}'.format(root_dir))
        os.mkdir(root_dir)
    results = shelve.open(os.path.join(root_dir, 'results.shelf'), writeback=True)
    for bias in biases:
        for n_cycle in n_cycles:
            for ampl in ampls:
                for phase in phases:
                    key = (row_clip, col_clip, ax_clip, az_clip,
                           bias, n_cycle, ampl, phase)
                    if not force and repr(key) in results:
                        logger.info('Skipping: {} already in results')
                        continue
                    plot_file = 'mtf_{}_{}_{}_{}_bias{}_cyc{}_ampl{}_phase{}.png'.format(*key)
                    plot_file = os.path.join(root_dir, plot_file)
                    logger.info('Computing results for {}'.format(os.path.basename(plot_file)))
                    error_img = error_func(n_cycle=n_cycle, ampl=ampl, phase=phase)
                    bias_img = bias_func(bias)
                    result = calc_plot_adj(row_clip, col_clip, ax_clip, az_clip,
                                           bias=bias_img, error=error_img,
                                           plot_file=plot_file)
                    # Remove images from results
                    for key in list(result.keys()):
                        if key.endswith('_img'):
                            del result[key]
                    results[repr(key)] = result
                    subprocess.call(['convert', plot_file, '-trim', plot_file])
    results.close()


def make_html(root_dir='mtf/grid'):
    import jinja2

    template = jinja2.Template(open('mtf/template.html').read())
    results_db = shelve.open(os.path.join(root_dir, 'results.shelf'))
    cases = []
    for key in sorted(results_db):
        vals = eval(key)
        names = 'row_clip col_clip ax_clip az_clip bias n_cycle ampl phase'.split()
        case = {name: val for name, val in zip(names, vals)}
        plot_file = 'mtf_{}_{}_{}_{}_bias{}_cyc{}_ampl{}_phase{}.png'.format(*vals)
        case['plot_name'] = os.path.basename(plot_file)
        case['plot_file'] = plot_file
        for name, val in results_db[key].items():
            case[name] = val

        for coeffs_min, color in ((-1.0, '#f00'),
                                  (-0.75, '#f33'),
                                  (-0.5, '#f66'),
                                  (-0.25, '#f99'),
                                  (0.0, '#fbb'),
                                  (0.1, '#fdd')):
            if case['coeffs_min'] < coeffs_min:
                case['bgcolor'] = 'bgcolor={}'.format(color)
                break
        else:
            case['bgcolor'] = ''

        cases.append(case)
    results_db.close()

    out = template.render(cases=cases)
    with open(os.path.join(root_dir, 'index.html'), 'w') as fh:
        fh.write(out)
