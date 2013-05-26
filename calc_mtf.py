"""
Compute "modulation transfer function" in a fashion.
"""

import os
import shelve
import ifunc
import numpy as np
import matplotlib.pyplot as plt

import pyyaks.logger
logger = pyyaks.logger.get_logger()

if 'ifuncs' not in globals():
    print 'Loading influence functions...'
    ifuncs, displ = ifunc.get_ifuncs_displ()
    N_AX, N_AZ = ifuncs.shape[-2:]
    ifuncs_3d = ifuncs.reshape(-1, N_AX, N_AZ)


def displ_sin_ax(n_cycle=1, ampl=0.5, phase=0.0):
    """
    Sinusoidal oscillation in axial direction.  Phase is specified
    in terms of cycles, so phase=0.25 corresponds to a cosine.
    """
    x = (np.arange(N_AX, dtype=float) / N_AX + phase) * 2 * np.pi * n_cycle
    x = x.reshape(-1, 1)
    return np.sin(x) * ampl / n_cycle


def displ_sin_az(n_cycle=1, ampl=0.5, phase=0.0):
    """
    Sinusoidal oscillation in azimuthal direction.  Phase is specified
    in terms of cycles, so phase=0.25 corresponds to a cosine.
    """
    x = (np.arange(N_AZ, dtype=float) / N_AZ + phase) * 2 * np.pi * n_cycle
    x = x.reshape(1, -1)
    return np.sin(x) * ampl / n_cycle


def displ_flat(bias):
    """
    Flat displacement = ``bias`` everywhere.
    """
    out = np.ones((N_AX, N_AZ)) * bias
    return out


def displ_uniform_coeffs(bias):
    """
    Response for all coefficients set to ``bias``.
    """
    out = np.sum(ifuncs_3d, axis=0) * bias
    return out


def calc_plot_adj(row_clip=4, col_clip=4, ax_clip=75, az_clip=150,
                  displ_func=displ_sin_ax, bias_func=displ_flat,
                  bias=1.0, ampl=0.5, n_cycle=1.0, phase=0.0,
                  plot_file=None):
    row_slice = slice(row_clip, -row_clip) if row_clip else slice(None, None)
    col_slice = slice(col_clip, -col_clip) if col_clip else slice(None, None)
    ax_slice = slice(ax_clip, -ax_clip)
    az_slice = slice(az_clip, -az_clip)

    displ = displ_func(n_cycle, ampl, phase) + bias_func(bias)

    # Get stddev of input displacement within clipped region
    input_stddev = np.std(displ[ax_slice, az_slice])

    # Get ifuncs and displ tht are clipped and sub-sampled (at default of n_ss=5)
    ifuncs_clip, displ_clip, M_2d = ifunc.clip_ifuncs_displ(
        ifuncs, displ,
        row_slice=row_slice, col_slice=col_slice,
        ax_clip=ax_clip, az_clip=az_clip)

    coeffs = ifunc.calc_coeffs(ifuncs_clip, displ_clip, n_ss=None, clip=0)
    adj = ifunc.calc_adj_displ(ifuncs[row_slice, col_slice, :, :], coeffs)

    ny, nx = ifuncs.shape[2:4]
    clipbox_x = [az_clip, nx - az_clip, nx - az_clip, az_clip, az_clip]
    clipbox_y = [ax_clip, ax_clip, ny - ax_clip, ny - ax_clip, ax_clip]

    resid = displ - adj
    resid_clip = resid[ax_slice, az_slice]
    resid_min, resid_max = np.percentile(resid[ax_slice, az_slice],
                                         [0.5, 99.5])
    resid_stddev = np.std(resid_clip)
    plt.figure(1, figsize=(8, 12))
    plt.clf()
    plt.subplot(3, 1, 1)
    ax = plt.gca()
    ax.axison = False
    plt.imshow(displ)
    plt.title('Displ-X (bias={:.2f} ampl={:.2f} n_cycle={:.1f} stddev={:.4f})'
              .format(bias, ampl, n_cycle, input_stddev))
    plt.colorbar(fraction=0.07)

    plt.subplot(3, 1, 2)
    plt.imshow(resid, vmin=resid_min, vmax=resid_max)
    ax = plt.gca()
    ax.axison = False
    ax.autoscale(enable=False)
    plt.title('Resids (min={:.4f} max={:.4f} stddev={:.4f})'
              .format(np.min(resid_clip), np.max(resid_clip), resid_stddev))
    plt.plot(clipbox_x, clipbox_y, '-m')
    plt.colorbar(fraction=0.07)

    plt.subplot(3, 1, 3)
    coeffs_all = np.zeros(ifuncs.shape[:2])
    coeffs_2d = coeffs.reshape(ifuncs_clip.shape[:2])
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
    names = 'input_stddev resid_stddev resid_min resid_max coeffs_min coeffs_max'.split()
    _locals = locals()
    results = {name: _locals[name] for name in names}
    return results


def calc_grid(row_clip=2, col_clip=2, ax_clip=40, az_clip=80,
              root_dir='mtf_grid',
              n_cycles=[0.5, 1, 2, 3, 4, 5], ampls=[0.25, 0.5, 0.75], biases=[1],
              force=False):
    results = shelve.open(os.path.join(root_dir, 'results.shelf'), writeback=True)
    for bias in biases:
        for n_cycle in n_cycles:
            for ampl in ampls:
                key = (row_clip, col_clip, ax_clip, az_clip,
                       bias, n_cycle, ampl)
                if not force and repr(key) in results:
                    logger.info('Skipping: {} already in results')
                    continue
                plot_file = 'mtf_{}_{}_{}_{}_bias{}_cyc{}_ampl{}.png'.format(*key)
                logger.info('Computing results for {}'.format(os.path.basename(plot_file)))
                result = calc_plot_adj(row_clip, col_clip, ax_clip, az_clip,
                                       displ_func=displ_sin_ax,
                                       n_cycle=n_cycle,  bias=bias, ampl=ampl,
                                       plot_file=os.path.join(root_dir, plot_file))
                results[repr(key)] = result
    results.close()


def make_html(root_dir='mtf_grid'):
    import jinja2

    template = jinja2.Template(open('mtf_template.html').read())
    results_db = shelve.open(os.path.join(root_dir, 'results.shelf'))
    cases = []
    for key in sorted(results_db):
        vals = eval(key)
        names = 'row_clip col_clip ax_clip az_clip bias n_cycle ampl'.split()
        case = {name: val for name, val in zip(names, vals)}
        plot_file = 'mtf_{}_{}_{}_{}_bias{}_cyc{}_ampl{}.jpg'.format(*vals)
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
