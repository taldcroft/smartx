import cPickle as pickle
import os
from itertools import izip

import matplotlib.pyplot as plt
import numpy as np
import ifunc
import calc_scatter

import xija.clogging as clogging   # get rid of this or something

SHERPA_CONFIGS = {'levmar': {'epsfcn': 10.0, 'verbose': 1},
                  'simplex': {'ftol': 1e-3}}


fit_logger = clogging.config_logger(
    'fit', level=clogging.INFO,
    format='[%(levelname)s] (%(processName)-10s) %(message)s')


def calc_inversion_solution(ifuncs, displ, n_ss, clip):
    coeffs = ifunc.calc_coeffs(ifuncs, displ, n_ss=n_ss, clip=clip)
    return coeffs


def calc_adj_displ(ifuncs, coeffs):
    """Return the adjusted displacement as a 2-d array for the given ``ifuncs``
    and ``coeffs``.
    """
    n_ax, n_az = ifuncs.shape[-2:]
    M_3d_all = ifuncs.reshape(-1, n_ax, n_az)
    M_2d_all = M_3d_all.reshape(M_3d_all.shape[0], -1).transpose()
    adj = M_2d_all.dot(coeffs)
    adj_2d = adj.reshape(n_ax, n_az)

    return adj_2d


class CalcModel(object):
    def set_calc_stat(self, calc_stat):
        self.calc_stat = calc_stat

    def __call__(self, parvals, x):
        """This is the Sherpa calc_model function, but in this case calc_model does not
        actually calculate anything but instead just stores the desired paramters.  This
        allows for multiprocessing where only the fit statistic gets passed between nodes.
        """
        self.calc_stat.parvals = parvals

        return np.ones_like(x)


class CalcStat(object):
    def __init__(self, base_adj_displ, M_2d, displ):
        self.M_2d = M_2d
        self.displ = displ
        self.base_adj_displ_MINUS_displ = base_adj_displ - displ
        self.min_fit_stat = None

    def __call__(self, _data, _model, staterror=None, syserror=None, weight=None):
        """Calculate fit statistic for the xija model.  The args _data and _model
        are sent by Sherpa but they are fictitious -- the real data and model are
        stored in the xija model self.model.
        """
        # print 'parvals =', self.parvals
        adj = self.M_2d.dot(np.abs(self.parvals))  # self.parvals set in CalcModel.__call__

        # Parvals represents the delta change in the adjuster coefficients.

        # fit_stat = np.sum((adj + base_adj_displ - displ) ** 2)
        # (but use precomputed value)
        fit_stat = np.sum((adj + self.base_adj_displ_MINUS_displ) ** 2)
        # print 'fit_stat =', fit_stat

        # fit_logger.info('Fit statistic: {:.20f} {}'.format(fit_stat, self.parvals))

        #sys.stdout.write('{}\r'.format(fit_stat))
        #sys.stdout.flush()
        return fit_stat, np.ones_like(self.displ)


def fit_adjuster_set(coeffs, adj_idxs, method='simplex'):
    """
    Find best fit parameters for an arbitrary subset of adjustors
    specified by the array ``adj_idxs``.  The input ``coeffs`` are
    the best-fit adjustor coefficients for the last iteration.
    """
    import sherpa.astro.ui as ui

    dummy_data = np.zeros(100)
    dummy_times = np.arange(100)
    ui.load_arrays(1, dummy_times, dummy_data)

    ui.set_method(method)
    ui.get_method().config.update(SHERPA_CONFIGS.get(method, {}))

    calc_model = CalcModel()
    ui.load_user_model(calc_model, 'axo_mod')  # sets global axo_mod

    parnames = []
    for adj_idx in adj_idxs:
        parnames.append('adj_{}'.format(adj_idx))
    ui.add_user_pars('axo_mod', parnames)
    ui.set_model(1, 'axo_mod')

    coeffs = coeffs.copy()  # Don't modify input coeffs
    coeffs[coeffs < 0] = 0  # Don't allow negative coeffs

    # Set frozen, min, and max attributes for each axo_mod parameter
    for adj_idx, par in zip(adj_idxs, axo_mod.pars):
        par.min = -1000
        par.max = 1000
        par.val = coeffs[adj_idx]
        print 'Setting {} to {}'.format(adj_idx, par.val)

    # Compute base adjusted displacements assuming all the fitted actuators
    # have zero drive level.
    coeffs[adj_idxs] = 0
    base_adj_displ = M_2d.dot(coeffs)

    m_2d = M_2d[:, adj_idxs].copy()
    print m_2d.shape
    calc_stat = CalcStat(base_adj_displ, m_2d, DISPL_X)
    ui.load_user_stat('axo_stat', calc_stat, lambda x: np.ones_like(x))
    ui.set_stat(axo_stat)
    calc_model.set_calc_stat(calc_stat)

    ui.fit(1)

    # Update coeffs with the values determined in fitting
    for adj_idx, par in zip(adj_idxs, axo_mod.pars):
        coeffs[adj_idx] = abs(par.val)

    return coeffs, ui.get_fit_results()


def rand_iter_fit(coeffs, n_samp=10, n_adj=10, method='simplex'):
    coeffs1 = coeffs.copy()
    idxs = np.mod(np.arange(len(coeffs) * n_samp), len(coeffs))
    np.random.shuffle(idxs)
    for i in range(0, len(idxs), n_adj):
        print '*' * 30
        print 'i =', i
        print '*' * 30
        adj_idxs = idxs[i:i + n_adj]
        coeffs1 = fit_adjuster_set(coeffs1, adj_idxs, method)

    return coeffs1


def sweep_cols_fit(coeffs, n_sweep=1000, method='simplex'):
    coeffs1 = coeffs.copy()
    n_col = 22
    n_row = 22
    n_col2 = n_col // 2
    idxs2d = np.arange(n_row * n_col).reshape(n_row, n_col)
    for i_sweep in range(n_sweep):
        for col in range(n_col2):
            for i_col in (n_col2 - 1 - col, n_col2 + col):
                print '*' * 30
                print 'col =', i_col
                print '*' * 30
                adj_idxs = idxs2d[:, i_col].flatten()
                coeffs1, fit_results = fit_adjuster_set(coeffs1, adj_idxs, method)
        pickle.dump(fit_results, open('sweep_fit_results.pkl', 'w'), protocol=-1)
        np.save('sweep_coeffs.npy', coeffs1)

        if os.path.exists('stop_sweep'):
            break

    return coeffs1, fit_results


def boxes_fit(coeffs, n_iter=1000, method='simplex'):
    coeffs1 = coeffs.copy()
    box_idxs = [0, 5, 9, 13, 17, 22]
    idxs2d = np.arange(N_ROWS * N_COLS).reshape(N_ROWS, N_COLS)
    fit_results = []
    for i_iter in range(n_iter):
        for row_i0, row_i1 in izip(box_idxs[:-1], box_idxs[1:]):
            for col_i0, col_i1 in izip(box_idxs[:-1], box_idxs[1:]):
                print '*' * 30
                print 'rows, cols =', row_i0, row_i1, col_i0, col_i1
                print '*' * 30
                adj_idxs = idxs2d[row_i0:row_i1, col_i0:col_i1].flatten()
                coeffs1, fit_result = fit_adjuster_set(coeffs1, adj_idxs, method)
                fit_results.append(fit_result)
        pickle.dump(fit_results, open('box_fit_results.pkl', 'w'), protocol=-1)
        np.save('box_coeffs.npy', coeffs1)
        if os.path.exists('stop_box'):
            break

    return coeffs1, fit_results


def make_scatter_plot(aoc, corr='X', filename=None):
    print 'Plotting scatter displ'

    plt.figure(11, figsize=(5, 3.5))
    plt.clf()
    plt.rc("legend", fontsize=9)

    scale = (np.max(aoc.scatter['corr'][corr]['vals']) /
             np.max(aoc.scatter['input']['vals'])) / 2.0
    scat = aoc.scatter['input']
    label = 'Input HPD={:.2f} RMSD={:.2f}'.format(scat['hpd'],
                                                  scat['rmsd'])
    plt.plot(scat['theta'], scat['vals'] * scale, '-b', label=label)
    x0 = max(scat['rmsd'] * 2, 3)

    scat = aoc.scatter['corr'][corr]
    label = 'Corr HPD={:.2f} RMSD={:.2f}'.format(scat['hpd'],
                                                 scat['rmsd'])
    plt.plot(scat['theta'], scat['vals'], '-r', label=label)

    plt.xlabel('Arcsec')
    plt.title('Scatter')
    plt.xlim(-x0, x0)
    plt.grid()
    plt.legend(loc='upper left')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)


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
    out['rmsd'] = 2 * np.sqrt(np.sum(angle[:i99] ** 2 * sym_scatter[:i99])
                              / np.sum(sym_scatter[:i99]))
    out['ee_angle'] = angle
    out['ee_val'] = ee
    out['ee_d50'] = out['hpd']
    out['ee_d90'] = angle[np.searchsorted(ee, 0.9)] * 2
    out['ee_d99'] = angle[i99] * 2
    return out  # angle_hpd, angle_rmsd, angle, ee


def calc_scatter_vals(coeffs, n_strips=21, n_proc=4, ax_clip=50,
                      az_clip=75, ifuncs=None):
    out = {'input': {}, 'corr': {}}
    theta_max = 2.55e-4
    thetas = np.linspace(-theta_max, theta_max, 10001)  # 10001
    out['theta'] = thetas

    row_slice = slice(ax_clip, N_AX - ax_clip)
    col_slice = slice(az_clip, N_AZ - az_clip)

    # Compute the column position of axial strips
    displ = displ_x_all[row_slice, col_slice]
    cols = np.linspace(0, displ.shape[1], n_strips + 1).astype(int)
    cols = (cols[1:] + cols[:-1]) // 2
    out['cols'] = cols

    print('Calculating scatter displ (input)')
    theta_arcs, scatter = calc_scatter.calc_scatter(
        displ[:, cols], graze_angle=1.428, thetas=thetas, n_proc=n_proc)
    print np.std(scatter), np.sum(scatter), np.std(displ[:, cols])
    figure(1)
    clf()
    plot(theta_arcs, scatter)
    scat = out['input']
    scat['img'] = displ.copy()
    scat['vals'] = scatter
    stats = calc_scatter_stats(theta_arcs, scatter)
    scat.update(stats)

    print('Calculating scatter displ (corrected)')
    adj_displ_all = calc_adj_displ(ifuncs, coeffs)
    adj_displ = adj_displ_all[row_slice, col_slice]
    resid = adj_displ - displ

    theta_arcs, scatter = calc_scatter.calc_scatter(
        resid[:, cols], graze_angle=1.428, thetas=thetas, n_proc=n_proc)
    print np.std(scatter), np.sum(scatter), np.std(displ[:, cols])
    plot(theta_arcs, scatter)
    scat = out['corr']
    scat['img'] = adj_displ.copy()
    scat['vals'] = scatter
    stats = calc_scatter_stats(theta_arcs, scatter)
    scat.update(stats)

    return out


