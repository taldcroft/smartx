import sherpa.astro.ui as ui
import numpy as np
import ifunc

import xija.clogging as clogging   # get rid of this or something

SHERPA_CONFIGS = {'levmar': {'epsfcn': 10.0, 'verbose': 1}}

if not 'ifuncs' in globals():
    N_SS = 5
    CLIP = 50
    ifuncs = ifunc.load_ifuncs(case='10+0_baseline/')
    N_ROWS, N_COLS, N_AX, N_AZ = ifuncs.shape
    ax_slice = slice(CLIP, N_AX - CLIP, N_SS)
    az_slice = slice(CLIP, N_AZ - CLIP, N_SS)
    i_ss, j_ss = np.mgrid[ax_slice, az_slice]
    M_3d_all = ifuncs.reshape(-1, N_AX, N_AZ)
    M_3d = M_3d_all[:, i_ss, j_ss]
    M_2d = M_3d.reshape(M_3d.shape[0], -1).transpose().copy()

    displ_x_all, displ_ry_all = ifunc.load_displ_legendre(N_AX, N_AZ, offset_az=2)
    DISPL_X = displ_x_all[i_ss, j_ss].flatten().copy()
    print 'Computing coeffs'
    coeffs = ifunc.calc_coeffs(ifuncs, displ_x_all, n_ss=N_SS, clip=CLIP)

fit_logger = clogging.config_logger(
    'fit', level=clogging.INFO,
    format='[%(levelname)s] (%(processName)-10s) %(message)s')


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

        return fit_stat, np.ones_like(self.displ)


def calc_inversion_solution(ifuncs, displ, n_ss, clip):
    coeffs = ifunc.calc_coeffs(ifuncs, displ, n_ss=n_ss, clip=clip)
    return coeffs


def fit_adjuster_set(coeffs, adj_idxs, method='simplex'):
    """
    Find best fit parameters for an arbitrary subset of adjustors
    specified by the array ``adj_idxs``.  The input ``coeffs`` are
    the best-fit adjustor coefficients for the last iteration.
    """
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

    return coeffs


def iter_fit(coeffs, n_samp=10, n_adj=10, method='simplex'):
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

