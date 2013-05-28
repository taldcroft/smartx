import sherpa.astro.ui as ui
import numpy as np
import ifunc

SHERPA_CONFIGS = {}
ONES = np.array(1000)

N_SS = 5
col_slice = slice(374, 400)

if 'ifuncs' not in globals():
    ifuncs = ifunc.load_ifuncs(case='10+0_baseline')
    N_ROWS, N_COLS, N_AX, N_AZ = ifuncs.shape
    displ_x_all, displ_ry_all = ifunc.load_displ_legendre(N_AX, N_AZ, offset_az=2)

    ifuncs = ifuncs[:, 10:11, :, col_slice]  # ifuncs from actuator column 10, columns 380-400
    N_ROWS, N_COLS, N_AX, N_AZ = ifuncs.shape

    i_ss, j_ss = np.mgrid[0:N_AX:N_SS, 0:N_AZ:N_SS]
    M_3d_all = ifuncs.reshape(-1, N_AX, N_AZ)
    M_3d = M_3d_all[:, i_ss, j_ss]
    M_2d = M_3d.reshape(M_3d.shape[0], -1).transpose().copy()

    displ_x_all = displ_x_all[:, col_slice]
    displ_x = displ_x_all[i_ss, j_ss].flatten().copy()


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
    def __init__(self, model, M_2d, displ):
        self.M_2d = M_2d
        self.displ = displ
        self.model = model
        self.min_fit_stat = None
        self.min_parvals = [par.val for par in self.model.pars]

    def __call__(self, _data, _model, staterror=None, syserror=None, weight=None):
        """Calculate fit statistic for the xija model.  The args _data and _model
        are sent by Sherpa but they are fictitious -- the real data and model are
        stored in the xija model self.model.
        """
        # adj = self.M_2d.dot(np.abs(self.parvals))  # self.parvals set in CalcModel.__call__
        adj = self.M_2d.dot(self.parvals)  # self.parvals set in CalcModel.__call__
        fit_stat = np.sum((adj - self.displ) ** 2)

        # fit_logger.info('Fit statistic: {:.20f} {}'.format(fit_stat, self.parvals))

        if self.min_fit_stat is None or fit_stat < self.min_fit_stat:
            self.min_fit_stat = fit_stat
            self.min_parvals = self.parvals

        return fit_stat, np.ones_like(self.displ)


def fit_coeffs(method='simplex'):
    method = method
    dummy_data = np.zeros(100)
    dummy_times = np.arange(100)
    ui.load_arrays(1, dummy_times, dummy_data)

    ui.set_method(method)
    ui.get_method().config.update(SHERPA_CONFIGS.get(method, {}))

    calc_model = CalcModel()
    ui.load_user_model(calc_model, 'axo_mod')  # sets global axo_mod

    parnames = []
    for row in range(N_ROWS):
        for col in range(N_COLS):
            parnames.append('adj_{}_{}'.format(row, col))
    ui.add_user_pars('axo_mod', parnames)
    ui.set_model(1, 'axo_mod')

    calc_stat = CalcStat(axo_mod, M_2d, displ_x)
    ui.load_user_stat('axo_stat', calc_stat, lambda x: np.ones_like(x))
    ui.set_stat(axo_stat)
    calc_model.set_calc_stat(calc_stat)

    # Set frozen, min, and max attributes for each axo_mod parameter
    for par in axo_mod.pars:
        par.val = 0.0
        par.min = -5
        par.max = 5

    ui.fit(1)

    coeffs = np.array([(par.val) for pars in axo_mod.pars])
    return coeffs
