import numpy as np

import ifunc
import calc_scatter
import logging

RAD2ARCSEC = 206000.  # convert to arcsec for better scale
AXES = ('X', 'RY')
cache = {}


class AutoDict(dict):
    """Implementation of perl's autovivification feature for Python dict."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


CASES = {'10+0_half_exemplar':
             dict(ifuncs={'load_func': ifunc.load_ifuncs,
                          'kwargs': {'case': '10+0_half/p1000'}},
                  displ={'load_func': ifunc.load_file_legendre,
                         'kwargs': {'filename': 'data/exemplar_021312.dat'}},
                  title='10+0_half with exemplar displacements'),
         '10+2_exemplar':
             dict(ifuncs={'load_func': ifunc.load_ifuncs,
                          'kwargs': {'case': 'local/10+2/p1000'}},
                  displ={'load_func': ifunc.load_file_legendre,
                         'kwargs': {'filename': 'data/exemplar_021312.dat'}},
                  title='10+2 supports with exemplar displacements'),
         '10+2_gravity':
             dict(ifuncs={'load_func': ifunc.load_ifuncs,
                          'kwargs': {'case': 'local/10+2/p1000'}},
                  displ={'load_func': ifunc.load_displ_grav,
                         'kwargs': {'case': 'local/10+2/p1000'}},
                  case_id='10+2_gravity',
                  title='10+2 supports with 1-g gravity load'),
         }


class AdjOpticsCase(object):
    """Provide the infrastructure to handle an analysis case for adjustable
    optics including a set of influence functions and displacements.

    :param ifuncs: dict to load ifuncs or define ifuncs
    :param displ: dict to load or define displacements
    :param case_id: case identifier string for report naming
    :param clip: number of rows / columns from edge to clip
    :param n_ss: sub-sample period (use 1 out of n_ss rows/columns)
    :param n_strips: number of axial strips for scatter calculation
    :param node_sep: node separation (microns)
    :param units: units
    """
    def __init__(self, ifuncs=None, displ=None,
                 case_id='10+2_exemplar',
                 subcase_id=1,
                 clip=20, n_ss=5, piston_tilt=True,
                 node_sep=500, units='um',
                 displ_axes=None,
                 corr_axes=None,
                 n_proc=4,
                 n_strips=9):

        case = CASES[case_id]
        ifuncs = case['ifuncs']
        displ = case['displ']

        self.title = case['title']
        self.case_id = case_id
        self.subcase_id = subcase_id
        self.ifuncs = dict()
        self.displ = AutoDict()
        self.piston_tilt = piston_tilt
        self.ifuncs_kwargs = ifuncs['kwargs']
        self.displ_kwargs = displ['kwargs']
        self.node_sep = node_sep
        self.units = units
        self.displ_axes = displ_axes or AXES
        self.corr_axes = corr_axes or AXES
        self.n_proc = n_proc
        self.n_strips = n_strips

        # Check if input ifuncs already has X and RY keys
        if all(axis in ifuncs for axis in AXES):
            for axis in AXES:
                self.ifuncs[axis] = ifuncs[axis].copy()
        else:  # load ifuncs
            logging.info('Loading ifuncs X...')
            self.ifuncs['X'] = ifuncs['load_func'](axis='X',
                                                   **ifuncs['kwargs'])
            logging.info('Computing ifuncs RY...')
            n_ax, n_az = self.ifuncs['X'].shape[-2:]
            ifx = self.ifuncs['X'] = self.ifuncs['X'].reshape(-1, n_ax, n_az)
            ifry = self.ifuncs['RY'] = np.empty_like(ifx)
            for i in range(ifx.shape[0]):
                ifry[i] = np.gradient(ifx[i], node_sep)[0] * RAD2ARCSEC

        self.n_ax, self.n_az = self.ifuncs['X'].shape[-2:]

        # Check if input displ already has X and RY keys
        if all(axis in displ for axis in AXES):
            for axis in AXES:
                self.displ[axis]['img']['full'] = \
                    displ[axis]['img']['full'].copy()
        else:  # load displacements
            logging.info('Loading displ ...')
            self.displ['X']['img']['full'], self.displ['RY']['img']['full'] = \
                displ['load_func'](self.n_ax, self.n_az, **displ['kwargs'])
            self.displ['RY']['img']['full'] *= RAD2ARCSEC

        # Provide clipped displacements
        for axis in AXES:
            self.displ[axis]['img']['clip'] = \
                self.displ[axis]['img']['full'][clip:-clip, clip:-clip]

        self.n_ss = n_ss
        self.clip = clip
        self.coeffs = AutoDict()  # [corr_axis]
        self.adj = AutoDict()  # [axis][corr_axis]
        self.resid = AutoDict()  # [clip][type] for clip=('clip'|'full')
                                 # type = ('img'|'std'|'mean')
        self.scatter = AutoDict()

    def normalize(self, std, axis='X', clip=True):
        pass

    def calc_adj(self):
        for corr in self.corr_axes:
            logging.info('Computing corr coeffs using axis {}...'
                         .format(corr))
            coeffs = ifunc.calc_coeffs(self.ifuncs[corr],
                                       self.displ[corr]['img']['full'],
                                       n_ss=self.n_ss, clip=self.clip)
            self.coeffs[corr] = coeffs
            clip = self.clip
            for axis in self.displ_axes:
                logging.info("Computing adj[{}][{}][full,clip]".
                             format(axis, corr))
                adj = ifunc.calc_adj_displ(self.ifuncs[axis], coeffs)
                self.adj[axis][corr]['full'] = adj
                self.adj[axis][corr]['clip'] = adj[clip:-clip, clip:-clip]

    def calc_stats(self):
        for axis in self.displ_axes:
            for corr in self.corr_axes:
                for clip in ('clip', 'full'):
                    displ = self.displ[axis]['img'][clip]
                    adj = self.adj[axis][corr][clip]
                    resid = displ - adj

                    self.resid[axis][corr]['img'][clip] = resid
                    self.resid[axis][corr]['std'][clip] = resid.std()
                    self.resid[axis][corr]['mean'][clip] = resid.mean()

                    self.displ[axis]['std'][clip] = displ.std()
                    self.displ[axis]['mean'][clip] = displ.mean()

    def calc_scatter(self, filename=None, calc_input=True):
        theta_max = 2.55e-4
        self.thetas = np.linspace(-theta_max, theta_max, 10001)  # 10001

        # Compute the column position of axial strips
        displ = self.displ['X']['img']['clip']
        cols = np.linspace(0, displ.shape[1], self.n_strips + 1).astype(int)
        cols = (cols[1:] + cols[:-1]) // 2
        self.scatter['cols'] = cols

        if calc_input:
            logging.info('Calculating scatter displ (input)')
            displ = self.displ['X']['img']['clip'][:, cols]
            thetas, scatter = calc_scatter.calc_scatter(
                displ, graze_angle=1.428, thetas=self.thetas,
                n_proc=self.n_proc)
            scat = self.scatter['input']
            scat['img'] = displ.copy()
            scat['theta'] = thetas
            scat['vals'] = scatter
            hpd, rmsd = calc_scatter_stats(thetas, scatter)
            scat['hpd'] = hpd
            scat['rmsd'] = rmsd

        for corr in self.corr_axes:
            logging.info('Calculating scatter displ (corrected)')
            displ = self.resid['X'][corr]['img']['clip'][:, cols]
            if self.piston_tilt:  # Remove piston and tilt
                remove_piston_tilt(displ)

            thetas, scatter = calc_scatter.calc_scatter(displ,
                                                        graze_angle=1.428,
                                                        thetas=self.thetas,
                                                        n_proc=self.n_proc)
            scat = self.scatter['corr'][corr]
            scat['img'] = displ.copy()
            scat['theta'] = thetas
            scat['vals'] = scatter
            hpd, rmsd = calc_scatter_stats(thetas, scatter)
            scat['hpd'] = hpd
            scat['rmsd'] = rmsd


def remove_piston_tilt(displ):
    """Remove piston and tilt independently from each axial strip in ``displ``.
    """
    x = np.arange(len(displ))
    ps = np.polyfit(x, displ, 1)
    for i in range(displ.shape[1]):
        displ[:, i] -= np.polyval(ps[:, i], x)


def calc_scatter_stats(theta, scatter):
    """Return stats about scatter distribution.
    """
    i_mid = len(theta) // 2
    i1 = 2 * i_mid - 1
    angle = theta[i_mid:i1]

    sym_scatter = scatter[i_mid:i1] + scatter[i_mid - 1:0:-1]
    sym_scatter /= sym_scatter.sum()

    ee = np.cumsum(sym_scatter)
    i_hpr = np.searchsorted(ee, 0.5)
    angle_hpd = angle[i_hpr] * 2

    i99 = np.searchsorted(ee, 0.99)
    angle_rmsd = 2 * np.sqrt(np.sum(angle[:i99] ** 2 * sym_scatter[:i99])
                             / np.sum(sym_scatter[:i99]))
    return angle_hpd, angle_rmsd
