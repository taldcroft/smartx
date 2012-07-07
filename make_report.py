#!/usr/bin/env python

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jinja2
import Ska.File

import pyyaks.context

import ifunc
import calc_scatter


RAD2ARCSEC = 206000.  # convert to arcsec for better scale
AXES = ('X', 'RY')
src = pyyaks.context.ContextDict('src')
files = pyyaks.context.ContextDict('files', basedir='reports')
files.update({'src_dir': '{{src.id}}',
              'index': '{{src.id}}/index',
              'img_corr': '{{src.id}}/img_{{src.axis}}_corr_{{src.corr}}',
              'scatter': '{{src.id}}/scatter_corr_{{src.corr}}',
              })
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


class IfuncsReport(object):
    """Provide the infrastructure to create a web based report characterizing a
    set of influence functions.

    :param ifuncs: dict to load ifuncs or define ifuncs
    :param displ: dict to load or define displacements
    :param case_id: case identifier string for report naming
    :param clip: number of rows / columns from edge to clip
    :param n_ss: sub-sample period (use 1 out of n_ss rows/columns)
    :param node_sep: node separation (microns)
    :param units: units
    """
    def __init__(self, ifuncs=None, displ=None,
                 case_id='10+2_exemplar',
                 subcase_id=1,
                 clip=20, n_ss=5, piston_tilt=True,
                 node_sep=500, units='um',
                 displ_axes=None,
                 corr_axes=None):

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

        # Check if input ifuncs already has X and RY keys
        if all(axis in ifuncs for axis in AXES):
            for axis in AXES:
                self.ifuncs[axis] = ifuncs[axis].copy()
        else:  # load ifuncs
            print 'Loading ifuncs X...'
            self.ifuncs['X'] = ifuncs['load_func'](axis='X',
                                                   **ifuncs['kwargs'])
            print 'Computing ifuncs RY...'
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
            print 'Loading displ ...'
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

        self.calc_adj()
        self.calc_stats()
        self.calc_scatter()

    def normalize(self, std, axis='X', clip=True):
        pass

    def calc_adj(self):
        for corr in self.corr_axes:
            print 'Computing corr coeffs using axis', corr, '...'
            coeffs = ifunc.calc_coeffs(self.ifuncs[corr],
                                       self.displ[corr]['img']['full'],
                                       n_ss=self.n_ss, clip=self.clip)
            self.coeffs[corr] = coeffs
            clip = self.clip
            for axis in self.displ_axes:
                print "Computing adj[{}][{}][full,clip]".format(axis, corr)
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

    def calc_scatter(self, filename=None, n_strips=9):
        axis = 'X'
        theta_max = 2.55e-4
        self.thetas = np.linspace(-theta_max, theta_max, 10001)  # 10001
        n_ss = self.n_az // n_strips

        resid = self.resid['X']['RY']['img']['full']
        cols = np.linspace(0, resid.shape[1], 10).astype(int)
        cols = (cols[1:] + cols[:-1]) // 2
        np.save('resid_X_RY.npy', resid[::2, cols])

        for corr in self.corr_axes:
            print 'Calculating scatter displ (input)'
            displ = self.displ[axis]['img']['clip'][:, ::n_ss]
            thetas, scatter = calc_scatter.calc_scatter(displ,
                                                        graze_angle=1.428,
                                                        thetas=self.thetas)
            scat = self.scatter['input'][corr]
            scat['theta'] = thetas
            scat['vals'] = scatter
            hpd, rmsd = calc_scatter_stats(thetas, scatter)
            scat['hpd'] = hpd
            scat['rmsd'] = rmsd

            print 'Calculating scatter displ (corrected)'
            displ = self.resid[axis][corr]['img']['clip'][:, ::n_ss]
            if self.piston_tilt:  # Remove piston and tilt
                remove_piston_tilt(displ)

            thetas, scatter = calc_scatter.calc_scatter(displ,
                                                        graze_angle=1.428,
                                                        thetas=self.thetas)
            scat = self.scatter['corr'][corr]
            scat['theta'] = thetas
            scat['vals'] = scatter
            hpd, rmsd = calc_scatter_stats(thetas, scatter)
            scat['hpd'] = hpd
            scat['rmsd'] = rmsd

    def make_scatter_plot(self, corr='X', filename=None, n_strips=9):
        print 'Plotting scatter displ'

        plt.figure(11, figsize=(5, 3.5))
        plt.clf()
        plt.rc("legend", fontsize=9)

        scale = (np.max(self.scatter['corr'][corr]['vals']) /
                 np.max(self.scatter['input'][corr]['vals'])) / 2.0
        scat = self.scatter['input'][corr]
        label = 'Input HPD={:.2f} RMSD={:.2f}'.format(scat['hpd'],
                                                      scat['rmsd'])
        plt.plot(scat['theta'], scat['vals'] * scale, '-b', label=label)
        x0 = scat['rmsd'] * 2

        scat = self.scatter['corr'][corr]
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

    def make_imgs_plot(self, axis='X', corr='X', filename=None):
        nx = self.n_az
        ny = self.n_ax

        displ = self.displ[axis]['img']
        adj = self.adj[axis][corr]
        resid = self.resid[axis][corr]['img']

        n_clip = self.clip
        clipbox_x = [n_clip, nx - n_clip, nx - n_clip, n_clip, n_clip]
        clipbox_y = [n_clip, n_clip, ny - n_clip, ny - n_clip, n_clip]

        plt.figure(10, figsize=(6, 8))
        plt.clf()

        ax = plt.subplot(3, 1, 1)
        vmin, vmax = np.percentile(np.hstack([displ['clip'], adj['clip']]),
                                   [0.5, 99.5])
        plt.imshow(displ['full'], vmin=vmin, vmax=vmax)
        ax.axison = False
        ax.autoscale(enable=False)
        plt.plot(clipbox_x, clipbox_y, '-m')
        plt.title('Input distortion {}'.format(axis))
        plt.colorbar(fraction=0.07)

        ax = plt.subplot(3, 1, 2)
        plt.imshow(adj['full'], vmin=vmin, vmax=vmax)
        plt.title('Adjustment using {}'.format(corr))
        ax.axison = False
        ax.autoscale(enable=False)
        plt.plot(clipbox_x, clipbox_y, '-m')
        plt.colorbar(fraction=0.07)

        vmin, vmax = np.percentile(resid['clip'], [0.5, 99.5])
        ax = plt.subplot(3, 1, 3)
        plt.title('Residual')
        plt.imshow(resid['full'], vmin=vmin, vmax=vmax)
        ax.axison = False
        ax.autoscale(enable=False)
        plt.plot(clipbox_x, clipbox_y, '-m')
        plt.colorbar(fraction=0.07)
        if filename is not None:
            plt.savefig(filename)

    def write_imgs_data(self, axis='X', corr='X', filename=None):
        resid = self.resid[axis][corr]['img']['full']
        print 'Writing', filename
        with open(filename, 'w') as out:
            for i_az in xrange(self.n_az):
                for i_ax in xrange(self.n_ax):
                    out.write('{:5d} {:5d} {:f}\n'
                              .format(i_az, i_ax, resid[i_ax, i_az]))


def remove_piston_tilt(displ):
    """Remove piston and tilt independently from each axial strip in ``displ``.
    """
    x = np.arange(len(displ))
    ps = np.polyfit(x, displ, 1)
    for i in range(displ.shape[1]):
        displ[:, i] -= np.polyval(ps[:, i], x)


def make_report(ifr):
    template = jinja2.Template(open('report_template.html').read())
    src['id'] = '{}/{}'.format(ifr.case_id, ifr.subcase_id)
    if not os.path.exists(files['src_dir'].abs):
        os.makedirs(files['src_dir'].abs)
    subcases = []
    stats = ('mean', 'std')
    clips = ('full', 'clip')

    ratios = AutoDict()
    for axis in ifr.displ_axes:
        for corr in ifr.corr_axes:
            for stat in stats:
                for clip in clips:
                    ratio = (ifr.displ[axis][stat][clip]
                             / ifr.resid[axis][corr][stat][clip])
                    ratios[axis][corr][stat][clip] = ratio

    for axis in ifr.displ_axes:
        src['axis'] = axis
        for corr in ifr.corr_axes:
            src['corr'] = corr
            ifr.make_imgs_plot(axis, corr, files['img_corr.png'].abs)
            ifr.write_imgs_data(axis, corr, files['img_corr.dat'].abs)
            if axis == 'X':
                ifr.make_scatter_plot(corr, files['scatter.png'].abs)
            with Ska.File.chdir(files['src_dir'].abs):
                subcases.append({'img_corr_file': files['img_corr.png'].rel,
                                 'scatter_file': files['scatter.png'].rel,
                                 'axis': axis,
                                 'corr': corr,
                                 'ratios': ratios[axis][corr],
                                 'displ': ifr.displ[axis],
                                 'resid': ifr.resid[axis][corr],
                                 })
    out = template.render(subcases=subcases,
                          ifr=ifr)
    with open(files['index.html'].abs, 'w') as f:
        f.write(out)


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Make report')
    parser.add_argument('--case-id',
                        type=str,
                        default='10+0_half_exemplar',
                        help='Case ID')
    parser.add_argument('--subcase-id',
                        type=str,
                        default='1',
                        help='Sub-case ID')
    parser.add_argument('--clip',
                        type=int, default=20,
                        help='Edge clippping')
    parser.add_argument('--ss',
                        type=int, default=5,
                        help='Sub-sampling')
    parser.add_argument('--piston-tilt',
                        type=int, default=1,
                        help='Sub-sampling')
    parser.add_argument('--displ-axes',
                        type=str, default="X,RY",
                        help='Displacement axes to compute')
    parser.add_argument('--corr-axes',
                        type=str, default="X,RY",
                        help='Correction axes to compute')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    displ_axes = args.displ_axes.split(',')
    corr_axes = args.corr_axes.split(',')
    ifr = IfuncsReport(case_id=args.case_id, subcase_id=args.subcase_id,
                       clip=args.clip, n_ss=args.ss,
                       piston_tilt=bool(args.piston_tilt),
                       displ_axes=displ_axes,
                       corr_axes=corr_axes)
    make_report(ifr)

    # def __init__(self, ifuncs=None, displ=None,
    #              case_id='10+2_exemplar',
    #              title='10+2 supports with exemplar displacements',
    #              clip=20, n_ss=5,
    #              node_sep=500, units='um'):
