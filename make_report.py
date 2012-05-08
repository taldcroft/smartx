import os

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import jinja2
import Ska.File

import pyyaks.context

import ifunc
import calc_scatter


AXES = ('X', 'RY')
RAD2ARCSEC = 206000.  # convert to arcsec for better scale
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


CASES = {'10+2_exemplar':
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
                 case_id='10+2_gravity',
                 title='10+2 supports with 1-g gravity load',
                 clip=20, n_ss=5,
                 node_sep=500, units='um'):
        if ifuncs is None:
            ifuncs = CASES[case_id]['ifuncs']
        if displ is None:
            displ = CASES[case_id]['displ']

        self.title = title
        self.case_id = case_id
        self.ifuncs = dict()
        self.displ = AutoDict()

        # Check if input ifuncs already has X and RY keys
        if all(axis in ifuncs for axis in AXES):
            for axis in AXES:
                self.ifuncs[axis] = ifuncs[axis].copy()
        else:  # load ifuncs
            print 'Loading ifuncs X...'
            self.ifuncs['X'] = ifuncs['load_func'](axis='X',
                                                   **ifuncs['kwargs'])
            print 'Computing ifuncs RY...'
            ifx = self.ifuncs['X']
            ifry = self.ifuncs['RY'] = np.empty_like(self.ifuncs['X'])
            for i in range(ifx.shape[0]):
                for j in range(ifx.shape[1]):
                    ifry[i, j] = np.gradient(ifx[i, j], node_sep)[0]

        self.n_ax, self.n_az = self.ifuncs['X'].shape[2:4]

        # Check if input displ already has X and RY keys
        if all(axis in displ for axis in AXES):
            for axis in AXES:
                self.displ[axis]['img']['full'] = \
                    displ[axis]['img']['full'].copy()
        else:  # load displacements
            print 'Loading displ ...'
            self.displ['X']['img']['full'], self.displ['RY']['img']['full'] = \
                displ['load_func'](self.n_ax, self.n_az, **displ['kwargs'])

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
        self.calc_adj()
        self.calc_stats()

    def normalize(self, std, axis='X', clip=True):
        pass

    def calc_adj(self):
        for corr in AXES:
            print 'Computing corr coeffs using axis', corr, '...'
            coeffs = ifunc.calc_coeffs(self.ifuncs[corr],
                                       self.displ[corr]['img']['full'],
                                       n_ss=self.n_ss, clip=self.clip)
            self.coeffs[corr] = coeffs
            clip = self.clip
            for axis in AXES:
                print "Computing adj[{}][{}][full,clip]".format(axis, corr)
                adj = ifunc.calc_adj_displ(self.ifuncs[axis], coeffs)
                self.adj[axis][corr]['full'] = adj
                self.adj[axis][corr]['clip'] = adj[clip:-clip, clip:-clip]

    def calc_stats(self):
        for axis in AXES:
            for corr in AXES:
                for clip in ('clip', 'full'):
                    displ = self.displ[axis]['img'][clip]
                    adj = self.adj[axis][corr][clip]
                    resid = displ - adj

                    self.resid[axis][corr]['img'][clip] = resid
                    self.resid[axis][corr]['std'][clip] = resid.std()
                    self.resid[axis][corr]['mean'][clip] = resid.mean()

                    self.displ[axis]['std'][clip] = displ.std()
                    self.displ[axis]['mean'][clip] = displ.mean()

    def make_scatter_plot(self, axis='X', corr='X', filename=None):
        print 'Calculating scatter displ'
        x, y = calc_scatter.calc_scatter(self.displ[axis]['img']['clip'])
        plt.figure(11, figsize=(5, 3.5))
        plt.clf()
        plt.plot(x, y, '-b')
        print 'Calculating scatter resid'
        x, y = calc_scatter.calc_scatter(self.resid[axis][corr]['img']['clip'])
        plt.plot(x, y, '-r')
        plt.xlabel('Arcsec')
        plt.title('Scatter')
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


def make_report(ifr):
    template = jinja2.Template(open('report_template.html').read())
    src['id'] = ifr.case_id
    subcases = []
    stats = ('mean', 'std')
    clips = ('full', 'clip')

    ratios = AutoDict()
    for axis in AXES:
        for corr in AXES:
            for stat in stats:
                for clip in clips:
                    ratio = (ifr.displ[axis][stat][clip]
                             / ifr.resid[axis][corr][stat][clip])
                    ratios[axis][corr][stat][clip] = ratio

    for axis in AXES:
        src['axis'] = axis
        for corr in AXES:
            src['corr'] = corr
            ifr.make_imgs_plot(axis, corr, files['img_corr.png'].abs)
            if axis == 'X':
                ifr.make_scatter_plot(axis, corr, files['scatter.png'].abs)
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
                          title=ifr.title)
    with open(files['index.html'].abs, 'w') as f:
        f.write(out)

if __name__ == '__main__':
    ifr = IfuncsReport()
    make_report()
