import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import pyyaks.context

import ifunc


RAD2ARCSEC = 206000.  # convert to arcsec for better scale
src = pyyaks.context.ContextDict('src')
files = pyyaks.context.ContextDict('files', basedir='reports')
files.update({'src_dir': '{{src.id}}',
              'index': '{{src.id}}/index',
              'img_x_corr_x': '{{src.id}}/img_x_corr_x',
              'img_x_corr_ry': '{{src.id}}/img_x_corr_ry',
              'img_ry_corr_x': '{{src.id}}/img_ry_corr_x',
              'img_ry_corr_ry': '{{src.id}}/img_ry_corr_ry',
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


class IfuncsReport(object):
    """Provide the infrastructure to create a web based report characterizing a
    set of influence functions.

    :param ifuncs: dict to load ifuncs or define ifuncs
    :param displ: dict to load or define displacements
    :param id_: ifuncs identifier for report naming
    :param clip: number of rows / columns from edge to clip
    :param n_ss: sub-sample period (use 1 out of n_ss rows/columns)
    :param node_sep: node separation (microns)
    :param units: units
    """
    def __init__(self, ifuncs=None, displ=None, id_=1, clip=20, n_ss=5,
                 node_sep=500, units='um'):
        if ifuncs is None:
            ifuncs = {'load_func': ifunc.load_ifuncs,
                      'kwargs': {'case': 'local/10+2/p1000'}}
        if displ is None:
            displ = {'load_func': ifunc.load_displ_grav,
                     'kwargs': {'case': 'local/10+2/p1000'}}

        self.ifuncs = {}
        self.displ = {}

        axes = ('X', 'RY')
        # Check if input ifuncs already has X and RY keys
        if all(axis in ifuncs for axis in axes):
            for axis in axes:
                self.ifuncs[axis] = ifuncs[axis].copy()
        else:  # load ifuncs
            ifuncs['axis'] = 'X'
            self.ifuncs['X'] = ifuncs['load_func'](**ifuncs['kwargs'])
            self.ifuncs['RY'] = np.gradient(self.ifuncs['X'], node_sep)[0]

        self.n_ax, self.n_az = self.ifuncs['X'].shape[2:4]

        # Check if input displ already has X and RY keys
        if all(axis in displ for axis in axes):
            for axis in axes:
                self.displ[axis] = displ[axis].copy()
        else:  # load displacements
            self.displ['X'], self.displ['RY'] = \
                displ['load_func'](self.n_ax, self.n_az, **displ['kwargs'])

        self.n_ss = n_ss
        self.clip = clip
        self.coeffs = {}
        self.adj = defaultdict(dict)
        self.resid_std = defaultdict(dict)
        self.resid_clip_std = defaultdict(dict)
        self.resid_mean = defaultdict(dict)
        self.resid_clip_mean = defaultdict(dict)

    def normalize(self, std, axis='X', clip=True):
        pass

    def calc_adj(self):
        for corr in ('X', 'RY'):
            print 'Correcting using axis', corr
            coeffs = ifunc.calc_coeffs(self.ifuncs[corr], self.displ[corr],
                                       n_ss=self.n_ss, clip=self.clip)
            self.coeffs[corr] = coeffs
            for axis in ('X', 'RY'):
                self.adj[axis][corr] = ifunc.calc_adj_displ(
                    self.ifuncs[axis], coeffs)

    def calc_stats(self, axis='X', corr='X'):
        displ = self.displ[axis]
        adj = self.adj[axis][corr]
        clip = self.clip
        displ_clip = displ[clip:-clip, clip:-clip]
        adj_clip = adj[clip:-clip, clip:-clip]
        resid = displ - adj
        resid_clip = displ_clip - adj_clip

        self.resid_std[axis][corr] = resid.std()
        self.resid_mean[axis][corr] = resid.mean()

        self.resid_clip_std[axis][corr] = resid_clip.std()
        self.resid_clip_mean[axis][corr] = resid_clip.mean()

    def make_imgs_plot(self, axis='X', corr='X'):
        nx = self.n_az
        ny = self.n_ax

        displ = self.displ[axis]
        adj = self.adj[axis][corr]
        clip = self.clip
        displ_clip = displ[clip:-clip, clip:-clip]
        adj_clip = adj[clip:-clip, clip:-clip]

        plt.figure(figsize=(6, 8))
        plt.clf()

        # REFACTOR THIS!
        ax = plt.subplot(3, 1, 1)
        vmin, vmax = np.percentile(np.hstack([displ_clip, adj_clip]),
                                   [0.5, 99.5])
        plt.imshow(displ, vmin=vmin, vmax=vmax)
        ax.axison = False
        ax.autoscale(enable=False)
        plt.plot([clip, nx - clip, nx - clip, clip, clip],
                 [clip, clip, ny - clip, ny - clip, clip], '-m')
        plt.title('Input distortion {}'.format(axis))
        plt.colorbar(fraction=0.07)

        ax = plt.subplot(3, 1, 2)
        plt.imshow(adj, vmin=vmin, vmax=vmax)
        plt.title('Adjustment using {}'.format(corr))
        ax.axison = False
        ax.autoscale(enable=False)
        plt.plot([clip, nx - clip, nx - clip, clip, clip],
                 [clip, clip, ny - clip, ny - clip, clip], '-m')
        plt.colorbar(fraction=0.07)

        resid = displ - adj
        resid_clip = displ_clip - adj_clip
        vmin, vmax = np.percentile(resid_clip, [1.0, 99.0])
        ax = plt.subplot(3, 1, 3)
        plt.title('Residual')
        plt.imshow(resid, vmin=vmin, vmax=vmax)
        ax.axison = False
        ax.autoscale(enable=False)
        plt.plot([clip, nx - clip, nx - clip, clip, clip],
                 [clip, clip, ny - clip, ny - clip, clip], '-m')
        plt.colorbar(fraction=0.07)

    def make_line_plot(self, axis='X', corr='X'):
        plt.figure(figsize=(3.5, 6))
        plt.clf()
        cols = slice(col0, col1)
        plt.plot(displ[:, cols].mean(axis=1) / 10., label='Input / 10')
        plt.plot(adj[:, cols].mean(axis=1) / 10., label='Adjust / 10')
        plt.plot(resid[:, cols].mean(axis=1), label='Resid')
        plt.title('Slice on mean of cols {}:{}'.format(col0, col1))
        plt.legend(loc='best')
        if save:
            plt.savefig(save + '_col.png')

        # Also show the RMS and mean
        print "Input stddev, mean: {:.4f},{:.4f}".format(displ.std(), displ.mean())
        print "Resid stddev, mean: {:.4f},{:.4f}".format(resid.std(), resid.mean())

    def plot(self):
        fig1 = plt.figure(1, figsize=(6, 8))
        fig2 = plt.figure(2, figsize=(6, 8))
        make_plots(self.displ_x, self.adj_x['x'],
                   fig1=fig1, fig2=fig2, clip=self.clip,
                   save=save + 'X' + '_X')

        fig1 = plt.figure(3, figsize=(6, 8))
        fig2 = plt.figure(4, figsize=(6, 8))
        make_plots(displ_ry * scale_ry, adj_ry * scale_ry,
                            fig1=fig1, fig2=fig2, clip=clip,
                            save=save + corr_using + '_RY')

        cols = np.linspace(0, displ_x.shape[1], 10).astype(int)
        cols = (cols[1:] + cols[:-1]) // 2


def main():
    pass


if __name__ == '__main__':
    make_report()
