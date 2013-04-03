#!/usr/bin/env python

import os
import cPickle as pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jinja2
import Ska.File
import pyyaks.context

from adj_opt_case import AdjOpticsCase, AutoDict

src = pyyaks.context.ContextDict('src')
files = pyyaks.context.ContextDict('files', basedir='reports')
files.update({'src_dir': '{{src.id}}',
              'index': '{{src.id}}/index',
              'aoc': '{{src.id}}/aoc',
              'img_corr': '{{src.id}}/img_{{src.axis}}_corr_{{src.corr}}',
              'img_scatter': ('{{src.id}}/img_scatter_{{src.scatter_type}}'
                              '_corr_{{src.corr}}'),
              'axial_resid': '{{src.id}}/axial_resid_{{src.corr}}',
              'scatter': '{{src.id}}/scatter_corr_{{src.corr}}',
              'enc_energy': '{{src.id}}/enc_energy_corr_{{src.corr}}',
              })


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
    parser.add_argument('--n-strips',
                        type=int, default=20,
                        help='Number of strips for scatter calculation')
    parser.add_argument('--piston-tilt',
                        type=int, default=0,
                        help='Apply piston / tilt correction')
    parser.add_argument('--displ-axes',
                        type=str, default="X,RY",
                        help='Displacement axes to compute')
    parser.add_argument('--corr-axes',
                        type=str, default="X,RY",
                        help='Correction axes to compute')
    parser.add_argument('--bias-mult',
                        type=float,
                        help='Bias multiplier factor')
    args = parser.parse_args()
    return args


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


def make_ee_plot(aoc, corr='X', filename=None):
    print 'Plotting encircled energy'

    plt.figure(21, figsize=(5, 3.5))
    plt.clf()
    plt.rc("legend", fontsize=9)

    scat = aoc.scatter['input']
    label = 'Input diam 50%={:.2f} 90%={:.2f} 99%={:.2f} arcsec'.format(
        scat['ee_d50'], scat['ee_d90'], scat['ee_d99'])
    plt.plot(scat['ee_angle'], scat['ee_val'], '-b', label=label)

    scat = aoc.scatter['corr'][corr]
    label = 'Corr diam 50%={:.2f} 90%={:.2f} 99%={:.2f} arcsec'.format(
        scat['ee_d50'], scat['ee_d90'], scat['ee_d99'])
    plt.plot(scat['ee_angle'], scat['ee_val'], '-r', label=label)

    plt.xlabel('Arcsec')
    plt.title('Encircled energy fraction')
    plt.xlim(0, 5)
    plt.grid()
    plt.legend(loc='lower left')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)


def make_axial_resid_plot(aoc, corr='X', filename=None):
    print 'Plotting axial residuals'

    plt.figure(12, figsize=(5, 3.5))
    plt.clf()

    displ = aoc.scatter['corr'][corr]['img']
    plt.plot(displ)
    plt.xlabel('Axial pixel')
    plt.ylabel('X residual (um)')
    plt.title('Axial strip residuals corrected on {}'.format(corr))
    plt.grid()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)


def make_imgs_plot(aoc, axis='X', corr='X', filename=None):
    nx = aoc.n_az
    ny = aoc.n_ax

    displ = aoc.displ[axis]['img']
    adj = aoc.adj[axis][corr]
    resid = aoc.resid[axis][corr]['img']
    coeffs = aoc.coeffs[corr].reshape(aoc.n_coeffs_ax, aoc.n_coeffs_az)

    n_clip = aoc.clip
    clipbox_x = [n_clip, nx - n_clip, nx - n_clip, n_clip, n_clip]
    clipbox_y = [n_clip, n_clip, ny - n_clip, ny - n_clip, n_clip]
    scatter_x = aoc.scatter['cols'] + n_clip

    plt.figure(10, figsize=(6, 11))
    plt.clf()

    ax = plt.subplot(4, 1, 1)
    vmin, vmax = np.percentile(np.hstack([displ['clip'], adj['clip']]),
                               [0.5, 99.5])
    plt.imshow(displ['full'], vmin=vmin, vmax=vmax)
    ax.axison = False
    ax.autoscale(enable=False)
    plt.plot(clipbox_x, clipbox_y, '-m')
    plt.title('Input distortion {}'.format(axis))
    plt.colorbar(fraction=0.07)

    ax = plt.subplot(4, 1, 2)
    plt.imshow(adj['full'], vmin=vmin, vmax=vmax)
    plt.title('Adjustment using {}'.format(corr))
    ax.axison = False
    ax.autoscale(enable=False)
    plt.plot(clipbox_x, clipbox_y, '-m')
    plt.colorbar(fraction=0.07)

    vmin, vmax = np.percentile(resid['clip'], [0.5, 99.5])
    ax = plt.subplot(4, 1, 3)
    plt.title('Residual')
    plt.imshow(resid['full'], vmin=vmin, vmax=vmax)
    ax.axison = False
    ax.autoscale(enable=False)
    plt.plot(clipbox_x, clipbox_y, '-m')
    plt.colorbar(fraction=0.07)
    for x in scatter_x:
        plt.plot([x, x], [n_clip, ny - n_clip], '-m')

    try:
        vmin, vmax = np.percentile(coeffs[3:-3, 3:-3], [0.5, 99.5])
    except ValueError:
        vmin, vmax = np.percentile(coeffs, [0.5, 99.5])
    ax = plt.subplot(4, 1, 4)
    plt.title('Coeffs')
    plt.imshow(coeffs, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.axison = False
    ax.autoscale(enable=False)
    plt.colorbar(fraction=0.07)

    if filename is not None:
        plt.savefig(filename)


def write_scatter_data(aoc, corr, scatter_type, filename):
    """Write image used in scatter intensity calculation.  This image consists
    of a small(ish) number of axial strips that might have been corrected for
    piston and tilt.
    """
    print 'Writing', filename
    if scatter_type == 'input':
        img = aoc.scatter['input']['img']
    else:
        img = aoc.scatter[scatter_type][corr]['img']
    np.savetxt(filename, img, fmt='%8.5f')


def write_imgs_data(aoc, axis='X', corr='X', filename=None):
    resid = aoc.resid[axis][corr]['img']['full']
    print 'Writing', filename
    with open(filename, 'w') as out:
        for i_az in xrange(aoc.n_az):
            for i_ax in xrange(aoc.n_ax):
                out.write('{:5d} {:5d} {:f}\n'
                          .format(i_az, i_ax, resid[i_ax, i_az]))


def make_report(aoc):
    template = jinja2.Template(open('report_template.html').read())
    src['id'] = '{}/{}'.format(aoc.case_id, aoc.subcase_id)
    if not os.path.exists(files['src_dir'].abs):
        os.makedirs(files['src_dir'].abs)
    subcases = []
    stats = ('mean', 'std')
    clips = ('full', 'clip')

    aoc.calc_adj()
    aoc.calc_stats()
    aoc.calc_scatter()

    ratios = AutoDict()
    for axis in aoc.displ_axes:
        for corr in aoc.corr_axes:
            for stat in stats:
                for clip in clips:
                    ratio = (aoc.displ[axis][stat][clip]
                             / aoc.resid[axis][corr][stat][clip])
                    ratios[axis][corr][stat][clip] = ratio

    for corr in aoc.corr_axes:
        src['corr'] = corr
        src['axis'] = 'X'
        for scatter_type in ('input', 'corr'):
            src['scatter_type'] = scatter_type
            write_scatter_data(aoc, corr, scatter_type,
                               files['img_scatter.dat'].abs)

        make_ee_plot(aoc, corr, files['enc_energy.png'].abs)
        make_scatter_plot(aoc, corr, files['scatter.png'].abs)
        make_axial_resid_plot(aoc, corr, files['axial_resid.png'].abs)

        for axis in aoc.displ_axes:
            src['axis'] = axis
            make_imgs_plot(aoc, axis, corr, files['img_corr.png'].abs)
            write_imgs_data(aoc, axis, corr, files['img_corr.dat'].abs)
            with Ska.File.chdir(files['src_dir'].abs):
                subcases.append(
                    {'img_corr_file': files['img_corr.png'].rel,
                     'scatter_file': files['scatter.png'].rel,
                     'enc_energy_file': files['enc_energy.png'].rel,
                     'axial_resid_file': files['axial_resid.png'].rel,
                     'axis': axis,
                     'corr': corr,
                     'ratios': ratios[axis][corr],
                     'displ': aoc.displ[axis],
                     'resid': aoc.resid[axis][corr],
                     })
    out = template.render(subcases=subcases,
                          aoc=aoc)
    with open(files['index.html'].abs, 'w') as f:
        f.write(out)

    # Save version of aoc without ifuncs, which are not interesting and big
    delattr(aoc, 'ifuncs')
    pickle.dump(aoc, open(files['aoc.pkl'].rel, 'w'), protocol=-1)


if __name__ == '__main__':
    args = get_args()
    displ_axes = args.displ_axes.split(',')
    corr_axes = args.corr_axes.split(',')
    aoc = AdjOpticsCase(case_id=args.case_id, subcase_id=args.subcase_id,
                        clip=args.clip, n_ss=args.ss, bias_mult=args.bias_mult,
                        piston_tilt=bool(args.piston_tilt),
                        n_strips=args.n_strips,
                        displ_axes=displ_axes,
                        corr_axes=corr_axes)

    make_report(aoc)
