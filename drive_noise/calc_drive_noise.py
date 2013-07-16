#!/usr/bin/env python

"""Run calc_adj for the exemplar data"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
import os
import socket

sys.path.insert(0, '..')
import calc_mtf
import ifunc


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Calc drive noise sensitivity')
    parser.add_argument('--case-id',
                        type=str,
                        default='10+0_baseline',
                        help='Case ID')
    parser.add_argument('--n-sim',
                        type=int,
                        default=2,
                        help='Number of simulations')
    parser.add_argument('--n-strips',
                        type=int, default=9,
                        help='Number of strips for scatter calculation')
    parser.add_argument('--displ-axes',
                        type=str, default="X",
                        help='Displacement axes to compute')
    parser.add_argument('--corr-axes',
                        type=str, default="X",
                        help='Correction axes to compute')
    parser.add_argument('--drive-ref',
                        type=float, default=5.0,
                        help='Drive reference (units of 100 ppm strain, default=5.0)')
    parser.add_argument('--n-proc',
                        type=int, default=0,
                        help='Number of processors (default=0 => no multiprocess)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    # Compute solution for zero-noise case
    error = calc_mtf.displ_exemplar(ampl=1.0, apply_10_0=True)
    bias = calc_mtf.displ_flat(0.4)
    out = calc_mtf.calc_plot_adj(row_clip=2, col_clip=2, ax_clip=40, az_clip=80,
                                 bias=bias, error=error, plot_file=None, max_iter=0, nnls=True)

    coeffs = out['coeffs_img'].flatten()
    noises = np.random.uniform(0, 0.05, args.n_sim)
    noises[0] = 0

    pid = os.getpid()
    hostname = socket.gethostname()
    outfile = os.path.join('drive_noise-{}-{}.dat'.format(hostname, pid))

    for noise in noises:
        print 'Drive noise = {} (fraction of {} drive voltage)'.format(
            noise, args.drive_ref)
        drive_noise = noise * args.drive_ref

        coeffs_noisy = coeffs.copy()
        if noise > 0:
            ok = coeffs > 0  # Only add noise to actuators that are being driven
            coeffs_noisy[ok] += np.random.normal(0.0, scale=drive_noise,
                                                 size=len(coeffs))[ok]
            coeffs_noisy = np.abs(coeffs_noisy)

        adj_displ = ifunc.calc_adj_displ(calc_mtf.ifuncs, coeffs_noisy)
        resid = bias.vals + error.vals - adj_displ
        resid_stats = ifunc.calc_scatter(resid,
                                         ax_clip=out['ax_clip'], az_clip=out['az_clip'],
                                         n_proc=args.n_proc, n_strips=args.n_strips)
        print 'Corr HPD={:.2f} RMSD={:.2f}'.format(resid_stats['hpd'],
                                                   resid_stats['rmsd'])

        hpd = resid_stats['hpd']
        rmsd = resid_stats['rmsd']

        with open(outfile, 'a') as f:
            print >>f, ' '.join(str(x) for x in
                                (noise, hpd, rmsd))
