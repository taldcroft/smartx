#!/usr/bin/env python

"""Run calc_adj for the exemplar data"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '..')
from adj_opt_case import AdjOpticsCase


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Calc drive noise sensitivity')
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
                        type=int, default=9,
                        help='Number of strips for scatter calculation')
    parser.add_argument('--piston-tilt',
                        type=int, default=0,
                        help='Sub-sampling')
    parser.add_argument('--displ-axes',
                        type=str, default="X",
                        help='Displacement axes to compute')
    parser.add_argument('--corr-axes',
                        type=str, default="X",
                        help='Correction axes to compute')
    parser.add_argument('--drive-ref',
                        type=str, default="median",
                        help='Drive reference (median|max)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    corr = 'X'
    args = get_args()
    displ_axes = args.displ_axes.split(',')
    corr_axes = args.corr_axes.split(',')
    aoc = AdjOpticsCase(case_id=args.case_id, subcase_id=args.subcase_id,
                        clip=args.clip, n_ss=args.ss,
                        piston_tilt=bool(args.piston_tilt),
                        n_strips=args.n_strips,
                        displ_axes=displ_axes,
                        corr_axes=corr_axes)

    # Compute stats for zero-noise case
    aoc.calc_adj()
    aoc.calc_stats()
    aoc.calc_scatter()

    coeffs = aoc.coeffs['X']
    drive_ref = getattr(np, args.drive_ref)(abs(coeffs))
    # noises = (0.00001, 0.001, 0.002, 0.005, 0.01, 0.02,
    #           0.05, 0.1, 0.15, 0.2)
    noises = np.linspace(0, 0.2, 11)

    hpds = []
    for noise in noises:
        print 'Drive noise = {} (fraction of {} drive voltage)'.format(
            noise, args.drive_ref)
        drive_noise = noise * drive_ref

        if noise > 0:
            coeffs_noisy = coeffs + np.random.normal(0.0, scale=drive_noise,
                                                     size=len(coeffs))

            aoc.coeffs[corr] = coeffs_noisy
            aoc.calc_adj()
            aoc.calc_stats()
            aoc.calc_scatter()

        scat = aoc.scatter['corr'][corr]
        print 'Corr HPD={:.2f} RMSD={:.2f}'.format(scat['hpd'],
                                                   scat['rmsd'])

        hpds.append(scat['hpd'])

    hpds = np.array(hpds) / hpds[0]
    plt.figure(figsize=(5, 3.5))
    plt.clf()
    plt.plot(noises, hpds, '-b')
    plt.plot(noises, hpds, 'ob')
    plt.xlabel('Drive noise relative to {} drive level'
               .format(args.drive_ref))
    # plt.title('HPD dependence on drive noise')
    plt.ylabel('HPD relative to zero-noise case')
    plt.ylim(0, None)
    plt.xlim(-0.02, 0.22)
    plt.tight_layout()
    plt.grid()
    plt.savefig('drive_noise.png')
