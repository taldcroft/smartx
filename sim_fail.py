#!/usr/bin/env python

import os
import socket

import numpy as np
import pyyaks.context

from adj_opt_case import AdjOpticsCase

src = pyyaks.context.ContextDict('src')
files = pyyaks.context.ContextDict('files', basedir='reports')
files.update({'src_dir': '{{src.id}}'})


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Simulate actuator failures')
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
                        help='Number of strips for scatter calc')
    parser.add_argument('--piston-tilt',
                        type=int, default=0,
                        help='Sub-sampling')
    parser.add_argument('--max-sim',
                        type=int,
                        default=100000,
                        help='Max number of simulations')
    parser.add_argument('--max-fail',
                        type=int,
                        default=20,
                        help='Max number of actuator failures')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    aoc = AdjOpticsCase(case_id=args.case_id, subcase_id=args.subcase_id,
                        clip=args.clip, n_ss=args.ss,
                        piston_tilt=bool(args.piston_tilt),
                        displ_axes=['X'], corr_axes=['X'],
                        n_proc=None)

    src['id'] = '{}/{}'.format(aoc.case_id, aoc.subcase_id)
    if not os.path.exists(files['src_dir'].abs):
        os.makedirs(files['src_dir'].abs)

    pid = os.getpid()
    hostname = socket.gethostname()
    outfile = os.path.join(files['src_dir'].abs,
                           'act_fail-{}-{}.dat'.format(hostname, pid))

    # Make an ordered list 0 .. n_acts-1 for shuffling later.
    ifuncs_x = aoc.ifuncs['X'].copy()
    ifuncs_ry = aoc.ifuncs['RY'].copy()
    n_acts = ifuncs_x.shape[0]
    i_acts = np.arange(n_acts)

    for i in range(args.max_sim):
        # Force some actuators to "fail" by taking them out of the optimization
        if args.max_fail > 0:
            n_fail = np.random.randint(args.max_fail) + 1
            np.random.shuffle(i_acts)
            ok = np.ones(n_acts, dtype=bool)
            i_fail = i_acts[:n_fail]
            ok[i_fail] = False
            aoc.ifuncs['X'] = ifuncs_x[ok]
            aoc.ifuncs['RY'] = ifuncs_ry[ok]
        else:
            n_fail = 0
            i_fail = [-1]

        aoc.calc_adj()
        aoc.calc_stats()
        aoc.calc_scatter(n_strips=args.n_strips, calc_input=False)

        hpd = aoc.scatter['corr']['X']['hpd']
        rmsd = aoc.scatter['corr']['X']['rmsd']

        with open(outfile, 'a') as f:
            i_fail_str = ','.join(str(x) for x in i_fail)
            print >>f, ' '.join(str(x) for x in
                                (n_fail, hpd, rmsd, i_fail_str))


if __name__ == '__main__':
    main()
