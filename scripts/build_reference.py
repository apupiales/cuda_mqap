# -*- coding: utf-8 -*-
"""
build_reference.py

Builds the best known front of an instance from result files and writes it into a reference version.

    python scripts/build_reference.py KC20-2fl-2rl "results/grid/KC20-2fl-2rl/*.txt" --into v0.3
    python scripts/build_reference.py KC20-2fl-1rl "results/grid/KC20-2fl-1rl/*.txt" --into v0.3 --from v0.2

The front is the non-dominated union of every solution of every run, optionally starting from the front of
an earlier version. The objective vectors decide which solutions survive, and only the survivors have
their permutation verified by recomputing its cost against the instance, which is what makes building a
reference from hundreds of runs affordable.

Unlike `compare_versions.py --update-reference`, this does not need a reference to exist already: it is
how an instance gets its first one. An instance with a published optimum (`mQAPData/<instance>.PO`) is
refused, because that front is proven optimal and is not versioned here.

Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from analyze_convergence import minimal  # noqa: E402
from compare_versions import (cost_of, next_version, permutations_of, read_front,  # noqa: E402
                              read_runs, reference_versions, write_version)
from prepare_original import load_instance  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[4])
    parser.add_argument('instance', help='instance name, e.g. KC20-2fl-2rl')
    parser.add_argument('patterns', nargs='+', metavar='PATH',
                        help='result files, globs allowed')
    parser.add_argument('--into', required=True, metavar='VERSION',
                        help='reference version to write into; created when it does not exist')
    parser.add_argument('--from', dest='source', metavar='VERSION',
                        help='version whose front this one starts from, and whose other instances are '
                             'copied when the target version is created (default: the latest, if any)')
    args = parser.parse_args()

    if os.path.exists(os.path.join(ROOT, 'mQAPData', args.instance + '.PO')):
        sys.exit('%s has a published optimum in mQAPData; that front is not versioned here' % args.instance)

    existing = reference_versions()
    if args.into not in existing and args.into != next_version():
        sys.exit('--into %s is neither an existing version nor the next one (%s)'
                 % (args.into, next_version()))
    source = args.source if args.source is not None else (existing[-1] if existing else None)
    if source == args.into:
        source = existing[-2] if len(existing) > 1 else None

    known = {}
    if source:
        path = os.path.join(ROOT, 'reference', source, args.instance + '.KBP')
        if os.path.exists(path):
            known = {costs: permutation for permutation, costs in read_front(path)}
            print('starting from reference/%s/%s.KBP, %d points' % (source, args.instance, len(known)))
        else:
            print('reference/%s has no front for %s: building it from the runs alone' % (source, args.instance))

    files = sorted({name for pattern in args.patterns for name in glob.glob(pattern)})
    if not files:
        sys.exit('no file matches %s' % ' '.join(args.patterns))

    # The objective vectors first: deciding who survives needs no permutation.
    candidates = dict(known)                      # costs -> permutation (1-based) or None when from a run
    keys = {}                                     # costs -> permutation key written by the program
    runs = 0
    for name in files:
        for run in read_runs(name):
            runs += 1
            for key, costs in run:
                if costs not in candidates:
                    candidates[costs] = None
                    keys[costs] = key
    print('%d runs in %d files, %d distinct objective vectors' % (runs, len(files), len(candidates)))

    front = minimal(list(candidates))
    print('%d of them are non-dominated' % len(front))

    n, _objectives, distances, flows = load_instance(os.path.join(ROOT, 'mQAPData', args.instance + '.dat'))
    permutations = {}
    for costs in front:
        if candidates[costs] is not None:
            permutations[costs] = candidates[costs]    # already verified when its version was written
            continue
        matches = [p for p in permutations_of(keys[costs], n)
                   if tuple(cost_of(n, distances, f, p) for f in flows) == costs]
        if not matches:
            sys.exit('%s: the cost of %s does not reproduce against the instance' % (args.instance, keys[costs]))
        permutations[costs] = [v + 1 for v in matches[0]]
    print('%d permutations verified against the instance' % len(front))

    name = write_version(source, args.instance, set(front), permutations,
                         {os.path.basename(f) for f in files[:4]}, args.into if args.into in existing else None)
    print('reference/%s/%s.KBP written with %d points (was %d)'
          % (name, args.instance, len(front), len(known)))


if __name__ == '__main__':
    main()
