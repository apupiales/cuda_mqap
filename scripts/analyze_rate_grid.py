# -*- coding: utf-8 -*-
"""
analyze_rate_grid.py

Finds the best configuration of population and greedy 2-opt rate for each instance of a grid.

    python scripts/analyze_rate_grid.py results/grid --out results/grid/best.json

Reads the cells scripts/run_rate_grid.ps1 wrote, `<config>_P<population>.txt` per instance, and scores
every one of them against the reference front of the instance: the published optimum when it has one,
otherwise the `.KBP` of a reference version (the latest unless `--reference` names another).

The ranking metric is **coverage**, the share of the points of the reference front a run finds, which is
what separates configurations once the hypervolume has saturated, and which costs nothing to compute on a
front of thousands of points. The hypervolume is reported for the finalists only — the best cell and the
best one with the default configuration — because on a 3-objective front of tens of thousands of points it
would cost hours per cell.

The difference between the best cell and the best default cell is tested with the Mann-Whitney U test
(two-sided) over their coverage samples, so the table says whether tuning the rate bought anything beyond
tuning the population.

Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later
"""
import argparse
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from analyze_convergence import hypervolume, minimal, reference_point_of  # noqa: E402
from compare_versions import read_front, read_runs, reference_path  # noqa: E402

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    mannwhitneyu = None

CELL = re.compile(r'^(rate(\d+)p(\d+))_P(\d+)\.txt$')
DEFAULT = 'rate100p1'
HYPERVOLUME_LIMIT = 5000      # reference points above which the hypervolume is not worth its cost


def cells_of(directory):
    """[(configuration, rate, period, population, path)] of one instance directory."""
    out = []
    for name in sorted(os.listdir(directory)):
        match = CELL.match(name)
        if match:
            out.append((match.group(1), int(match.group(2)) / 100.0, int(match.group(3)),
                        int(match.group(4)), os.path.join(directory, name)))
    return out


def pvalue(a, b):
    if mannwhitneyu is None or a == b:
        return None
    try:
        return float(mannwhitneyu(a, b, alternative='two-sided').pvalue)
    except ValueError:
        return None


def analyze(instance, directory, version, verbose):
    path = reference_path(instance, version)
    front = read_front(path)
    reference = {costs for _permutation, costs in front}
    point = reference_point_of(sorted(reference))
    shown = os.path.relpath(path, ROOT).replace(os.sep, '/')

    measured = []
    for configuration, rate, period, population, file in cells_of(directory):
        runs = read_runs(file)
        covers = [100.0 * len(reference & {costs for _key, costs in run}) / len(reference) for run in runs]
        if not covers:
            continue
        measured.append({'configuration': configuration, 'rate': rate, 'period': period,
                         'population': population, 'runs': len(covers),
                         'coverage': statistics.mean(covers), 'coverage_best': max(covers),
                         'samples': covers, 'path': file})
    if not measured:
        return None

    # Quality first; among cells that reach the same coverage, the cheapest one, which is the smaller
    # population and then the smaller share of the population given to the local search.
    measured.sort(key=lambda row: (-row['coverage'], -row['coverage_best'],
                                   row['population'], row['rate'], -row['period']))
    best = measured[0]
    defaults = [row for row in measured if row['configuration'] == DEFAULT]
    best_default = defaults[0] if defaults else None

    # The hypervolume of the finalists only, and only when the front is small enough to be worth it.
    total = None
    if len(reference) <= HYPERVOLUME_LIMIT:
        total = hypervolume(minimal(sorted(reference)), point)
        for row in {id(best): best, **({id(best_default): best_default} if best_default else {})}.values():
            volumes = [100.0 * hypervolume(minimal([c for _k, c in run]), point) / total
                       for run in read_runs(row['path'])]
            row['hypervolume'] = statistics.mean(volumes)

    # Did any run find something the reference front does not dominate? Only asked of an instance with a
    # published optimum, where the answer must be none and so checks the whole pipeline: for the rest the
    # reference was built from these very runs by build_reference.py, so by construction nothing is beyond
    # it, and the union of every run would be far too large to filter again here.
    beyond = None
    if path.endswith('.PO'):
        seen = {costs for _configuration, _r, _p, _pop, file in cells_of(directory)
                for run in read_runs(file) for _key, costs in run}
        survivors = set(minimal(sorted(reference | seen)))
        beyond = len(survivors - reference)

    out = {
        'instance': instance,
        'reference': shown,
        'reference_points': len(reference),
        'beyond_reference': beyond,
        'best': {k: v for k, v in best.items() if k not in ('samples', 'path')},
        'best_default': ({k: v for k, v in best_default.items() if k not in ('samples', 'path')}
                         if best_default else None),
        'p_best_vs_default': pvalue(best['samples'], best_default['samples']) if best_default else None,
        'cells': [{k: v for k, v in row.items() if k not in ('samples', 'path')} for row in measured],
    }
    if verbose:
        print('\n%s, reference %s with %d points' % (instance, shown, len(reference)))
        for row in measured[:5]:
            print('  %-10s P=%-6d coverage %6.2f %% (best run %6.2f %%)'
                  % (row['configuration'], row['population'], row['coverage'], row['coverage_best']))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[4])
    parser.add_argument('grid', help='directory with one subdirectory per instance')
    parser.add_argument('--reference', metavar='VERSION', help='reference version (default: the latest)')
    parser.add_argument('--out', default='', help='JSON file with the full ranking')
    parser.add_argument('--instances', default='', help='comma-separated subset')
    parser.add_argument('--quiet', action='store_true', help='only the summary table')
    args = parser.parse_args()

    wanted = [name for name in args.instances.split(',') if name] if args.instances else None
    results = []
    for instance in sorted(os.listdir(args.grid)):
        directory = os.path.join(args.grid, instance)
        if not os.path.isdir(directory) or (wanted and instance not in wanted):
            continue
        row = analyze(instance, directory, args.reference, not args.quiet)
        if row:
            results.append(row)

    print('\n%-15s %-10s %-8s %9s   %-10s %-8s %9s   %-9s %s'
          % ('instance', 'best', 'P', 'coverage', 'default', 'P', 'coverage', 'p', 'beyond'))
    for row in results:
        best, default = row['best'], row['best_default']
        print('%-15s %-10s %-8d %8.2f %%   %-10s %-8s %8s   %-9s %s'
              % (row['instance'], best['configuration'], best['population'], best['coverage'],
                 default['configuration'] if default else '-',
                 default['population'] if default else '-',
                 '%.2f %%' % default['coverage'] if default else '-',
                 '%.3g' % row['p_best_vs_default'] if row['p_best_vs_default'] is not None else '-',
                 '-' if row['beyond_reference'] is None else row['beyond_reference']))

    if args.out:
        with open(args.out, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(results, handle, indent=1)
        print('\nwrote %s' % args.out)


if __name__ == '__main__':
    main()
