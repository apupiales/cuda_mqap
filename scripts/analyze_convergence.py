# -*- coding: utf-8 -*-
"""
analyze_convergence.py

Reads the traces written by `cuda_mqap --trace` and answers, per instance, how many generations the
search needs: when it reaches the published optimal front, and when it stops improving.

    python scripts/analyze_convergence.py results/trace/*.csv --po-dir mQAPData

Each trace is a CSV with `run,generation,f1,f2[,f3]`: the distinct non-dominated solutions of every
generation of every run. The instance name is taken from the file name (the leading `KC<n>-<m>fl-<type>`).

Indicators, per run:

  hypervolume   Volume dominated by the front with respect to a reference point that is fixed for the
                whole file (the componentwise maximum of generation 0 over every run), so the curves of
                different generations and runs are comparable. The survival is elitist, so this value
                can only grow: a flat curve means the search really stopped.
  t_stall       First generation after which the hypervolume grows less than `--epsilon` (relative)
                during `--patience` generations. It is the generation where the run stagnates.
  t_final       First generation whose front already equals the last front of the run. It needs no
                reference data and says when nothing new was found again.
  coverage      Only when the instance has a published `.PO` front: share of its points already found.
  t_optimum     First generation with coverage 1.0, that is, when the optimal front was reached.

`t_stall`, `t_final` and `t_optimum` are random variables: every run stagnates at a different
generation, so the summary reports the median and the percentiles over the runs, not one number.

Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a
copy of the GNU General Public License along with this program. If not, see
<https://www.gnu.org/licenses/>.
"""
import argparse
import csv
import glob
import os
import re
import sys


# --------------------------------------------------------------------------- hypervolume

def non_dominated(points):
    """Keeps the minimal points (every objective is minimized)."""
    out = []
    for a in points:
        if not any(b != a and all(x <= y for x, y in zip(b, a)) and any(x < y for x, y in zip(b, a))
                   for b in points):
            out.append(a)
    return out


def hypervolume_2d(points, reference):
    """Area dominated by the points and bounded by the reference point."""
    total = 0.0
    previous = reference[1]
    for f1, f2 in sorted(points):
        if f1 >= reference[0] or f2 >= previous:
            continue
        total += (reference[0] - f1) * (previous - f2)
        previous = f2
    return total


def insert_2d(front, point):
    """Adds a point to a minimal set sorted by the first objective, dropping what it dominates."""
    if any(a <= point[0] and b <= point[1] for a, b in front):
        return front
    kept = [p for p in front if not (point[0] <= p[0] and point[1] <= p[1])]
    kept.append(point)
    kept.sort()
    return kept


def hypervolume_3d(points, reference):
    """Volume dominated by the points, slicing along the third objective.

    The projection of the points already processed is kept between slices, instead of being rebuilt,
    which is what makes the traces of the 3-objective instances tractable.
    """
    ordered = sorted(points, key=lambda p: p[2])
    total = 0.0
    front = []
    for i, point in enumerate(ordered):
        front = insert_2d(front, (point[0], point[1]))
        low = point[2]
        high = ordered[i + 1][2] if i + 1 < len(ordered) else reference[2]
        if high <= low or low >= reference[2]:
            continue
        total += hypervolume_2d(front, (reference[0], reference[1])) * (high - low)
    return total


def hypervolume(points, reference):
    if not points:
        return 0.0
    if len(reference) == 2:
        return hypervolume_2d(points, reference)
    if len(reference) == 3:
        return hypervolume_3d(points, reference)
    raise ValueError('only 2 and 3 objectives are supported')


def self_test():
    """Small cases computed by hand, so a wrong hypervolume cannot pass unnoticed."""
    assert hypervolume([(1, 1)], (2, 2)) == 1.0
    assert hypervolume([(0, 1), (1, 0)], (2, 2)) == 3.0
    assert hypervolume([(0, 0), (1, 1)], (2, 2)) == 4.0
    assert hypervolume([(1, 1, 1)], (2, 2, 2)) == 1.0
    assert hypervolume([(0, 1, 1), (1, 0, 1)], (2, 2, 2)) == 3.0
    assert hypervolume([(0, 0, 0), (1, 1, 1)], (2, 2, 2)) == 8.0


# --------------------------------------------------------------------------- data

def read_trace(path):
    """{run: {generation: [point, ...]}} and the number of objectives."""
    runs = {}
    objectives = 0
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        columns = [c for c in reader.fieldnames or [] if c.startswith('f')]
        objectives = len(columns)
        if objectives < 2:
            sys.exit('%s: the trace has no objective columns' % path)
        for row in reader:
            run = int(row['run'])
            generation = int(row['generation'])
            point = tuple(int(row[c]) for c in columns)
            runs.setdefault(run, {}).setdefault(generation, []).append(point)
    return runs, objectives


def read_po(po_dir, instance):
    path = os.path.join(po_dir, instance + '.PO')
    if not os.path.exists(path):
        return None
    points = []
    for line in open(path, encoding='utf-8'):
        values = [int(v) for v in line.split()]
        if values:
            points.append(tuple(values[-2:]))
    return set(points)


def instance_of(path):
    match = re.search(r'(KC\d+-\d+fl-[A-Za-z0-9]+)', os.path.basename(path))
    return match.group(1) if match else os.path.splitext(os.path.basename(path))[0]


# --------------------------------------------------------------------------- indicators

def stall_generation(sampled, curve, patience, epsilon):
    """First sampled generation after which the hypervolume grows less than epsilon during `patience`.

    Returns the last generation when the window does not fit any more, which means the run was still
    improving at the end of the trace and the cap of --iterations was too low.
    """
    for i, generation in enumerate(sampled):
        end = i
        while end + 1 < len(sampled) and sampled[end] - generation < patience:
            end += 1
        if sampled[end] - generation < patience:
            break
        reference = curve[i] if curve[i] > 0 else 1.0
        if (curve[end] - curve[i]) / reference <= epsilon:
            return generation
    return sampled[-1]


def final_front_generation(fronts):
    """First generation whose front is already the last one."""
    last = max(fronts)
    target = fronts[last]
    for g in sorted(fronts):
        if fronts[g] == target:
            return g
    return last


def percentile(values, fraction):
    if not values:
        return float('nan')
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


# Simple operations the hypervolume of one front is worth; it sets how often it is computed so that a
# trace of a 3-objective instance (hundreds of points per generation) stays tractable.
HYPERVOLUME_BUDGET = 2.0e8


def hypervolume_step(runs, samples, requested):
    """Recorded fronts between two hypervolume samples: 1 unless the fronts are too large."""
    if requested > 0:
        return requested
    points = [len(front) for data in runs.values() for front in data.values()]
    average = sum(points) / len(points) if points else 0.0
    cost = len(runs) * (samples + 1) * average * average
    return max(1, int(cost / HYPERVOLUME_BUDGET) + (1 if cost > HYPERVOLUME_BUDGET else 0))


def analyze(path, po_dir, patience, epsilon, out_dir, hv_every):
    instance = instance_of(path)
    runs, objectives = read_trace(path)
    po = read_po(po_dir, instance) if (po_dir and objectives == 2) else None

    # Reference point common to the whole file: the worst corner of the first generation.
    first = [p for data in runs.values() for p in data.get(0, [])]
    if not first:
        sys.exit('%s: there is no generation 0 in the trace' % path)
    reference = tuple(max(p[o] for p in first) + 1 for o in range(objectives))

    present = sorted({g for data in runs.values() for g in data})
    generations = present[-1]
    step = hypervolume_step(runs, len(present) - 1, hv_every)
    sampled = present[::step]
    if sampled[-1] != generations:
        sampled.append(generations)
    summary = {'instance': instance, 'file': os.path.basename(path), 'runs': len(runs),
               'generations': generations, 'objectives': objectives, 'hv_step': step}
    stalls, finals, optima, coverages = [], [], [], []
    curves = {}
    for run, data in sorted(runs.items()):
        fronts = {g: frozenset(points) for g, points in data.items()}
        curve = [hypervolume(sorted(fronts[g]), reference) for g in sampled]
        curves[run] = curve
        stalls.append(stall_generation(sampled, curve, patience, epsilon))
        finals.append(final_front_generation(fronts))
        if po:
            found = [(g, len(po & set(fronts[g])) / len(po)) for g in sorted(fronts)]
            coverages.append(found[-1][1])
            reached = [g for g, value in found if value >= 1.0]
            optima.append(reached[0] if reached else None)

    summary['t_stall_median'] = percentile(stalls, 0.5)
    summary['t_stall_p90'] = percentile(stalls, 0.9)
    summary['t_stall_max'] = max(stalls)
    summary['t_final_median'] = percentile(finals, 0.5)
    summary['t_final_p90'] = percentile(finals, 0.9)
    if po:
        solved = [g for g in optima if g is not None]
        summary['optimum_runs'] = '%d/%d' % (len(solved), len(optima))
        summary['t_optimum_median'] = percentile(solved, 0.5) if solved else float('nan')
        summary['t_optimum_p90'] = percentile(solved, 0.9) if solved else float('nan')
        summary['coverage_final_mean'] = sum(coverages) / len(coverages)
    else:
        summary['optimum_runs'] = '-'
        summary['t_optimum_median'] = float('nan')
        summary['t_optimum_p90'] = float('nan')
        summary['coverage_final_mean'] = float('nan')

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        curve_path = os.path.join(out_dir, instance + '_curve.csv')
        final = {run: curve[-1] for run, curve in curves.items()}
        with open(curve_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['generation', 'hypervolume_median', 'hypervolume_fraction_of_final_median',
                             'front_size_median'])
            for i, g in enumerate(sampled):
                values = [curves[run][i] for run in curves]
                fractions = [curves[run][i] / final[run] if final[run] else 0.0 for run in curves]
                sizes = [len(runs[run].get(g, [])) for run in curves]
                writer.writerow([g, '%.6g' % percentile(values, 0.5), '%.6f' % percentile(fractions, 0.5),
                                 '%.1f' % percentile(sizes, 0.5)])
        summary['curve'] = curve_path
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    parser.add_argument('traces', nargs='+', help='CSV files written by --trace (globs allowed)')
    parser.add_argument('--po-dir', default='mQAPData', help='directory with the .PO fronts (default mQAPData)')
    parser.add_argument('--patience', type=int, default=20, help='generations without improvement (default 20)')
    parser.add_argument('--epsilon', type=float, default=1e-4,
                        help='relative hypervolume growth counted as no improvement (default 1e-4)')
    parser.add_argument('--out-dir', default='', help='directory for the per-generation curves')
    parser.add_argument('--hv-every', type=int, default=0,
                        help='generations between two hypervolume samples (default: chosen from the size '
                             'of the fronts, so that a 3-objective trace stays tractable)')
    args = parser.parse_args()

    self_test()

    paths = []
    for pattern in args.traces:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    rows = [analyze(path, args.po_dir, args.patience, args.epsilon, args.out_dir, args.hv_every)
            for path in paths]

    head = ('%-16s %5s %6s %7s %11s %9s %11s %9s %12s %11s %9s' %
            ('instance', 'runs', 'gens', 'hv_step', 't_stall_med', 't_stall_90', 't_final_med',
             't_final_90', 'optimum_runs', 't_opt_med', 'coverage'))
    print(head)
    print('-' * len(head))
    for row in rows:
        known = row['coverage_final_mean'] == row['coverage_final_mean']  # false for nan: no .PO front
        print('%-16s %5d %6d %7d %11.1f %9.1f %11.1f %9.1f %12s %11s %9s' % (
            row['instance'], row['runs'], row['generations'], row['hv_step'], row['t_stall_median'],
            row['t_stall_p90'], row['t_final_median'], row['t_final_p90'], row['optimum_runs'],
            '%.1f' % row['t_optimum_median'] if row['t_optimum_median'] == row['t_optimum_median'] else '-',
            '%.1f%%' % (100.0 * row['coverage_final_mean']) if known else '-'))
        if max(row['t_stall_median'], row['t_final_median']) >= row['generations']:
            print('%-16s   the run is still improving at the last generation: raise --iterations'
                  % ('(%s)' % row['instance']))
    print('\npatience = %d generations, epsilon = %g' % (args.patience, args.epsilon))
    if args.out_dir:
        print('per-generation curves in %s' % args.out_dir)


if __name__ == '__main__':
    main()
