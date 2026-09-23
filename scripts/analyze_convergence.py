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
import bisect
import csv
import glob
import os
import random
import re
import sys


# --------------------------------------------------------------------------- hypervolume

try:
    import numpy as np
except ImportError:  # the slow reference path still works without it
    np = None


def non_dominated(points):
    """Keeps the minimal points (every objective is minimized)."""
    out = []
    for a in points:
        if not any(b != a and all(x <= y for x, y in zip(b, a)) and any(x < y for x, y in zip(b, a))
                   for b in points):
            out.append(a)
    return out


def hypervolume_2d_slow(points, reference):
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


def hypervolume_3d_slow(points, reference):
    """Volume dominated by the points, slicing along the third objective."""
    ordered = sorted(points, key=lambda p: p[2])
    total = 0.0
    front = []
    for i, point in enumerate(ordered):
        front = insert_2d(front, (point[0], point[1]))
        low = point[2]
        # The slab must stop at the reference: a point worse than it in the third objective would
        # otherwise stretch the previous slab and inflate the volume.
        high = min(ordered[i + 1][2] if i + 1 < len(ordered) else reference[2], reference[2])
        if high <= low or low >= reference[2]:
            continue
        total += hypervolume_2d_slow(front, (reference[0], reference[1])) * (high - low)
    return total


def _staircase_area(xs, ys, reference):
    """Area of a staircase already sorted by x ascending and y descending."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    keep = (x < reference[0]) & (y < reference[1])
    x, y = x[keep], y[keep]
    if x.size == 0:
        return 0.0
    previous = np.empty_like(y)
    previous[0] = reference[1]
    previous[1:] = y[:-1]
    return float(np.sum((reference[0] - x) * (previous - y)))


def hypervolume_2d_fast(points, reference):
    data = np.asarray(sorted(set(points)), dtype=np.float64)
    if data.size == 0:
        return 0.0
    # Staircase: walking in x ascending order, keep the points that lower the best y seen so far.
    best = np.minimum.accumulate(data[:, 1])
    keep = np.empty(len(data), dtype=bool)
    keep[0] = True
    keep[1:] = best[1:] < best[:-1]
    return _staircase_area(data[keep, 0], data[keep, 1], reference)


def hypervolume_3d_fast(points, reference):
    """Same slicing as the reference version, but the staircase is kept between slabs and its area is
    recomputed only when a point actually changes it."""
    data = np.asarray(points, dtype=np.float64)
    data = data[np.argsort(data[:, 2], kind='stable')]
    z = data[:, 2]
    upper = np.empty_like(z)
    upper[:-1] = z[1:]
    upper[-1] = reference[2]
    np.minimum(upper, reference[2], out=upper)
    heights = upper - z

    total = 0.0
    area = 0.0
    xs, ys = [], []  # staircase, x ascending and y descending
    for i in range(len(data)):
        a, b = data[i, 0], data[i, 1]
        position = bisect.bisect_left(xs, a)
        dominated = position > 0 and ys[position - 1] <= b
        if not dominated:
            while position < len(xs) and ys[position] >= b:
                del xs[position]
                del ys[position]
            xs.insert(position, a)
            ys.insert(position, b)
            area = _staircase_area(xs, ys, reference)
        if heights[i] > 0 and z[i] < reference[2]:
            total += area * heights[i]
    return total


def hypervolume(points, reference):
    if not points:
        return 0.0
    if len(reference) == 2:
        return hypervolume_2d_fast(points, reference) if np is not None else \
            hypervolume_2d_slow(points, reference)
    if len(reference) == 3:
        return hypervolume_3d_fast(points, reference) if np is not None else \
            hypervolume_3d_slow(points, reference)
    raise ValueError('only 2 and 3 objectives are supported')


def self_test():
    """Small cases computed by hand, so a wrong hypervolume cannot pass unnoticed."""
    assert hypervolume([(1, 1)], (2, 2)) == 1.0
    assert hypervolume([(0, 1), (1, 0)], (2, 2)) == 3.0
    assert hypervolume([(0, 0), (1, 1)], (2, 2)) == 4.0
    assert hypervolume([(1, 1, 1)], (2, 2, 2)) == 1.0
    assert hypervolume([(0, 1, 1), (1, 0, 1)], (2, 2, 2)) == 3.0
    assert hypervolume([(0, 0, 0), (1, 1, 1)], (2, 2, 2)) == 8.0
    # A point outside the reference in one objective must not add anything, and must not stretch the
    # slab of the point before it.
    assert hypervolume([(0, 0, 0), (1, 1, 5)], (2, 2, 2)) == 8.0
    assert hypervolume([(1, 1, 5)], (2, 2, 2)) == 0.0
    # Covering a set can never lower the volume, which is what makes the curve usable.
    assert hypervolume([(0, 1, 1), (1, 0, 1)], (2, 2, 2)) <= hypervolume([(0, 0, 1), (1, 0, 1)], (2, 2, 2))
    if np is None:
        return
    # The vectorized version has to agree with the reference one, which is the version checked by hand.
    generator = random.Random(20260922)
    for objectives in (2, 3):
        for _ in range(20):
            reference = tuple([100] * objectives)
            points = [tuple(generator.randrange(0, 120) for _ in range(objectives))
                      for _ in range(generator.randrange(1, 40))]
            slow = hypervolume_2d_slow(points, reference) if objectives == 2 else \
                hypervolume_3d_slow(points, reference)
            fast = hypervolume(points, reference)
            assert abs(slow - fast) <= 1e-6 * max(1.0, abs(slow)), (objectives, points, slow, fast)


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


def minimal(points):
    """Non-dominated subset, computed by sweeping instead of comparing every pair."""
    ordered = sorted(set(points))
    kept = []
    for point in ordered:
        if not any(all(k <= p for k, p in zip(keeper, point)) for keeper in kept[-64:]) and \
                not any(all(k <= p for k, p in zip(keeper, point)) for keeper in kept):
            kept.append(point)
    return kept


def read_front_csv(path):
    """Reference front: a CSV with f1,f2[,f3], or a .PO / .KBP file of the instance.

    Those hold a 1-based permutation followed by its costs, so the split is the longest prefix that is a
    permutation of 1..k; the costs are far larger than any gene, so the rule is not ambiguous.
    """
    if path.lower().endswith(('.po', '.kbp')):
        points = []
        for line in open(path, encoding='utf-8'):
            values = [int(v) for v in line.split()]
            if not values:
                continue
            size = 0
            for k in range(len(values) - 1, 0, -1):
                if sorted(values[:k]) == list(range(1, k + 1)):
                    size = k
                    break
            points.append(tuple(values[size:]))
        return points
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        columns = [c for c in reader.fieldnames or [] if c.startswith('f')]
        return [tuple(int(row[c]) for c in columns) for row in reader]


def reference_point_of(front, margin=0.1):
    """Corner that bounds the reference front, with a margin so its own extremes count."""
    objectives = len(front[0])
    ideal = [min(p[o] for p in front) for o in range(objectives)]
    nadir = [max(p[o] for p in front) for o in range(objectives)]
    return tuple(nadir[o] + max(1, int(margin * (nadir[o] - ideal[o]))) for o in range(objectives))


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


# Work the hypervolume of a whole file is worth, counted as runs * samples * points^2. It sets how often
# the hypervolume is computed, so that a 3-objective trace with thousands of points per generation stays
# tractable. The value is calibrated against the vectorized implementation: about 3.4e7 units per second
# on the machine where it was measured, so this is roughly two minutes per file. With no numpy the
# reference implementation is two orders of magnitude slower, so pass --hv-every or --hv-runs there.
HYPERVOLUME_BUDGET = 4.0e9


def hypervolume_step(runs, samples, requested):
    """Recorded fronts between two hypervolume samples: 1 unless the fronts are too large."""
    if requested > 0:
        return requested
    points = [len(front) for data in runs.values() for front in data.values()]
    average = sum(points) / len(points) if points else 0.0
    cost = len(runs) * (samples + 1) * average * average
    return max(1, int(cost / HYPERVOLUME_BUDGET) + (1 if cost > HYPERVOLUME_BUDGET else 0))


def analyze(path, po_dir, patience, epsilon, out_dir, hv_every, hv_runs, reference_front, hv_window,
            cover_target):
    instance = instance_of(path)
    runs, objectives = read_trace(path)
    po = read_po(po_dir, instance) if (po_dir and objectives == 2) else None

    # A reference front makes the numbers absolute and comparable between files: the hypervolume is
    # reported as a share of the one that front dominates, and the reference point comes from it, so it
    # does not depend on how each run started. Without one, the corner of the first generation is used
    # and the curve can only be read inside its own file.
    front = reference_front or (list(po) if po else None)
    if front:
        reference = reference_point_of(front)
        reference_volume = hypervolume(minimal(front), reference)
    else:
        first = [p for data in runs.values() for p in data.get(0, [])]
        if not first:
            sys.exit('%s: there is no generation 0 in the trace' % path)
        reference = tuple(max(p[o] for p in first) + 1 for o in range(objectives))
        reference_volume = 0.0

    present = sorted({g for data in runs.values() for g in data})
    generations = present[-1]
    # The hypervolume is the expensive part, so it can be limited to the first runs; the rest of the
    # indicators always use every run. Fewer runs buy a finer curve at the same cost.
    measured = sorted(runs)[:hv_runs] if hv_runs > 0 else sorted(runs)
    granularity = min((b - a for a, b in zip(present, present[1:])), default=1)
    if hv_window > 0:
        # A window in generations, translated into recorded fronts, so every file uses the same one.
        step = max(1, int(round(hv_window / granularity)))
    else:
        step = hypervolume_step({r: runs[r] for r in measured}, len(present) - 1, hv_every)
    sampled = present[::step]
    if sampled[-1] != generations:
        sampled.append(generations)
    # Distance in generations between two hypervolume samples. With a subsampled trace, or with fronts
    # large enough to need a step, this is the real window the stagnation test looks at.
    spacing = min((b - a for a, b in zip(sampled, sampled[1:])), default=1)
    summary = {'instance': instance, 'file': os.path.basename(path), 'runs': len(runs),
               'generations': generations, 'objectives': objectives, 'hv_step': spacing,
               'hv_runs': len(measured), 'reference_volume': reference_volume}
    # Share of the reference front already found. It needs no window: a set intersection per recorded
    # generation. With an elitist survival a point that enters the front stays, so it does not go down.
    reference_points = set(minimal(front)) if front else set()
    stalls, finals, optima, coverages = [], [], [], []
    cover_ends, cover_reached = [], []
    curves = {}
    for run, data in sorted(runs.items()):
        fronts = {g: frozenset(points) for g, points in data.items()}
        if run in measured:
            curve = [hypervolume(sorted(fronts[g]), reference) for g in sampled]
            curves[run] = curve
            stalls.append(stall_generation(sampled, curve, patience, epsilon))
        finals.append(final_front_generation(fronts))
        if reference_points:
            share = [(g, len(reference_points & set(fronts[g])) / len(reference_points))
                     for g in sorted(fronts)]
            cover_ends.append(share[-1][1])
            hit = [g for g, value in share if value >= cover_target]
            cover_reached.append(hit[0] if hit else None)
        if po:
            found = [(g, len(po & set(fronts[g])) / len(po)) for g in sorted(fronts)]
            coverages.append(found[-1][1])
            reached = [g for g, value in found if value >= 1.0]
            optima.append(reached[0] if reached else None)

    summary['cover_end'] = (sum(cover_ends) / len(cover_ends)) if cover_ends else float('nan')
    hit = [g for g in cover_reached if g is not None]
    summary['t_cover'] = percentile(hit, 0.5) if hit else float('nan')
    summary['cover_runs'] = '%d/%d' % (len(hit), len(cover_reached)) if cover_reached else '-'
    ends = [curve[-1] for curve in curves.values()]
    summary['hv_end_share'] = (percentile(ends, 0.5) / reference_volume) if reference_volume > 0 else float('nan')
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
            writer.writerow(['generation', 'hypervolume_median', 'hypervolume_share_median',
                             'front_size_median'])
            for i, g in enumerate(sampled):
                values = [curves[run][i] for run in curves]
                divisor = reference_volume if reference_volume > 0 else None
                fractions = [curves[run][i] / (divisor if divisor else (final[run] or 1.0)) for run in curves]
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
    parser.add_argument('--reference-front', default='',
                        help='CSV (f1,f2[,f3]) or .PO file whose hypervolume the curves are reported '
                             'against. Without it the KC10 instances use their published front and the '
                             'rest fall back to the value each run reaches at its last generation, '
                             'which cannot be compared between files')
    parser.add_argument('--hv-window', type=int, default=0,
                        help='window of the stagnation test, in generations, applied to every file '
                             'whatever granularity its trace has. Use it to compare t_stall between '
                             'files; without it each file picks its own step from the cost')
    parser.add_argument('--cover-target', type=float, default=0.99,
                        help='share of the reference front that t_cover looks for (default 0.99)')
    parser.add_argument('--write-reference', default='',
                        help='write the non-dominated union of every point of the given traces to this '
                             'CSV and stop; that file is the best known front of the campaign')
    parser.add_argument('--hv-runs', type=int, default=0,
                        help='runs used for the hypervolume, the slow part (default: all). The other '
                             'indicators always use every run')
    parser.add_argument('--hv-every', type=int, default=0,
                        help='generations between two hypervolume samples (default: chosen from the size '
                             'of the fronts, so that a 3-objective trace stays tractable)')
    args = parser.parse_args()

    self_test()

    paths = []
    for pattern in args.traces:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    if args.write_reference:
        union = []
        for path in paths:
            data, _ = read_trace(path)
            # Only the last front of each run: the earlier ones are dominated by it, and reading every
            # generation of a large trace would not fit in memory.
            union.extend(p for run in data.values() for p in run[max(run)])
        front = minimal(union)
        with open(args.write_reference, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['f%d' % (o + 1) for o in range(len(front[0]))])
            writer.writerows(front)
        print('%d non-dominated points out of %d written to %s'
              % (len(front), len(set(union)), args.write_reference))
        return

    reference = read_front_csv(args.reference_front) if args.reference_front else None
    rows = [analyze(path, args.po_dir, args.patience, args.epsilon, args.out_dir, args.hv_every,
                    args.hv_runs, reference, args.hv_window, args.cover_target) for path in paths]

    head = ('%-16s %5s %6s %7s %11s %9s %11s %12s %11s %9s %8s %9s %8s' %
            ('instance', 'runs', 'gens', 'hv_gens', 't_stall_med', 't_stall_90', 't_final_med',
             'optimum_runs', 't_opt_med', 'coverage', 'hv_end', 'cover_end', 't_cover'))
    print(head)
    print('-' * len(head))
    for row in rows:
        known = row['coverage_final_mean'] == row['coverage_final_mean']  # false for nan: no .PO front
        share = row['hv_end_share']
        cover = row['cover_end']
        print('%-16s %5d %6d %7d %11.1f %9.1f %11.1f %12s %11s %9s %8s %9s %8s' % (
            row['instance'], row['runs'], row['generations'], row['hv_step'], row['t_stall_median'],
            row['t_stall_p90'], row['t_final_median'], row['optimum_runs'],
            '%.1f' % row['t_optimum_median'] if row['t_optimum_median'] == row['t_optimum_median'] else '-',
            '%.1f%%' % (100.0 * row['coverage_final_mean']) if known else '-',
            '%.2f%%' % (100.0 * share) if share == share else '-',
            '%.2f%%' % (100.0 * cover) if cover == cover else '-',
            '%.0f' % row['t_cover'] if row['t_cover'] == row['t_cover'] else '-'))
        if max(row['t_stall_median'], row['t_final_median']) >= row['generations']:
            print('%-16s   the run is still improving at the last generation: raise --iterations'
                  % ('(%s)' % row['instance']))
    print('\npatience = %d generations, epsilon = %g' % (args.patience, args.epsilon))
    for row in rows:
        if row['hv_step'] > args.patience:
            print('%-16s   the hypervolume is sampled every %d generations, so its stagnation test asks '
                  'for no growth over that window, not over --patience'
                  % ('(%s)' % row['instance'], row['hv_step']))
    if args.out_dir:
        print('per-generation curves in %s' % args.out_dir)


if __name__ == '__main__':
    main()
