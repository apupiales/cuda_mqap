# -*- coding: utf-8 -*-
"""
compare_versions.py

Compares the fronts that several versions or configurations reach on the same instance, over many runs,
and says whether the difference is statistically significant.

    python scripts/compare_versions.py KC20-2fl-1uni original=out/orig/result_*.txt this=out/new.txt

Each argument after the instance is `label=path` of a result file, the format both versions write: one
`{ 'permutation': [f1, f2[, f3]], ... },` block per run. The first one is the baseline every other label
is tested against.

Two indicators per run, both against the same reference front, so the numbers of different versions,
populations and runs are comparable:

  hypervolume  Share of the volume the reference front dominates with respect to a reference point
               derived from the reference front itself. It saturates quickly: a front with few points
               in the right place already dominates almost the whole volume.
  coverage     Share of the points of the reference front the run actually found. This is what separates
               configurations once the hypervolume is close to 100 %.

The reference front is `mQAPData/<instance>.PO` when the instance has a published optimum, and
`reference/<version>/<instance>.KBP`, the best front known to this project, otherwise. The version is the
latest directory of `reference/` unless `--reference` names another one, and the report says which one was
used, because a share of hypervolume or of coverage only means something next to the front it came from.

A run can contain a solution the reference front does not dominate; those are reported, and with
`--update-reference` they are added, since a reference front is the best front known whoever found it.
Every addition is verified by recomputing the cost of its permutation against the instance. A published
version is never edited: the addition goes into the next version directory, which starts as a copy of the
one it came from, and the script prints the factors that convert a share published against the previous
version into the new scale. See reference/README.md.

The difference against the baseline is tested with the Mann-Whitney U test (two-sided), which compares
distributions without assuming normality, as is usual when comparing stochastic optimizers. It needs
scipy; without it the script still reports mean, standard deviation and best run.

Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later
"""
import argparse
import datetime
import glob
import json
import math
import os
import re
import shutil
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from analyze_convergence import hypervolume, minimal, reference_point_of  # noqa: E402
from prepare_original import load_instance  # noqa: E402

try:
    from scipy.stats import mannwhitneyu
except ImportError:  # the script still reports the descriptive statistics
    mannwhitneyu = None


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and a != b


def read_runs(path):
    """[(permutation key, objective vector)] per run block of a result file."""
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    runs = []
    for block in re.findall(r'\{(.*?)\},?\s*(?=\{|$)', text, re.S):
        rows = [(key, tuple(int(v) for v in values.split(',')))
                for key, values in re.findall(r"'(\d+)'\s*:\s*\[([\d,\s]+)\]", block)]
        if rows:
            runs.append(rows)
    return runs


def read_front(path):
    """[(permutation 1-based, objective vector)] of a .PO or .KBP file."""
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = [int(v) for v in line.split()]
            if not values:
                continue
            # The permutation is the longest prefix that is a permutation of 1..k; the rest are the costs.
            size = next((k for k in range(len(values), 0, -1)
                         if sorted(values[:k]) == list(range(1, k + 1))), 0)
            if not size:
                sys.exit('%s: this line is neither a permutation nor costs: %s' % (path, line.strip()))
            out.append((values[:size], tuple(values[size:])))
    return out


def cost_of(n, distances, flow, permutation):
    total = 0
    for i in range(n):
        row, pi = i * n, permutation[i] * n
        for j in range(n):
            f = flow[row + j]
            if f:
                total += f * distances[pi + permutation[j]]
    return total


def permutations_of(key, n):
    """Every way of reading a result key as a permutation of 0..n-1.

    The program writes the genes concatenated without a separator, which is ambiguous for n > 10, so the
    caller keeps the reading whose recomputed cost matches the fitness written next to it.
    """
    width = 2 if n > 10 else 1
    out = []

    def walk(pos, used, acc):
        if pos == len(key):
            if len(acc) == n:
                out.append(list(acc))
            return
        for w in range(1, width + 1):
            if pos + w > len(key):
                break
            piece = key[pos:pos + w]
            if w > 1 and piece[0] == '0':
                continue  # the genes are written without leading zeros
            value = int(piece)
            if value < n and value not in used:
                used.add(value)
                acc.append(value)
                walk(pos + w, used, acc)
                acc.pop()
                used.discard(value)

    walk(0, set(), [])
    return out


def version_key(name):
    """Sorts v0.2 before v0.10, which a plain string sort does not."""
    return [int(piece) for piece in re.findall(r'\d+', name)] or [0]


def reference_versions():
    """The reference version directories, oldest first."""
    root = os.path.join(ROOT, 'reference')
    if not os.path.isdir(root):
        return []
    found = [name for name in os.listdir(root)
             if name.startswith('v') and os.path.isdir(os.path.join(root, name))]
    return sorted(found, key=version_key)


def next_version():
    """The name after the latest one: v0.2 -> v0.3, and v0.1 when there is none."""
    existing = reference_versions()
    if not existing:
        return 'v0.1'
    pieces = version_key(existing[-1])
    pieces[-1] += 1
    return 'v' + '.'.join(str(piece) for piece in pieces)


def write_version(source, instance, front, permutations, contributors, into=None):
    """Writes a reference version: the updated instance, a copy of the rest, and its summary.

    A published version is never edited, so by default this creates the next directory and refuses to
    touch one that already exists. `into` names an open version instead, which is how the several
    instances of one experiment land in the same version. Returns its name.
    """
    name = into or next_version()
    target = os.path.join(ROOT, 'reference', name)
    if into:
        if not os.path.isdir(target):
            sys.exit('reference/%s does not exist; drop --into to create the next version' % name)
    elif os.path.isdir(target):
        sys.exit('reference/%s already exists; pass --into %s to add to it' % (name, name))
    else:
        os.makedirs(target)
    if source:
        for existing in sorted(os.listdir(os.path.join(ROOT, 'reference', source))):
            if (existing.endswith('.KBP') and existing != instance + '.KBP'
                    and not os.path.exists(os.path.join(target, existing))):
                shutil.copyfile(os.path.join(ROOT, 'reference', source, existing),
                                os.path.join(target, existing))
    with open(os.path.join(target, instance + '.KBP'), 'w', encoding='utf-8', newline='\n') as handle:
        for costs in sorted(front):
            handle.write('%s %s\n' % (' '.join(str(v) for v in permutations[costs]),
                                      ' '.join(str(v) for v in costs)))
    # Several instances of one experiment write into the same version, so the record accumulates.
    path = os.path.join(target, 'summary.json')
    summary = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {
        'version': name,
        'date': datetime.date.today().isoformat(),
        'provenance': 'Copy of %s with every solution found afterwards that it does not dominate. Every '
                      'addition is verified by recomputing the cost of its permutation against the '
                      'instance.' % (source or 'nothing'),
        'added': [],
    }
    summary['added'].append({
        'instance': instance,
        'from': source,
        'points': len(front),
        'found_by': sorted(contributors),
    })
    summary['instances'] = {}
    for existing in sorted(os.listdir(target)):
        if not existing.endswith('.KBP'):
            continue
        points = [tuple(int(v) for v in line.split())
                  for line in open(os.path.join(target, existing), encoding='utf-8') if line.split()]
        summary['instances'][existing[:-len('.KBP')]] = {'points': len(points)}
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    return name


def reference_path(instance, version=None):
    """The published optimum when the instance has one, otherwise the .KBP of a reference version."""
    published = os.path.join(ROOT, 'mQAPData', instance + '.PO')
    if os.path.exists(published):
        return published
    available = reference_versions()
    if version is None:
        if not available:
            sys.exit('%s: no reference front: neither mQAPData/%s.PO nor any reference/v* directory'
                     % (instance, instance))
        version = available[-1]
    path = os.path.join(ROOT, 'reference', version, instance + '.KBP')
    if os.path.exists(path):
        return path
    sys.exit('%s: no reference front: neither mQAPData/%s.PO nor reference/%s/%s.KBP'
             % (instance, instance, version, instance))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    parser.add_argument('instance', help='instance name, e.g. KC20-2fl-1uni')
    parser.add_argument('groups', nargs='+', metavar='LABEL=PATH',
                        help='result file per version or configuration; the first one is the baseline')
    parser.add_argument('--reference', metavar='VERSION',
                        help='reference version to measure against, such as v0.1 (default: the latest '
                             'directory of reference/)')
    parser.add_argument('--into', metavar='VERSION',
                        help='with --update-reference, the version directory to write into, which must '
                             'already exist; this is how several instances of one experiment share a '
                             'version (default: create the next one)')
    parser.add_argument('--update-reference', action='store_true',
                        help='write the next reference version with the solutions this one does not '
                             'dominate (never for an instance with a published .PO)')
    args = parser.parse_args()

    versions = []
    for group in args.groups:
        if '=' not in group:
            sys.exit('expected LABEL=PATH, got %r' % group)
        # Split at the last '=': a label such as "this version, P = 64" contains one, a path does not.
        label, pattern = group.rsplit('=', 1)
        found = sorted(glob.glob(pattern))
        if not found:
            sys.exit('%s: no file matches %s' % (label, pattern))
        versions.append((label, read_runs(found[0]), found[0]))

    path = reference_path(args.instance, args.reference)
    shown = os.path.relpath(path, ROOT).replace(os.sep, '/')
    source = os.path.basename(os.path.dirname(path)) if path.endswith('.KBP') else None
    front = read_front(path)
    known = {costs: permutation for permutation, costs in front}
    n, _objectives, distances, flows = load_instance(os.path.join(ROOT, 'mQAPData', args.instance + '.dat'))

    # Solutions of any version that the reference front does not dominate, with their permutation checked.
    additions = {}
    for label, runs, _path in versions:
        for run in runs:
            for key, costs in run:
                if costs in known or costs in additions or any(dominates(r, costs) for r in known):
                    continue
                matches = [p for p in permutations_of(key, n)
                           if tuple(cost_of(n, distances, f, p) for f in flows) == costs]
                if not matches:
                    sys.exit('%s: the cost of %s does not reproduce against the instance' % (label, key))
                additions[costs] = ([v + 1 for v in matches[0]], label)

    update = args.update_reference and additions and path.endswith('.KBP')
    union = dict(known)
    union.update({costs: permutation for costs, (permutation, _label) in additions.items()})
    reference = set(minimal(list(union))) if update else set(known)

    point = reference_point_of(sorted(reference))
    total = hypervolume(minimal(sorted(reference)), point)

    print('instance %s, reference front %s with %d points, reference point %s'
          % (args.instance, shown, len(reference), point))
    if additions:
        by_label = {}
        for _costs, (_permutation, label) in additions.items():
            by_label[label] = by_label.get(label, 0) + 1
        print('  %d solutions are not dominated by that front (%s)'
              % (len(additions), ', '.join('%s: %d' % kv for kv in sorted(by_label.items()))))
        if update:
            removed = len([c for c in known if c not in reference])
            name = write_version(source, args.instance, reference, union,
                                 {label for _c, (_p, label) in additions.items()}, args.into)
            old_total = hypervolume(minimal(sorted(known)), point)
            print('  reference/%s/%s.KBP written: %d -> %d points (%d removed). A hypervolume share '
                  'published against %s rescales by %.7f and a coverage share by %.5f'
                  % (name, args.instance, len(known), len(reference), removed, source,
                     old_total / total, float(len(known)) / len(reference)))
        elif path.endswith('.KBP'):
            print('  pass --update-reference to write the next reference version with them')

    measured = []
    for label, runs, _path in versions:
        volumes = [100.0 * hypervolume(minimal([c for _k, c in run]), point) / total for run in runs]
        covers = [100.0 * len(reference & {c for _k, c in run}) / len(reference) for run in runs]
        measured.append((label, volumes, covers))

    for column, title in ((1, 'hypervolume, % of the one the reference front dominates'),
                          (2, 'coverage, % of the points of the reference front')):
        print('\n  %s' % title)
        print('  %-34s %5s %9s %9s %9s %11s' % ('configuration', 'runs', 'mean', 'sd', 'best', 'p'))
        baseline = measured[0][column]
        for group in measured:
            values = group[column]
            if group is measured[0] or mannwhitneyu is None:
                p = ''
            else:
                pvalue = mannwhitneyu(baseline, values, alternative='two-sided').pvalue
                if math.isnan(pvalue):
                    p = '%11s' % '—'  # every value is identical, the test is undefined
                else:
                    p = '%11.2e%s' % (pvalue, '' if pvalue >= 0.05 else ' *')
            print('  %-34s %5d %9.3f %9.3f %9.3f %s'
                  % (group[0], len(values), statistics.mean(values),
                     statistics.stdev(values) if len(values) > 1 else 0.0, max(values), p))
    if mannwhitneyu is None:
        print('\n  scipy is not installed, so no test was run (pip install scipy)')
    else:
        print('\n  * marks p < 0.05 against %s' % measured[0][0])


if __name__ == '__main__':
    main()
