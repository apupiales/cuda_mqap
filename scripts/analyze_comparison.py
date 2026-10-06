# -*- coding: utf-8 -*-
"""
analyze_comparison.py

Analyses the campaign of scripts/run_comparison.ps1: cuda_mqap against the pymoo baselines.

    python scripts/analyze_comparison.py results/comparison --block budget
    python scripts/analyze_comparison.py results/comparison --block gar60 --protocol time

For every instance of a block, every result file <algorithm>_<protocol>.txt is measured against one reference
front, with the indicators of compare_versions.py:

  hypervolume  share of the volume the reference front dominates, reference point from that front;
  coverage     share of the points of the reference front the run found exactly.

The reference front is the published optimum (.PO) on KC10 and the latest reference/v* front on KC20 and KC30.
The Gar60 instances have none, so their reference is the non-dominated union of every run of every algorithm
and protocol of the campaign on that instance, which is also reported as a candidate front.

Each baseline is compared with cuda_mqap of the same protocol by the two-sided Mann-Whitney U test, with the
Holm correction over the baselines of the instance, and the Vargha-Delaney A12 effect size (the probability that
a run of cuda_mqap beats a run of the baseline; 0.5 is a tie). Over the instances: wins, ties and losses at
corrected p < 0.05, and the Friedman test with the mean rank of each algorithm on the mean coverage.

Output: a Markdown summary and a JSON file with every number, in the campaign directory.

Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later
"""
import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from analyze_convergence import hypervolume, minimal, reference_point_of  # noqa: E402
from compare_versions import dominates, read_front, read_runs, reference_path  # noqa: E402
from scipy.stats import friedmanchisquare, mannwhitneyu, rankdata  # noqa: E402

OURS = 'cuda_mqap'
PROTOCOL_OF_OURS = {'p64': 'p64', 'time': None}   # time: cuda_mqap_default (KC) or cuda_mqap_p4096 (Gar60)


def a12(a, b):
    """Vargha-Delaney: probability that a value of a is larger than one of b, ties counting half."""
    wins = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    return wins / (len(a) * len(b))


def holm(pvalues):
    """Holm-Bonferroni adjusted p values, in the original order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def measure(runs, reference, point, total):
    hv, cov = [], []
    for run in runs:
        points = sorted({costs for _key, costs in run})
        hv.append(100.0 * hypervolume(minimal(points), point) / total if points else 0.0)
        cov.append(100.0 * len(set(points) & reference) / len(reference))
    return hv, cov


def files_of(directory, protocol):
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, '*.txt'))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.endswith('.partial'):
            continue
        algorithm, _, suffix = name.rpartition('_')
        if algorithm == OURS:
            if protocol == 'p64' and suffix == 'p64':
                out[OURS] = path
            elif protocol == 'time' and suffix in ('default', 'p4096'):
                out[OURS] = path
        elif suffix == protocol:
            out[algorithm] = path
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[4])
    parser.add_argument('root', help='campaign directory, e.g. results/comparison')
    parser.add_argument('--block', required=True, choices=('budget', 'gar60', 'time'))
    parser.add_argument('--protocol', choices=('p64', 'time'),
                        help='p64 (same work) or time (same wall time); default p64, and time for the time block')
    args = parser.parse_args()
    protocol = args.protocol or ('time' if args.block == 'time' else 'p64')

    block = os.path.join(args.root, args.block)
    instances = sorted(d for d in os.listdir(block) if os.path.isdir(os.path.join(block, d)))
    report = {'block': args.block, 'protocol': protocol, 'instances': {}}
    for instance in instances:
        directory = os.path.join(block, instance)
        files = files_of(directory, protocol)
        if OURS not in files or len(files) < 2:
            continue
        runs = {label: read_runs(path) for label, path in files.items()}
        if instance.startswith('Gar60'):
            # No published or versioned front: the non-dominated union of everything run on the instance.
            union = set()
            for path in glob.glob(os.path.join(directory, '*.txt')):
                if not path.endswith('.partial.txt'):
                    for run in read_runs(path):
                        union.update(costs for _key, costs in run)
            reference = set(minimal(sorted(union)))
            source = 'union of the campaign'
        else:
            path = reference_path(instance)
            reference = {costs for _permutation, costs in read_front(path)}
            source = os.path.relpath(path, os.path.dirname(HERE)).replace(os.sep, '/')
        point = reference_point_of(sorted(reference))
        total = hypervolume(minimal(sorted(reference)), point)
        entry = {'reference': source, 'reference_points': len(reference), 'algorithms': {}}
        for label, data in runs.items():
            hv, cov = measure(data, reference, point, total)
            beyond = set()
            if not instance.startswith('Gar60'):
                for run in data:
                    for _key, costs in run:
                        if costs not in reference and not any(dominates(r, costs) for r in reference):
                            beyond.add(costs)
            entry['algorithms'][label] = {'runs': len(data), 'hv': hv, 'coverage': cov,
                                          'hv_mean': statistics.fmean(hv), 'coverage_mean': statistics.fmean(cov),
                                          'hv_sd': statistics.pstdev(hv), 'coverage_sd': statistics.pstdev(cov),
                                          'beyond_reference': len(beyond)}
        baselines = [label for label in entry['algorithms'] if label != OURS]
        ours = entry['algorithms'][OURS]
        for indicator in ('hv', 'coverage'):
            raw = []
            for label in baselines:
                other = entry['algorithms'][label][indicator]
                same = ours[indicator] == other or (len(set(ours[indicator])) == 1 and set(ours[indicator]) == set(other))
                raw.append(1.0 if same else float(mannwhitneyu(ours[indicator], other, alternative='two-sided').pvalue))
            for label, p, q in zip(baselines, raw, holm(raw)):
                entry['algorithms'][label]['p_' + indicator] = p
                entry['algorithms'][label]['p_holm_' + indicator] = q
                entry['algorithms'][label]['a12_' + indicator] = a12(ours[indicator], entry['algorithms'][label][indicator])
        report['instances'][instance] = entry

    # Over the instances that have every algorithm.
    labels = sorted({label for e in report['instances'].values() for label in e['algorithms']},
                    key=lambda l: (l != OURS, l))
    complete = [i for i, e in report['instances'].items() if all(l in e['algorithms'] for l in labels)]
    summary = {'instances': len(complete), 'labels': labels}
    for indicator in ('hv', 'coverage'):
        tally = {}
        for label in labels[1:]:
            w = t = l = 0
            for instance in complete:
                a = report['instances'][instance]['algorithms'][label]
                if a['p_holm_' + indicator] >= 0.05:
                    t += 1
                elif a['a12_' + indicator] > 0.5:
                    w += 1
                else:
                    l += 1
            tally[label] = {'cuda_mqap_wins': w, 'ties': t, 'cuda_mqap_losses': l}
        summary['tally_' + indicator] = tally
        if len(complete) >= 2 and len(labels) >= 3:
            matrix = [[report['instances'][i]['algorithms'][l][indicator + '_mean'] for l in labels] for i in complete]
            ranks = [rankdata([-v for v in row]) for row in matrix]   # 1 = best
            summary['mean_rank_' + indicator] = {l: statistics.fmean(r[k] for r in ranks) for k, l in enumerate(labels)}
            columns = [[row[k] for row in matrix] for k in range(len(labels))]
            summary['friedman_p_' + indicator] = float(friedmanchisquare(*columns).pvalue)
    report['summary'] = summary

    name = '%s_%s' % (args.block, protocol)
    with open(os.path.join(args.root, 'summary_%s.json' % name), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)

    lines = ['# Comparison: block %s, protocol %s' % (args.block, protocol), '',
             'Mean over the runs; p is the Holm-corrected two-sided Mann-Whitney p value against cuda_mqap,',
             'and A12 the probability that a run of cuda_mqap beats one of the baseline.', '']
    for indicator, title in (('coverage', 'Coverage of the reference front (%)'), ('hv', 'Hypervolume share (%)')):
        lines += ['## ' + title, '', '| Instance | Reference | ' + ' | '.join(labels) + ' |',
                  '|---|---|' + '---|' * len(labels)]
        for instance, e in report['instances'].items():
            cells = []
            for label in labels:
                a = e['algorithms'].get(label)
                if a is None:
                    cells.append('—')
                elif label == OURS:
                    cells.append('**%.2f**' % a[indicator + '_mean'])
                else:
                    mark = '' if a['p_holm_' + indicator] >= 0.05 else (' ▼' if a['a12_' + indicator] > 0.5 else ' ▲')
                    cells.append('%.2f%s' % (a[indicator + '_mean'], mark))
            lines.append('| %s | %d | %s |' % (instance, e['reference_points'], ' | '.join(cells)))
        lines += ['', '▼ significantly worse than cuda_mqap, ▲ significantly better (Holm-corrected p < 0.05).', '']
        tally = summary.get('tally_' + indicator, {})
        if tally:
            lines += ['cuda_mqap against each baseline over %d instances (wins / ties / losses):' % summary['instances'], '']
            for label, t in tally.items():
                lines.append('- %s: %d / %d / %d' % (label, t['cuda_mqap_wins'], t['ties'], t['cuda_mqap_losses']))
            if 'mean_rank_' + indicator in summary:
                ranks = summary['mean_rank_' + indicator]
                lines += ['', 'Mean rank (1 = best): ' + ', '.join('%s %.2f' % (l, ranks[l]) for l in labels)
                          + '; Friedman p = %.2e' % summary['friedman_p_' + indicator]]
            lines.append('')
    beyond = {i: e['algorithms'][l]['beyond_reference'] for i, e in report['instances'].items()
              for l in e['algorithms'] if e['algorithms'][l]['beyond_reference']}
    if beyond:
        lines += ['Points not dominated by the reference front were found on: ' + ', '.join(sorted(set(beyond))), '']
    with open(os.path.join(args.root, 'summary_%s.md' % name), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
