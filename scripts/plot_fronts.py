# -*- coding: utf-8 -*-
"""
plot_fronts.py

Plots the best known front of an instance against the front of one run and the population that run started from.

    python scripts/plot_fronts.py KC30-3fl-1rl --result results/fronts/KC30-3fl-1rl_result.txt ^
        --initial results/fronts/KC30-3fl-1rl_initial.csv --out results/fronts --png

Three series, each one can be hidden from the legend of the HTML:

  initial population  The 2P random permutations the run started from (`cuda_mqap --initial FILE`), in
                      grey: where the search begins.
  run                 The front the run ended with: the last block of the result file (`--output`), that
                      is, the most recent run appended to it, or the one `--block` names.
  best known front    `mQAPData/<instance>.PO` when the instance has a published optimum, and otherwise
                      `reference/<version>/<instance>.KBP`, the latest version unless `--reference` names
                      another one.

The HTML is interactive (Plotly): a 3D scatter that rotates and zooms for 3 objectives, a 2D one for 2.
It loads Plotly from its CDN, so it needs a connection to be opened; `--self-contained` embeds the library
instead, about 4.6 MB more per file. `--png` also writes a static figure (matplotlib) with the three
projections of the objective space and a 3D view, which is what a document can show inline.

The console and the title report how many points of the best known front the run found, and how many
points of the run the best known front does not dominate: those would improve it, see reference/README.md.

Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from compare_versions import read_front, read_runs, reference_path  # noqa: E402

COLORS = {'initial': '#9e9e9e', 'run': '#d95f02', 'reference': '#1b62a5'}


def read_initial(path, run):
    """Objective vectors of one run of an `--initial` CSV, as an int64 array."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        objectives = [i for i, name in enumerate(header) if name.startswith('f')]
        for row in reader:
            if int(row[0]) == run:
                rows.append([int(row[i]) for i in objectives])
    if not rows:
        sys.exit('%s: no rows for run %d' % (path, run))
    return np.array(rows, dtype=np.int64)


def not_dominated_by(points, front, chunk=2048):
    """How many of `points` no point of `front` dominates (or equals)."""
    count = 0
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk][:, None, :]
        weakly = np.all(front[None, :, :] <= block, axis=2)
        count += int(np.sum(~np.any(weakly, axis=1)))
    return count


def summary(instance, reference, run, initial, reference_label):
    found = len({tuple(p) for p in run} & {tuple(p) for p in reference})
    beyond = not_dominated_by(run, reference)
    lines = [
        '%s: best known front %s, %d points' % (instance, reference_label, len(reference)),
        'run: %d distinct points, %d of them on the best known front (%.1f %% of it), %d not dominated by it'
        % (len(run), found, 100.0 * found / len(reference), beyond),
    ]
    if initial is not None:
        lines.append('initial population: %d solutions' % len(initial))
    return lines, found, beyond


def plot_html(path, instance, series, objectives, title, self_contained):
    import plotly.graph_objects as go

    figure = go.Figure()
    for name, points, color, size, opacity in series:
        hover = '<br>'.join('f%d = %%{%s:,}' % (o + 1, 'xyz'[o]) for o in range(objectives))
        if objectives == 3:
            figure.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers', name=name,
                marker=dict(size=size, color=color, opacity=opacity), hovertemplate=hover))
        else:
            figure.add_trace(go.Scattergl(
                x=points[:, 0], y=points[:, 1], mode='markers', name=name,
                marker=dict(size=size + 2, color=color, opacity=opacity), hovertemplate=hover))
    if objectives == 3:
        figure.update_layout(scene=dict(xaxis_title='f1', yaxis_title='f2', zaxis_title='f3'))
    else:
        figure.update_layout(xaxis_title='f1', yaxis_title='f2')
    figure.update_layout(title=dict(text=title, font=dict(size=14)), template='plotly_white',
                         legend=dict(itemsizing='constant', x=0.01, y=0.99), margin=dict(l=0, r=0, t=90, b=0))
    figure.write_html(path, include_plotlyjs=True if self_contained else 'cdn', full_html=True)


def plot_png(path, instance, series, objectives, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pairs = [(0, 1), (0, 2), (1, 2)] if objectives == 3 else [(0, 1)]
    if objectives == 3:
        figure = plt.figure(figsize=(13, 11))
        axes = [figure.add_subplot(2, 2, i + 1) for i in range(3)]
        view = figure.add_subplot(2, 2, 4, projection='3d')
    else:
        figure = plt.figure(figsize=(8, 6.5))
        axes = [figure.add_subplot(1, 1, 1)]
        view = None
    for name, points, color, size, opacity in series:
        size = size if objectives == 3 else 3 * size  # one panel instead of four: room for larger marks
        for ax, (a, b) in zip(axes, pairs):
            ax.scatter(points[:, a], points[:, b], s=size, c=color, alpha=opacity, label=name,
                       linewidths=0, rasterized=True)
        if view is not None:
            view.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, c=color, alpha=opacity,
                         linewidths=0, rasterized=True)
    for ax, (a, b) in zip(axes, pairs):
        ax.set_xlabel('f%d' % (a + 1))
        ax.set_ylabel('f%d' % (b + 1))
        ax.ticklabel_format(style='sci', scilimits=(0, 0))
        ax.grid(alpha=0.3)
    if view is not None:
        view.set_xlabel('f1')
        view.set_ylabel('f2')
        view.set_zlabel('f3')
    legend = axes[0].legend(loc='upper right', markerscale=4)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[4])
    parser.add_argument('instance', help='instance name, e.g. KC30-3fl-1rl')
    parser.add_argument('--result', required=True, help='result file of cuda_mqap (--output)')
    parser.add_argument('--block', type=int, default=-1,
                        help='run block of the result file to plot (default -1, the last one appended)')
    parser.add_argument('--initial', help='initial population of cuda_mqap (--initial); optional')
    parser.add_argument('--run', type=int, default=0, help='run of the --initial file to plot (default 0)')
    parser.add_argument('--reference', help='reference version, e.g. v0.3 (default: the latest)')
    parser.add_argument('--label', default='', help='text added to the title, e.g. the command of the run')
    parser.add_argument('--out', default=os.path.join('results', 'fronts'), help='output directory')
    parser.add_argument('--png', action='store_true', help='also write a static PNG')
    parser.add_argument('--self-contained', action='store_true', help='embed Plotly in the HTML')
    args = parser.parse_args()

    reference_file = reference_path(args.instance, args.reference)
    reference = np.array([costs for _, costs in read_front(reference_file)], dtype=np.int64)
    blocks = read_runs(args.result)
    if not blocks:
        sys.exit('%s: no run blocks' % args.result)
    run = np.array(sorted({costs for _, costs in blocks[args.block]}), dtype=np.int64)
    initial = read_initial(args.initial, args.run) if args.initial else None
    objectives = reference.shape[1]
    if run.shape[1] != objectives or (initial is not None and initial.shape[1] != objectives):
        sys.exit('the files do not have the same number of objectives')

    reference_label = os.path.relpath(reference_file, os.path.dirname(HERE)).replace(os.sep, '/')
    lines, found, beyond = summary(args.instance, reference, run, initial, reference_label)
    print('\n'.join(lines))

    series = []
    if initial is not None:
        series.append(('initial population (%d)' % len(initial), initial, COLORS['initial'], 1.5, 0.25))
    series.append(('best known front (%d)' % len(reference), reference, COLORS['reference'], 2, 0.6))
    series.append(('run (%d)' % len(run), run, COLORS['run'], 3, 0.9))
    title = ('%s: the run found %d of the %d points of the best known front (%.1f %%), %d beyond it'
             % (args.instance, found, len(reference), 100.0 * found / len(reference), beyond))
    if args.label:
        title += '<br><sup>%s</sup>' % args.label

    os.makedirs(args.out, exist_ok=True)
    html = os.path.join(args.out, args.instance + '.html')
    plot_html(html, args.instance, series, objectives, title, args.self_contained)
    print('written', html)
    if args.png:
        png = os.path.join(args.out, args.instance + '.png')
        plot_png(png, args.instance, series, objectives, title.replace('<br><sup>', '\n').replace('</sup>', ''))
        print('written', png)


if __name__ == '__main__':
    main()
