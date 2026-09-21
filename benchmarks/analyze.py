# -*- coding: utf-8 -*-
"""
analyze.py

Turns the CSV reports written by run_comparison.ps1 (nsys) into the tables documented in
README.md / LEEME.md. Usage: python analyze.py <run directory>

Metrics, per version:
  window        first kernel -> last kernel of the process. Everything is measured inside this window,
                so the fixed cost of starting the process and creating the CUDA context is left out.
  gpu_busy      time the GPU spends running kernels, over the window. This is the fraction of the
                algorithm that really runs in parallel (the rest is the GPU waiting for the host).
  transfer      time in host <-> device copies, over the window.
  occupancy     time-weighted average of the thread slots in use: sum over launches of
                min(threads, SLOTS) * duration, over window * SLOTS. SLOTS = SMs * threads per SM.
  useful        the same, counting only the threads that pass the guard of the kernel. The original
                launches blocks of 32x32 = 1024 threads for matrices of n x n, so with n = 10 only
                100/1024 of the threads it starts do any work.
  thread-sec    total parallel work executed: sum of threads * duration over every launch.

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
import csv
import glob
import os
import sys

# Resident thread slots of the GPU the measurements were taken on (RTX 2060: 30 SMs x 1024 threads).
# Change these two numbers when measuring on another GPU.
SMS = 30
THREADS_PER_SM = 1024
SLOTS = SMS * THREADS_PER_SM

# Kernels of the original launched with dim3(32, 32) and guarded by (j < n && k < n).
GUARDED_KERNELS = ('populationTo2DRepresentation', 'multiplicationWithFlowMatrix',
                   'multiplicationWithTranposedDistanceMatrix', 'matrixMultiplication')


def read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def column(row, *names):
    for name in names:
        for key in row:
            if key.strip().lower() == name.lower():
                return row[key]
    return None


def number(value):
    """nsys writes plain integers and dotted decimals; quoted locale fields may carry commas."""
    if value in (None, ''):
        return 0.0
    try:
        return float(str(value).replace(',', ''))
    except ValueError:
        return 0.0


def report(run, case, name):
    hits = glob.glob('%s/rep/%s*%s.csv' % (run, case, name))
    return hits[0] if hits else None


def guard_factor(kernel_name, facilities):
    if any(g in kernel_name for g in GUARDED_KERNELS):
        return (facilities * facilities) / 1024.0
    return 1.0


def analyze(run, case, wall, cpu, facilities):
    out = {'case': case, 'wall_s': wall, 'cpu_s': cpu}

    kernels = read_csv(report(run, case, 'cuda_gpu_kern_sum'))
    out['launches'] = sum(number(column(r, 'Instances', 'Count')) for r in kernels)

    memory = read_csv(report(run, case, 'cuda_gpu_mem_time_sum'))
    out['memcpys'] = sum(number(column(r, 'Count', 'Instances')) for r in memory)

    out['syncs'] = 0.0
    for r in read_csv(report(run, case, 'cuda_api_sum')):
        if 'Synchronize' in (column(r, 'Name') or ''):
            out['syncs'] += number(column(r, 'Num Calls', 'Count', 'Instances'))

    kernel_ns = transfer_ns = occupied = useful_occupied = 0.0
    threads_total = useful_threads = thread_seconds = useful_thread_seconds = 0.0
    launches = 0
    start = end = None
    per_kernel = {}
    for r in read_csv(report(run, case, 'cuda_gpu_trace')):
        duration = number(column(r, 'Duration (ns)'))
        begin = number(column(r, 'Start (ns)'))
        grid, block = number(column(r, 'GrdX')), number(column(r, 'BlkX'))
        if grid <= 0 or block <= 0:
            transfer_ns += duration  # memcpy / memset row
            continue
        threads = (grid * max(number(column(r, 'GrdY')), 1) * max(number(column(r, 'GrdZ')), 1) *
                   block * max(number(column(r, 'BlkY')), 1) * max(number(column(r, 'BlkZ')), 1))
        name = (column(r, 'Name') or '')
        factor = guard_factor(name, facilities)
        kernel_ns += duration
        launches += 1
        threads_total += threads
        useful_threads += threads * factor
        thread_seconds += threads * duration / 1e9
        useful_thread_seconds += threads * factor * duration / 1e9
        occupied += min(threads, SLOTS) * duration
        useful_occupied += min(threads * factor, SLOTS) * duration
        start = begin if start is None else min(start, begin)
        end = begin + duration if end is None else max(end, begin + duration)
        entry = per_kernel.setdefault(name.split('(')[0].strip(), {'n': 0, 'ns': 0.0, 'threads': 0.0})
        entry['n'] += 1
        entry['ns'] += duration
        entry['threads'] += threads

    window = (end - start) if start is not None else 0.0
    out['window_ms'] = window / 1e6
    out['kernel_ms'] = kernel_ns / 1e6
    out['gpu_busy_pct'] = 100.0 * kernel_ns / window if window else 0.0
    out['transfer_pct'] = 100.0 * transfer_ns / window if window else 0.0
    out['occupancy_pct'] = 100.0 * occupied / (window * SLOTS) if window else 0.0
    out['useful_occupancy_pct'] = 100.0 * useful_occupied / (window * SLOTS) if window else 0.0
    out['useful_threads_pct'] = 100.0 * useful_threads / threads_total if threads_total else 0.0
    out['thread_seconds'] = thread_seconds
    out['useful_thread_seconds'] = useful_thread_seconds
    out['threads_per_launch'] = threads_total / launches if launches else 0.0
    out['cpu_per_wall_pct'] = 100.0 * cpu / wall if wall else 0.0
    out['_per_kernel'] = per_kernel
    return out


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'results', 'run')
    facilities = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    times = read_csv(os.path.join(run, 'times.csv'))
    if not times:
        sys.exit('No times.csv in %s; run run_comparison.ps1 first.' % run)

    rows = [analyze(run, t['name'], float(t['wall']), float(t['cpu']), facilities) for t in times]

    head = ('%-12s %8s %8s %9s %10s %9s %9s %9s %8s %8s %9s %8s %9s' %
            ('case', 'wall(s)', 'cpu(s)', 'cpu/wall', 'window(ms)', 'kernel_ms', 'gpu_busy', 'transfer',
             'launches', 'memcpys', 'syncs', 'occup', 'useful_th'))
    print(head)
    print('-' * len(head))
    for o in rows:
        print('%-12s %8.3f %8.3f %8.0f%% %10.1f %9.1f %8.1f%% %8.2f%% %8.0f %8.0f %9.0f %7.1f%% %8.1f%%' % (
            o['case'], o['wall_s'], o['cpu_s'], o['cpu_per_wall_pct'], o['window_ms'], o['kernel_ms'],
            o['gpu_busy_pct'], o['transfer_pct'], o['launches'], o['memcpys'], o['syncs'],
            o['occupancy_pct'], o['useful_threads_pct']))

    print('\nWork executed and useful occupancy (threads that pass the guard of the kernel):')
    print('%-12s %14s %16s %14s %16s' % ('case', 'thread-sec', 'useful-thr-sec', 'occupancy', 'useful occupancy'))
    for o in rows:
        print('%-12s %14.1f %16.1f %13.1f%% %15.1f%%' % (
            o['case'], o['thread_seconds'], o['useful_thread_seconds'], o['occupancy_pct'],
            o['useful_occupancy_pct']))

    for case in (rows[0]['case'], rows[-1]['case']):
        entry = next(o for o in rows if o['case'] == case)
        print('\n--- %s: kernels by GPU time ---' % case)
        print('%-44s %8s %10s %13s' % ('kernel', 'calls', 'time(ms)', 'threads/call'))
        for name, e in sorted(entry['_per_kernel'].items(), key=lambda kv: -kv[1]['ns'])[:12]:
            print('%-44s %8d %10.1f %13.0f' % (name[:44], e['n'], e['ns'] / 1e6, e['threads'] / e['n']))

    path = os.path.join(run, 'summary.csv')
    fields = [k for k in rows[0] if not k.startswith('_')]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for o in rows:
            writer.writerow(o)
    print('\nwritten: %s' % path)


if __name__ == '__main__':
    main()
