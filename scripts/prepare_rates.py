# -*- coding: utf-8 -*-
"""
prepare_rates.py

Prepares one build tree per greedy configuration, ready for nvcc.

How much of the population the greedy 2-opt improves is a compile-time constant, `kGreedyRate` and
`kGreedyPeriod` in include/config.h, so comparing configurations needs one binary each. This script copies
src/ and include/ into a directory per configuration and substitutes those two constants; nothing else
changes, so the only difference between the binaries is the configuration they were built with.

Usage:
    python scripts/prepare_rates.py out/dir 1.0:1 0.5:1 0.25:1 0.1:1
    python scripts/prepare_rates.py out/dir --list

Each configuration is `rate:period`, and its directory is named after it: rate100p1, rate050p1, and so
on, which is also the label the grid uses in its result files.

Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later
"""
import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def name_of(rate, period):
    """rate100p1 for 1.0:1: the directory, and the label of the configuration in every report."""
    return 'rate%03dp%d' % (round(rate * 100), period)


def parse(text):
    """'0.5:2' -> (0.5, 2); the period defaults to 1."""
    pieces = text.split(':')
    if len(pieces) > 2:
        sys.exit('expected rate[:period], got %r' % text)
    try:
        rate = float(pieces[0])
        period = int(pieces[1]) if len(pieces) == 2 else 1
    except ValueError:
        sys.exit('expected rate[:period] with numbers, got %r' % text)
    if not 0.0 <= rate <= 1.0:
        sys.exit('the rate must be in [0, 1], got %r' % pieces[0])
    if period < 1:
        sys.exit('the period must be at least 1, got %r' % period)
    return rate, period


def prepare(target, rate, period):
    """Writes the tree of one configuration and returns its path."""
    directory = os.path.join(target, name_of(rate, period))
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    for part in ('src', 'include', 'tests'):
        shutil.copytree(os.path.join(ROOT, part), os.path.join(directory, part))

    config = os.path.join(directory, 'include', 'config.h')
    with open(config, encoding='utf-8', newline='') as handle:
        text = handle.read()
    for pattern, replacement in (
            (r'constexpr float kGreedyRate = [^;]+;', 'constexpr float kGreedyRate = %sf;' % rate),
            (r'constexpr int kGreedyPeriod = [^;]+;', 'constexpr int kGreedyPeriod = %d;' % period)):
        text, count = re.subn(pattern, replacement.replace('\\', '\\\\'), text, count=1)
        if count != 1:
            sys.exit('%s: %s was not found exactly once' % (config, pattern))
    with open(config, 'w', encoding='utf-8', newline='') as handle:
        handle.write(text)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[4])
    parser.add_argument('target', help='directory the configuration trees are written into')
    parser.add_argument('configurations', nargs='*', metavar='RATE[:PERIOD]',
                        help='greedy configurations, such as 1.0 0.5:1 0.1:2 (default: 1.0 0.5 0.25 0.1)')
    parser.add_argument('--list', action='store_true', help='print the names and exit, writing nothing')
    args = parser.parse_args()

    requested = args.configurations or ['1.0', '0.5', '0.25', '0.1']
    pairs = [parse(text) for text in requested]
    if args.list:
        for rate, period in pairs:
            print('%-12s rate %s, period %d' % (name_of(rate, period), rate, period))
        return

    for rate, period in pairs:
        directory = prepare(args.target, rate, period)
        print('%-12s rate %s, period %d -> %s' % (name_of(rate, period), rate, period, directory))


if __name__ == '__main__':
    main()
