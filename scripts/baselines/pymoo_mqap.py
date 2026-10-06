# -*- coding: utf-8 -*-
"""
pymoo_mqap.py

Reference MOEAs from pymoo on the mQAP, for the comparison with cuda_mqap.

    python scripts/baselines/pymoo_mqap.py mQAPData/KC20-2fl-1rl.dat --algorithm nsga2 --pop 64 --gen 300 --runs 30
    python scripts/baselines/pymoo_mqap.py data/gar60/Gar60-2fl-1rl.dat --algorithm moead-ls --evals 200000 --runs 30

Algorithms, all with permutation operators (random sampling, order crossover, inversion mutation):

  nsga2, nsga3, moead          pymoo's implementations as they are.
  nsga2-ls, nsga3-ls, moead-ls The same, with the adapted greedy 2-opt of cuda_mqap applied to the offspring
                               of every generation as a repair: the same pair traversal (r in [0, n-2], s in
                               [1, n-1], r != s), the same acceptance (the criterion of the generation does not
                               get worse) and a criterion drawn per generation among the sum and each objective.
                               --greedy-rate applies it to that fraction of the offspring.

The budget is --gen generations of --pop individuals, --evals full evaluations (pymoo's n_eval) or --seconds of
wall time per run;
the swap deltas of the local search are counted apart, as cuda_mqap counts them, so that the analysis can
put every algorithm on the same scale.

Output: the result file format of cuda_mqap (one block per run with the distinct non-dominated solutions
of the final population), which scripts/compare_versions.py reads, and a JSON file next to it with the time,
the evaluations and the swap deltas of every run. The runs are independent processes (--jobs); each one
runs on one core, so the time of a run is single-threaded.

Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numba import njit
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

ALGORITHMS = ('nsga2', 'nsga3', 'moead', 'nsga2-ls', 'nsga3-ls', 'moead-ls')


def load_instance(path):
    """(n, m, D [n][n], F [m][n][n]) in the order src/instance.cpp reads them: distances, then each flow."""
    with open(path, encoding='utf-8') as f:
        head, rest = f.read().split('\n', 1)
    head = head.replace(':', ' ').replace('=', ' ').split()
    n = int(head[head.index('facilities') + 1])
    m = int(head[head.index('objectives') + 1])
    values = np.array(rest.split(), dtype=np.int64)
    D = values[:n * n].reshape(n, n)
    F = values[n * n:n * n * (m + 1)].reshape(m, n, n)
    return n, m, D, F


def costs(X, D, F):
    """cost_k(p) = sum_ij F_k[i][j] D[p(i)][p(j)] for every row p of X."""
    Dp = D[X[:, :, None], X[:, None, :]]                 # [pop][n][n]
    return np.einsum('kij,pij->pk', F, Dp)


@njit(cache=True)
def swap_delta(p, F, D, r, s):
    n = p.shape[0]
    pr = p[r]
    ps = p[s]
    delta = (F[r, r] - F[s, s]) * (D[ps, ps] - D[pr, pr]) + (F[r, s] - F[s, r]) * (D[ps, pr] - D[pr, ps])
    for k in range(n):
        if k == r or k == s:
            continue
        pk = p[k]
        delta += (F[k, r] - F[k, s]) * (D[pk, ps] - D[pk, pr]) + (F[r, k] - F[s, k]) * (D[ps, pk] - D[pr, pk])
    return delta


@njit(cache=True)
def greedy_2opt(X, F, D, criterion, apply):
    """The adapted greedy 2-opt of cuda_mqap on the rows of X where apply is true, in place."""
    rows, n = X.shape
    m = F.shape[0]
    deltas = np.zeros(m, dtype=np.int64)
    trials = 0
    for row in range(rows):
        if not apply[row]:
            continue
        p = X[row]
        for r in range(n - 1):
            for s in range(1, n):
                if r == s:
                    continue
                trials += 1
                total = 0
                for o in range(m):
                    deltas[o] = swap_delta(p, F[o], D, r, s)
                    total += deltas[o]
                value = total if criterion == 0 else deltas[criterion - 1]
                if value <= 0:
                    tmp = p[r]
                    p[r] = p[s]
                    p[s] = tmp
    return trials


class MQAP(Problem):
    def __init__(self, n, m, D, F):
        super().__init__(n_var=n, n_obj=m, xl=0, xu=n - 1, vtype=int)
        self.D = D
        self.F = F
        self.full = 0
        self.swaps = 0      # counted here: pymoo deep-copies the algorithm, and with it the repair

    def _evaluate(self, X, out, *args, **kwargs):
        X = X.astype(np.int64)
        self.full += X.shape[0]
        out['F'] = costs(X, self.D, self.F).astype(float)


class Greedy2Opt(Repair):
    """Local search as a repair, applied by pymoo to every offspring before it is evaluated."""

    def __init__(self, rate, rng):
        super().__init__()
        self.rate = rate
        self.rng = rng

    def _do(self, problem, X, **kwargs):
        X = np.ascontiguousarray(X.astype(np.int64))
        criterion = int(self.rng.integers(0, problem.n_obj + 1))   # one criterion per generation
        apply = self.rng.random(X.shape[0]) < self.rate if self.rate < 1.0 else np.ones(X.shape[0], dtype=np.bool_)
        problem.swaps += greedy_2opt(X, problem.F, problem.D, criterion, apply)
        return X


def build(name, m, pop, rate, rng):
    local = Greedy2Opt(rate, rng) if name.endswith('-ls') else None
    common = dict(sampling=PermutationRandomSampling(), crossover=OrderCrossover(), mutation=InversionMutation())
    base = name.replace('-ls', '')
    if base == 'nsga2':
        algorithm = NSGA2(pop_size=pop, eliminate_duplicates=True, repair=local, **common)
    elif base == 'nsga3':
        ref_dirs = get_reference_directions('energy', m, pop, seed=1)
        algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=pop, eliminate_duplicates=True, repair=local, **common)
    else:
        ref_dirs = get_reference_directions('energy', m, pop, seed=1)
        algorithm = MOEAD(ref_dirs=ref_dirs, n_neighbors=min(20, pop - 1), prob_neighbor_mating=0.9,
                          repair=local, **common)
    return algorithm, local


def one_run(job):
    path, name, pop, gen, evals, seconds_budget, rate, seed = job
    n, m, D, F = load_instance(path)
    problem = MQAP(n, m, D, F)
    rng = np.random.default_rng(seed)
    algorithm, _ = build(name, m, pop, rate, rng)
    if seconds_budget:
        termination = TimeBasedTermination(seconds_budget)
    else:
        termination = ('n_eval', evals) if evals else ('n_gen', gen)
    begin = time.perf_counter()
    result = minimize(problem, algorithm, termination, seed=seed, verbose=False)
    seconds = time.perf_counter() - begin
    # The distinct non-dominated solutions of the final population, with their exact integer costs.
    X = np.asarray(result.pop.get('X'), dtype=np.int64)
    C = costs(X, D, F)
    front = NonDominatedSorting().do(C.astype(float), only_non_dominated_front=True)
    solutions = {}
    for i in front:
        solutions[tuple(int(v) for v in X[i])] = [int(v) for v in C[i]]
    return {'seed': seed, 'seconds': seconds, 'full_evaluations': problem.full,
            'swap_evaluations': problem.swaps, 'generations': result.algorithm.n_gen,
            'front': [[list(k), v] for k, v in solutions.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[4])
    parser.add_argument('instance', help='.dat file')
    parser.add_argument('--algorithm', choices=ALGORITHMS, default='nsga2')
    parser.add_argument('--pop', type=int, default=64, help='population size (default 64)')
    parser.add_argument('--gen', type=int, default=300, help='generations, when --evals is not given (default 300)')
    parser.add_argument('--evals', type=int, default=0, help='budget of full evaluations instead of --gen')
    parser.add_argument('--seconds', type=float, default=0.0, help='wall-time budget per run instead of --gen')
    parser.add_argument('--greedy-rate', type=float, default=1.0, help='share of the offspring the -ls variants improve')
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=1, help='seed of the first run; run i uses seed + i')
    parser.add_argument('--jobs', type=int, default=os.cpu_count(), help='runs executed at once, one per process')
    parser.add_argument('--output', help='result file (default results/baselines/<instance>_<algorithm>.txt)')
    args = parser.parse_args()

    instance = os.path.splitext(os.path.basename(args.instance))[0]
    output = args.output or os.path.join('results', 'baselines', '%s_%s.txt' % (instance, args.algorithm))
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    jobs = [(args.instance, args.algorithm, args.pop, args.gen, args.evals, args.seconds, args.greedy_rate, args.seed + i)
            for i in range(args.runs)]
    with ProcessPoolExecutor(max_workers=min(args.jobs, args.runs)) as pool:
        runs = list(pool.map(one_run, jobs))

    with open(output, 'w', encoding='utf-8') as f:
        for run in runs:
            f.write('{\n')
            for permutation, values in run['front']:
                f.write("'%s': [%s],\n" % (''.join(str(v) for v in permutation), ', '.join(str(v) for v in values)))
            f.write('},\n')
    meta = {'instance': instance, 'algorithm': args.algorithm, 'pop': args.pop, 'gen': args.gen, 'evals': args.evals,
            'seconds': args.seconds,
            'greedy_rate': args.greedy_rate,
            'runs': [{k: r[k] for k in ('seed', 'seconds', 'full_evaluations', 'swap_evaluations', 'generations')}
                     | {'front_size': len(r['front'])} for r in runs]}
    with open(os.path.splitext(output)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    seconds = [r['seconds'] for r in runs]
    print('%s %s: %d runs, %.2f s per run (median), %d full evaluations and %d swap deltas per run, fronts of %d-%d'
          % (instance, args.algorithm, len(runs), float(np.median(seconds)), runs[0]['full_evaluations'],
             runs[0]['swap_evaluations'], min(len(r['front']) for r in runs), max(len(r['front']) for r in runs)))
    print('written', output)


if __name__ == '__main__':
    sys.exit(main())
