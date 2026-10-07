# Comparison with the CPU and with other MOEAs

**English** | [Español](COMPARACION.md)

This document belongs to the branch `develop_comparison_vs_cpu_and_moeas`, which starts from
`develop_large_population_multiblock` and keeps everything its [README](README.md) describes. It answers two
questions the rest of the project left open:

1. **How much does the GPU contribute?** The speed-ups of the README are measured against the original 2019
   code, whose time went to API calls. Here the same algorithm runs on the GPU and on a multicore CPU.
2. **How good is the result against other algorithms?** Here `cuda_mqap` is compared with NSGA-II, NSGA-III and
   MOEA/D from [pymoo](https://pymoo.org), each with and without the same greedy 2-opt, at **equal work** and at
   **equal wall time**, on the 23 Knowles–Corne instances and on 16 larger instances with n = 60.

---

## Contents

1. [What this branch adds](#what-this-branch-adds)
2. [Requirements](#requirements)
3. [Protocol](#protocol)
4. [How to reproduce it](#how-to-reproduce-it)
5. [Results](#results)
6. [What the results say](#what-the-results-say)
7. [Limitations](#limitations)

---

## What this branch adds

| Piece | Where | What it does |
|---|---|---|
| CPU version of the algorithm | `src/solver_cpu.cpp`, `--cpu`, `--threads N` | The same algorithm with OpenMP: same data layout, Fisher-Yates initialization, O(n²) fitness, survival by dominator counting and front peeling, crowding, tournament and mutations, greedy 2-opt with the O(n) delta, the same pair traversal and the same gate. Only the random generator differs (xoshiro256** per row instead of Philox) |
| Shared gate | `include/gate.h` | `greedyApplies` without CUDA dependencies, so the kernels, the CPU version and the host share it |
| Work counter | `SolveStats`, line `Evaluations:` of the output | Full O(n²) evaluations and O(n) swap deltas of the greedy, counted on the host from the deterministic gate, so that different algorithms can be given the same budget |
| Garrett instances | `scripts/fetch_gar60.ps1` → `data/gar60/` | Downloads the 16 instances with n = 60 and 2 or 3 objectives (Garrett and Dasgupta, 2009), as published with PasMoQAP (Sanhueza et al., CEC 2017). That repository declares no license, so they are not versioned (`data/` is ignored) |
| pymoo baselines | `scripts/baselines/pymoo_mqap.py` | NSGA-II, NSGA-III and MOEA/D with permutation operators, and each one with the greedy 2-opt of `cuda_mqap` as a repair (`-ls`). Budget in generations, evaluations or seconds; output in the result file format |
| Campaign | `scripts/run_comparison.ps1` | The four blocks of the [protocol](#protocol), resumable |
| Analysis | `scripts/analyze_comparison.py` | Hypervolume, coverage, Mann–Whitney with Holm correction, Vargha–Delaney A12, wins/ties/losses and Friedman test |
| Tests | `tests/test_kernels.cu` | Two checks of the CPU version: valid permutations, exact fitness, ranks consistent with dominance, a front without repetitions and the same work count as the GPU version. **29 checks** in all |

The CPU version is the same algorithm, not an approximation, and that was checked: 30 runs per version on
KC20-2fl-1rl and KC20-2fl-3uni (P = 64, 300 generations) do not differ in coverage (Mann–Whitney p = 0.14 and
0.51), and the only hypervolume difference with p < 0.05 (0.085 points, p = 0.022) is no longer significant with
the Holm correction over the four tests.

The baselines use the same cost function as `cuda_mqap`, which reproduces the 374 published optimal costs of the
KC10 instances, and their memetic variants use the same local search: the same traversal of pairs, the same
acceptance and one criterion per generation drawn among the sum and each objective.

## Requirements

- The Release build of the [README](README.md#build). The CPU version needs OpenMP, which Visual Studio and the
  CMake build enable on their own.
- Python 3 with `numpy`, `scipy`, `numba` and `pymoo` (measured with pymoo 0.6.2):

```
pip install numpy scipy numba pymoo
```

- A connection, once, to download the Garrett instances: `.\scripts\fetch_gar60.ps1`.

## Protocol

Thirty runs per algorithm and instance, seed 20261006, every final front measured against the same reference
front of its instance.

| Block | What it measures | Settings |
|---|---|---|
| `speedup` | The same algorithm on the GPU and on 12 CPU threads | KC10-2fl-1rl, KC20-2fl-1rl and KC30-3fl-1rl; P = 64 … 16,384; 20 generations; median of 3 runs; greedy on every offspring |
| `budget` | **Same work** on the 23 Knowles–Corne instances | P = 64 and the generations of the original version (70 on KC10, 300 on KC20 and KC30); `cuda_mqap --untuned` |
| `gar60` | Same work **and** same wall time on the 16 Garrett instances | Same work: P = 64, 100 generations. Same time: `cuda_mqap` with P = 4096 and 100 generations; each pymoo run, with P = 100, gets the wall time of one such run (11 to 16 s) |
| `time` | **Same wall time** on the 23 Knowles–Corne instances | `cuda_mqap` with the [default call](README.md#the-default-call-of-each-instance) of each instance; each pymoo run, with P = 100, gets the wall time of one such run |

- **Baselines.** pymoo 0.6.2: random permutation sampling, order crossover, inversion mutation; duplicate
  elimination in NSGA-II and NSGA-III; Riesz energy reference directions for NSGA-III and MOEA/D, 20 neighbours
  and mating probability 0.9 in MOEA/D. The `-ls` variants apply the greedy 2-opt of `cuda_mqap` to every
  offspring. At equal work they perform **at least as many** swap evaluations as `cuda_mqap`, because pymoo also
  repairs the initial population and the offspring it regenerates to replace duplicates.
- **pymoo runs are single-threaded** and six run at a time, one per physical core, so that a time budget means the
  same for each of them.
- **Reference fronts.** The published optimum on KC10, [`reference/v0.4`](reference/README.md) on KC20 and KC30,
  and on the Garrett instances, which have none, the non-dominated union of every run of every algorithm and
  protocol of the campaign.
- **Statistics.** Each baseline against `cuda_mqap`: two-sided Mann–Whitney with the Holm correction over the six
  baselines of the instance, and the Vargha–Delaney A12 effect size. Over the instances: wins/ties/losses at
  corrected p < 0.05 and the Friedman test on the mean ranks.

## How to reproduce it

```
:: Once: the Garrett instances and the Python packages
powershell -ExecutionPolicy Bypass -File scripts\fetch_gar60.ps1
pip install numpy scipy numba pymoo

:: The four blocks, in the order they were run; an interrupted campaign resumes where it stopped
powershell -ExecutionPolicy Bypass -File scripts\run_comparison.ps1 -Block speedup,budget,gar60,time

:: The summaries (Markdown and JSON in results\comparison)
python scripts\analyze_comparison.py results\comparison --block budget
python scripts\analyze_comparison.py results\comparison --block gar60 --protocol p64
python scripts\analyze_comparison.py results\comparison --block gar60 --protocol time
python scripts\analyze_comparison.py results\comparison --block time
```

On the RTX 2060 and the i5-11400 of the measurement the blocks took 4 min (`speedup`), 1 h 12 min (`budget`),
5 h 15 min (`gar60`) and 18 h 19 min (`time`, dominated by the seven KC30 instances, whose default call takes
104 s per run). Every result goes to `results\comparison\<block>\<instance>\<algorithm>_<protocol>.txt`, in the
result file format, with a `.json` (pymoo) or `.log` (`cuda_mqap`) next to it.

The algorithms can also be run one by one:

```
build\x64\Release\cuda_mqap.exe mQAPData\KC20-2fl-1rl.dat --untuned --runs 30 --cpu --threads 12
python scripts\baselines\pymoo_mqap.py mQAPData\KC20-2fl-1rl.dat --algorithm nsga2-ls --pop 64 --gen 300 --runs 30
python scripts\baselines\pymoo_mqap.py data\gar60\Gar60-3fl-1rl.dat --algorithm moead-ls --pop 100 --seconds 16
```

## Results

### 1. The same algorithm on the GPU and on the CPU

Time per generation, GPU / CPU, and speed-up of the GPU (RTX 2060 against an i5-11400 with 12 OpenMP threads):

| Instance | P = 64 | P = 256 | P = 1,024 | P = 4,096 | P = 16,384 |
|---|---|---|---|---|---|
| KC10-2fl-1rl | 0.56 / 0.4 ms · **0.7×** | 0.80 / 0.9 ms · **1.1×** | 1.47 / 5.3 ms · **3.6×** | 3.66 / 52.0 ms · **14.2×** | 11.74 / 749.9 ms · **63.9×** |
| KC20-2fl-1rl | 0.82 / 1.4 ms · **1.7×** | 1.34 / 3.2 ms · **2.4×** | 2.95 / 12.4 ms · **4.2×** | 6.75 / 85.3 ms · **12.6×** | 27.36 / 944.8 ms · **34.5×** |
| KC30-3fl-1rl | 1.59 / 6.4 ms · **4.0×** | 2.38 / 14.2 ms · **5.9×** | 5.23 / 46.3 ms · **8.8×** | 16.71 / 217.3 ms · **13.0×** | 67.33 / 1567.1 ms · **23.3×** |

At the population of the original version the GPU brings little, and on KC10 it is slower: a generation is a
fraction of a millisecond on either device. The ratio grows with the population, to 23–64× at P = 16,384, where
the CPU needs 0.75 to 1.6 s per generation. A run of 300 generations at the population cap, routine on the GPU,
would take hours on the CPU.

### 2. Same work on the Knowles–Corne instances

P = 64, 70 or 300 generations, 30 runs, 23 instances. W/T/L: wins, ties and losses of `cuda_mqap` against each
baseline (Holm-corrected Mann–Whitney, p < 0.05):

| Algorithm | Mean rank, coverage | Mean rank, hypervolume | cuda_mqap W/T/L, coverage | cuda_mqap W/T/L, hypervolume |
|---|---|---|---|---|
| **cuda_mqap** | **2.04** | **1.76** | — | — |
| NSGA-II+LS | 2.52 | 2.15 | 9 / 10 / 4 | 13 / 7 / 3 |
| NSGA-III+LS | 2.26 | 2.30 | 9 / 10 / 4 | 12 / 9 / 2 |
| MOEA/D+LS | 3.57 | 3.78 | 16 / 5 / 2 | 22 / 1 / 0 |
| NSGA-II | 5.65 | 5.65 | 21 / 2 / 0 | 23 / 0 / 0 |
| NSGA-III | 5.83 | 6.00 | 21 / 2 / 0 | 23 / 0 / 0 |
| MOEA/D | 6.13 | 6.35 | 21 / 2 / 0 | 23 / 0 / 0 |

Friedman p = 4.0·10⁻²¹ (coverage) and 7.9·10⁻²³ (hypervolume).

- `cuda_mqap` has the best mean rank in both indicators.
- **The local search is what separates the algorithms.** Without it, the three pymoo algorithms cover less than
  17 % of the reference front on every instance, and 0 % on every KC20 and KC30 instance.
- **Against the memetic variants, which share its local search, it is close.** It wins on the real-like KC20
  instances and on KC30-2fl-1rl (37.0 % against 31.8 % coverage on KC20-2fl-1rl, 3.7 % against 1.5 % on
  KC30-2fl-1rl), and loses on KC10-2fl-3rl, -3uni and -5rl and on KC20-2fl-2uni (17.5 % against 34.2 %), where
  pymoo's order crossover and duplicate elimination keep more distinct solutions in a population of 64.
- On the three-objective KC30 instances no algorithm covers more than 0.1 % of the reference front at this budget,
  so the ranking there rests on the hypervolume.

The full per-instance tables are in `results\comparison\summary_budget_p64.md`.

### 3. The Garrett instances (n = 60)

The reference front of these instances is the union of the campaign, to which `cuda_mqap` contributes most
points, so the coverage is not informative and the comparison rests on the **hypervolume**. The hypervolume is
Pareto compliant, so the ordering is valid; its absolute values depend on that union.

| Protocol | Mean rank of cuda_mqap | Best baseline (mean rank) | cuda_mqap W/T/L against each of the six | Friedman p |
|---|---|---|---|---|
| Same work (P = 64, 100 generations) | **1.19** | NSGA-II+LS (2.81) | 15 / 1 / 0 | 6.9·10⁻¹⁶ |
| Same wall time (11–16 s per run) | **1.00** | NSGA-II+LS (2.59) | **16 / 0 / 0** | 4.2·10⁻¹⁶ |

Mean hypervolume share (%) of `cuda_mqap` and of the best baseline of each instance:

| Instance | cuda_mqap, same work | best baseline | cuda_mqap, same time | best baseline |
|---|---|---|---|---|
| Gar60-2fl-1rl | **93.5** | 90.2 (MOEA/D+LS) | **98.5** | 89.5 (NSGA-II+LS) |
| Gar60-2fl-1uni | **85.8** | 83.3 (MOEA/D+LS) | **95.2** | 81.0 (NSGA-II+LS) |
| Gar60-2fl-2rl | **93.7** | 89.7 (NSGA-III+LS) | **98.6** | 88.6 (NSGA-III+LS) |
| Gar60-2fl-2uni | **81.8** | 79.2 (MOEA/D+LS) | **93.9** | 75.2 (MOEA/D+LS) |
| Gar60-2fl-3rl | **92.2** | 88.8 (MOEA/D+LS) | **98.3** | 87.1 (NSGA-II+LS) |
| Gar60-2fl-3uni | **68.4** | 65.3 (MOEA/D+LS) | **88.4** | 59.5 (MOEA/D+LS) |
| Gar60-2fl-4rl | **93.8** | 90.5 (NSGA-II+LS) | **98.5** | 90.4 (NSGA-II+LS) |
| Gar60-2fl-4uni | **91.1** | 89.9 (NSGA-II+LS) | **96.5** | 89.9 (NSGA-II+LS) |
| Gar60-2fl-5rl | **88.0** | 81.9 (MOEA/D+LS) | **97.8** | 80.3 (NSGA-II+LS) |
| Gar60-2fl-5uni | **0.0** | 0.0 (MOEA/D+LS) | **25.3** | 0.0 (MOEA/D+LS) |
| Gar60-3fl-1rl | **80.1** | 75.8 (NSGA-II+LS) | **95.0** | 77.1 (NSGA-II+LS) |
| Gar60-3fl-1uni | **68.3** | 63.9 (MOEA/D+LS) | **89.7** | 63.7 (NSGA-III+LS) |
| Gar60-3fl-2rl | **79.2** | 74.4 (NSGA-II+LS) | **94.7** | 76.0 (NSGA-II+LS) |
| Gar60-3fl-2uni | **77.9** | 76.2 (NSGA-II+LS) | **91.0** | 76.7 (NSGA-II+LS) |
| Gar60-3fl-3rl | **78.1** | 72.8 (NSGA-II+LS) | **94.8** | 73.4 (NSGA-III+LS) |
| Gar60-3fl-3uni | **51.7** | 45.6 (MOEA/D+LS) | **81.5** | 38.3 (MOEA/D+LS) |

- **At n = 60, `cuda_mqap` wins even at equal work**, which it did not on KC10.
- **At equal time the gap opens**, most of all with three objectives: 81.5–95.0 % against 38.3–77.1 % for the best
  baseline of each instance.
- The only tie is **Gar60-2fl-5uni** at equal work, where every algorithm stays at 0 %: its flow matrices have a
  correlation of 0.8 and almost no run dominates the region the reference point bounds. At equal time `cuda_mqap`
  reaches 25.3 % and the baselines stay at 0. PasMoQAP also obtained its lowest values on this instance.

### 4. Same wall time on the Knowles–Corne instances

`cuda_mqap` with the [default call](README.md#the-default-call-of-each-instance) of each instance; each pymoo run,
with P = 100, receives the wall time of one such run. 30 runs, 23 instances.

| Algorithm | Mean rank, coverage | Mean rank, hypervolume | cuda_mqap W/T/L, coverage | cuda_mqap W/T/L, hypervolume |
|---|---|---|---|---|
| **cuda_mqap** | **1.00** | **1.00** | — | — |
| NSGA-II+LS | 2.59 | 2.46 | 22 / 1 / 0 | 22 / 1 / 0 |
| NSGA-III+LS | 2.63 | 2.59 | 22 / 1 / 0 | 22 / 1 / 0 |
| MOEA/D+LS | 3.98 | 4.00 | 22 / 1 / 0 | 22 / 1 / 0 |
| NSGA-II | 5.74 | 5.35 | 23 / 0 / 0 | 23 / 0 / 0 |
| NSGA-III | 5.78 | 5.67 | 23 / 0 / 0 | 23 / 0 / 0 |
| MOEA/D | 6.28 | 6.93 | 23 / 0 / 0 | 23 / 0 / 0 |

Friedman p = 1.1·10⁻²⁴ (coverage) and 6.7·10⁻²⁶ (hypervolume).

Per instance, the mean of `cuda_mqap` and of the best baseline in each indicator:

| Instance | Default call: time per run | cuda_mqap, coverage | best baseline | cuda_mqap, hypervolume | best baseline |
|---|---|---|---|---|---|
| KC10-2fl-1rl | 0.9 s | **99.7 %** | 64.1 % (NSGA-II+LS) | **100.00 %** | 99.83 % (NSGA-II+LS) |
| KC10-2fl-1uni | 0.2 s | **97.9 %** | 64.6 % (NSGA-III+LS) | **99.96 %** | 94.81 % (NSGA-III+LS) |
| KC10-2fl-2rl | 0.2 s | **100.0 %** | 75.8 % (NSGA-II+LS) | **100.00 %** | 96.83 % (NSGA-II+LS) |
| KC10-2fl-2uni | 0.2 s | **100.0 %** | 90.0 % (NSGA-II+LS) | **100.00 %** | 90.00 % (NSGA-II+LS) |
| KC10-2fl-3rl | 0.8 s | **100.0 %** | 48.3 % (NSGA-II+LS) | **100.00 %** | 98.85 % (NSGA-II+LS) |
| KC10-2fl-3uni | 6.8 s | **100.0 %** | 61.5 % (NSGA-II+LS) | **100.00 %** | 99.65 % (NSGA-II+LS) |
| KC10-2fl-4rl | 0.8 s | **99.9 %** | 37.1 % (NSGA-II+LS) | **100.00 %** | 99.06 % (NSGA-II+LS) |
| KC10-2fl-5rl | 0.8 s | **100.0 %** | 42.0 % (NSGA-II+LS) | **100.00 %** | 99.68 % (NSGA-II+LS) |
| KC20-2fl-1rl | 33.1 s | **95.4 %** | 65.8 % (NSGA-III+LS) | **99.99 %** | 99.76 % (NSGA-II+LS) |
| KC20-2fl-1uni | 39.0 s | **97.7 %** | 34.1 % (NSGA-III+LS) | **99.99 %** | 98.90 % (NSGA-III+LS) |
| KC20-2fl-2rl | 34.1 s | **59.2 %** | 28.7 % (NSGA-II+LS) | **99.75 %** | 99.16 % (NSGA-II+LS) |
| KC20-2fl-2uni | 32.2 s | **100.0 %** | 61.2 % (NSGA-III+LS) | **100.00 %** | 98.53 % (NSGA-II+LS) |
| KC20-2fl-3rl | 33.2 s | **58.2 %** | 21.4 % (NSGA-II+LS) | **99.16 %** | 97.72 % (NSGA-II+LS) |
| KC20-2fl-3uni | 47.2 s | **73.7 %** | 14.9 % (NSGA-III+LS) | **99.68 %** | 98.41 % (NSGA-II+LS) |
| KC20-2fl-4rl | 31.0 s | **46.8 %** | 25.2 % (NSGA-III+LS) | **98.74 %** | 95.22 % (NSGA-III+LS) |
| KC20-2fl-5rl | 33.9 s | **62.7 %** | 22.6 % (NSGA-II+LS) | **99.83 %** | 99.34 % (NSGA-II+LS) |
| KC30-2fl-1rl | 53.5 s | **45.0 %** | 10.3 % (NSGA-III+LS) | **99.60 %** | 98.44 % (NSGA-II+LS) |
| KC30-3fl-1rl | 103.3 s | **14.6 %** | 0.1 % (MOEA/D+LS) | **96.79 %** | 88.14 % (NSGA-III+LS) |
| KC30-3fl-1uni | 104.6 s | **14.3 %** | 0.0 % (NSGA-III+LS) | **95.40 %** | 76.43 % (NSGA-III+LS) |
| KC30-3fl-2rl | 103.2 s | **23.7 %** | 0.1 % (MOEA/D+LS) | **98.63 %** | 91.28 % (NSGA-III+LS) |
| KC30-3fl-2uni | 103.2 s | **43.0 %** | 0.7 % (NSGA-III+LS) | **98.33 %** | 82.61 % (NSGA-III+LS) |
| KC30-3fl-3rl | 103.7 s | **26.9 %** | 0.0 % (NSGA-II+LS) | **98.45 %** | 87.95 % (NSGA-III+LS) |
| KC30-3fl-3uni | 104.8 s | **20.4 %** | 0.0 % (NSGA-II+LS) | **96.25 %** | 77.23 % (NSGA-III+LS) |

- `cuda_mqap` has mean rank **1.00** in both indicators and wins on 22 or 23 of the 23 instances against every
  baseline. The only tie is KC10-2fl-2uni, whose optimal front has a single point, which the memetic baselines also
  find in most runs.
- **KC10**: in under 7 s per run, `cuda_mqap` finds 97.9–100 % of the published optimal front; with the same time
  the best baseline finds 37.1–90.0 %.
- **KC20 and KC30**: 45.0–100 % of the reference front on the two-objective instances against 10.3–65.8 % for the
  best baseline; on the three-objective KC30 instances, 14.3–43.0 % against at most 0.7 %, and 95.4–98.6 % of the
  reference hypervolume against 76.4–91.3 %.
- **The best known fronts can still improve.** On 11 instances the campaign found points that `reference/v0.4` does
  not dominate: mostly from `cuda_mqap` (from 2 on KC20-2fl-5rl to 5342 on KC30-3fl-3rl), but also some from the
  baselines (up to 15 from NSGA-II on KC20-2fl-3rl). They are candidates for a `v0.5`, which has not been built yet;
  every figure here is measured against `v0.4`.

## What the results say

- **The GPU does not make a small-population memetic NSGA-II much faster; it makes large populations affordable.**
  At P = 64 the CPU version is as fast; at P = 16,384 the GPU is 23–64 times faster.
- **At equal work the algorithm is competitive, not dominant.** It has the best mean rank, but against pymoo's
  memetic NSGA-II and NSGA-III the differences are mostly ties, and on several small or uniform instances pymoo's
  crossover with duplicate elimination is better. What separates every memetic algorithm from its plain
  counterpart is the local search, in line with the mQAP literature.
- **At equal time the advantage becomes systematic**: mean rank 1.00 on the 23 Knowles–Corne instances and on the
  16 Garrett instances, because the GPU spends the same seconds on populations 40 to
  650 times larger than the baselines' 100. The quality gains of this project come mainly from the population the
  GPU makes affordable, together with the tuned intensity of the local search, rather than from a better operator
  design.

## Limitations

- The baselines are general-purpose MOEAs from one library, with their default operators. The specialized mQAP
  algorithms (Garrett and Dasgupta; Drugan's stochastic Pareto local search; PasMoQAP) could not be run, because
  their code is not public. The published PasMoQAP results use a hypervolume normalized against its own runs and
  are not comparable with these values.
- In the equal-time protocol each pymoo run uses one CPU core, since pymoo does not parallelize a single run,
  while `cuda_mqap` uses the whole GPU. Block 1 bounds what a parallel CPU implementation could gain.
- On the Garrett instances the reference front is the union of this campaign, so only the hypervolume is used
  there.
- One GPU (RTX 2060) and one CPU (i5-11400): relative results should transfer, absolute times should not.
- Instances with four objectives are not supported: the kernels handle 2 or 3, and at n = 60 four flow matrices
  would not fit in 64 KB of shared memory.
