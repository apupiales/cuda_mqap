# cuda_mqap — Parallel NSGA-II + Adapted Greedy 2-opt in CUDA for the mQAP

**English** | [Español](LEEME.md)

CUDA C implementation of the multiobjective evolutionary algorithm **NSGA-II**, combined with an
**adapted Greedy 2-opt** local search, to solve instances of the **multiobjective Quadratic Assignment
Problem** (mQAP). The fitness evaluation, the NSGA-II steps, selection, mutation and local search run on
the GPU.

This branch holds the **original implementation** (single translation unit `kernel.cu`, instances compiled
into the program). Rewritten and optimized versions, with runtime instance loading, tests, the fixes of the
[known issues](#known-issues-of-the-original-code) and larger populations, live in other branches: see
[Repository branches](#repository-branches).

---

## Contents

1. [Features](#features)
2. [Repository branches](#repository-branches)
3. [The problem: mQAP](#the-problem-mqap)
4. [How the program works](#how-the-program-works)
5. [Repository layout](#repository-layout)
6. [Requirements](#requirements)
7. [Open in Visual Studio 2026 (plug and play)](#open-in-visual-studio-2026-plug-and-play)
8. [Configuration](#configuration)
9. [Output](#output)
10. [Result analysis](#result-analysis)
11. [Measured parallelization (GPU and CPU use)](#measured-parallelization-gpu-and-cpu-use)
12. [Known issues of the original code](#known-issues-of-the-original-code)
13. [Credits and citations](#credits-and-citations)
14. [License](#license)

---

## Features

1. Creation of the initial population (random permutations).
2. Fitness calculation for each objective.
3. Parallel NSGA-II:
   - 3.1 Dominance matrix.
   - 3.2 Total dominance.
   - 3.3 Pareto fronts.
   - 3.4 Rank.
   - 3.5 Population fitness ordered by objective (bitonic sort, used in the crowding calculation).
   - 3.6 Crowding distance.
   - 3.7 Offspring population.
4. Binary tournament selection.
5. Mutation: exchange (two passes) and transposition (reversal of a segment).
6. Adapted Greedy 2-opt local search.

Instances with 2 or 3 objectives and up to 60 facilities.

---

## Repository branches

The original program lives in this branch. The other branches keep the same algorithm (NSGA-II with an
adapted Greedy 2-opt on the mQAP) but rewrite its implementation; each one builds on the previous one.

| Branch | What it contains | Main differences from this original version |
|---|---|---|
| `master` (this one) | Original implementation (2019): everything in `kernel.cu`, with the instances compiled into the program, plus the Visual Studio project and this documentation | — |
| [`develop_with_claude_opus_5`](https://github.com/apupiales/cuda_mqap/tree/develop_with_claude_opus_5) | Modular rewrite and GPU optimization of the same algorithm | Code split into `include/`, `src/` and `tests/`; instances read from the `.dat` files at runtime and parameters on the command line; fitness in O(n²) instead of three dense matrix products; 2-opt with O(n) delta evaluation; the whole NSGA-II inside one block per run; three kernel launches per generation with no host synchronization; `--runs` executes independent runs concurrently; automated tests, `--verify` and the [known issues](#known-issues-of-the-original-code) fixed. KC30-3fl-1rl: 42.4 s → 0.14 s on an RTX 2060 |
| [`develop_p512_single_block`](https://github.com/apupiales/cuda_mqap/tree/develop_p512_single_block) | The previous one, with a survival that does not store the dominance matrix | Population up to 512 on any GPU (the shared memory grows linearly with P instead of quadratically). Identical results and lower GPU time for P ≤ 256 |
| [`develop_large_population_multiblock`](https://github.com/apupiales/cuda_mqap/tree/develop_large_population_multiblock) | The refactored version with the survival split across several blocks | Population up to 65536: cooperative launch for the Pareto fronts and segmented sorts (CUB) for crowding and selection. For P ≤ 256 it uses the same single-block kernel, with identical results |

The two last branches exist because a larger population explores more of the Pareto front: on
KC10-2fl-3uni, the average share of the optimal front found per run goes from 68 % with P = 256 to 82 %
with P = 4096 (100 runs of each). Their README documents the measurements and the limits of each GPU.

---

## The problem: mQAP

`n` facilities must be assigned to `n` locations. A chromosome is a permutation (`short[FACILITIES_LOCATIONS]`).
Each objective `k` has its own flow matrix `Fk`, and all objectives share the distance matrix `D`. The
program computes each objective as the trace of a matrix product:

```
f_k = Trace(Fk · X · Dᵀ · Xᵀ)          k = 1..m   (m = 2 or 3)
```

where `X` is the permutation written as a binary `n × n` matrix. All objectives are minimized, so the
result is an approximation of the **Pareto front** (the set of non-dominated solutions).

The instances are the KC10, KC20 and KC30 test suites of Knowles and Corne (see
[Credits and citations](#credits-and-citations)), stored in `mQAPData/*.dat`. For the ten-facility
instances, `mQAPData/*.PO` contains the published Pareto optimal fronts.

---

## How the program works

Everything lives in `kernel.cu`. `main` repeats the whole genetic algorithm `TIMES` times; each run
executes `ITERATIONS + 1` iterations over a population array of `NSGA2_POPULATION_SIZE = 2 · POPULATION_SIZE`
chromosomes (`Rt` = parents + offspring):

1. **`parallelPopulationFitnessCalculation`**: builds the binary matrix of every chromosome and runs the
   product chain `multiplicationWithFlowMatrix` → `multiplicationWithTranposedDistanceMatrix` →
   `matrixMultiplication` → `calculateTrace`. Fitness rows have `OBJECTIVES + 1` columns; the extra
   column keeps the original index so that it survives sorting.
2. **`parallelNSGA2`**:
   - dominance matrix (`get2ObjectivePopulationDominanceMatrix` or `get3ObjectivePopulationDominanceMatrix`, so only 2 and 3 objectives are supported);
   - total dominance, Pareto fronts and rank;
   - per-objective bitonic sort (`bitonicSortStep`) and crowding distance;
   - selection of the next `POPULATION_SIZE` parents by front and crowding distance.
3. **`BinaryTournamentSelection`**: the winners are copied to the second half of the population array.
4. **Variation of the offspring half** `[POPULATION_SIZE, NSGA2_POPULATION_SIZE)`:
   - two `exchangeMutation` passes;
   - `transpositionMutation`;
   - `greedy2Opt`, whose acceptance criterion is chosen at random between a single objective and the
     average of the objectives.

At the end of each run the final population is appended to the result file (see [Output](#output)).

Conventions: host/device mirrored arrays use the `h_` / `d_` prefixes. `kernel.cu` contains commented-out
blocks with fixed test populations, used to validate the fitness values (for example, F0 = 228322 and
F1 = 193446 for a fixed KC10 permutation).

---

## Repository layout

| Path | Content |
|---|---|
| `kernel.cu` | The whole program: kernels, host code and `main` (single translation unit) |
| `general_dev_settings.cu` | `TIMES` (repetitions of the whole run), `DEV_MODE` and the `PRINT_*` debug flags |
| `settings_KC*_*fl_*.cu` | One file per instance: sizes, population, iterations, mutation probabilities and the instance matrices in `__constant__` memory |
| `mQAPData/` | Instances (`.dat`), Pareto optimal fronts (`.PO`) and their provenance (`README.txt`) |
| `mQAPMetrics/` | Node.js scripts: distance metric and 3D plots |
| `benchmarks/` | Measurement of the GPU and CPU use of the four versions (see [Measured parallelization](#measured-parallelization-gpu-and-cpu-use)) |
| `comparative_results_kcX_datasets.xlsx` | Comparative results of the instances |
| `cuda_mqap.slnx`, `cuda_mqap.vcxproj` | Visual Studio solution and project |
| `cuda_mqap.props`, `cuda_toolkit.props` | Project settings and detection of the installed CUDA version |
| `.vsconfig`, `.gitattributes` | Visual Studio components and line endings |

The `.cu` files other than `kernel.cu` are included with `#include` and are never compiled on their own.

---

## Requirements

- NVIDIA GPU with *compute capability* ≥ 7.5 (GeForce RTX 20xx or newer) and an up-to-date driver.
- **Visual Studio 2026** with the *Desktop development with C++* workload. When the solution is opened,
  Visual Studio reads `.vsconfig` and offers to install any missing component.
- **CUDA Toolkit 12.x or 13.x** (≥ 11.8), installed **after** Visual Studio so that its *Visual Studio
  Integration* is added to it. The project was developed with CUDA 13.4.

---

## Open in Visual Studio 2026 (plug and play)

```
git clone https://github.com/apupiales/cuda_mqap.git
cd cuda_mqap
start cuda_mqap.slnx
```

1. Select `Release | x64` and press **F5** (or Ctrl+F5).
2. The program runs the instance selected in `kernel.cu` (KC10-2fl-1rl by default), prints the initial
   population and the final solution, and writes `result_KCX_Yfl_Z_nsga2_greedy_2opt.txt` in the
   repository root (ignored by git).
3. The executable is written to `build\x64\<Configuration>\cuda_mqap.exe`.

Nothing depends on the machine where the project was created:

| What | How it adapts |
|---|---|
| CUDA version | `cuda_toolkit.props` takes it from `CUDA_PATH` (e.g. `...\CUDA\v12.6` → `CUDA 12.6.props`). To use another installed version: `set CudaVersion=12.6` before opening Visual Studio, or `msbuild /p:CudaVersion=12.6` |
| Missing CUDA integration | The build stops with a message that explains how to fix it |
| C++ toolset | `$(DefaultPlatformToolset)` of the Visual Studio that opens it (v145 in VS 2026); Windows SDK `10.0` (latest installed) |
| GPU | Native code for `sm_75`, `sm_80`, `sm_86` and `sm_89`, plus PTX that the driver compiles for newer GPUs (RTX 50xx) |
| Stack | 8 MB reserved: `parallelPopulationFitnessCalculation` keeps ~0.8–0.9 MB of arrays on the stack for KC20/KC30, close to the 1 MB default of Windows |
| Paths | Relative to the repository; outputs in `build\` (ignored by git) |

**Command line** (from an *x64 Native Tools* console, since `nvcc` needs `cl.exe`):

```
nvcc -O3 -arch=sm_75 kernel.cu -o cuda_mqap.exe
```

---

## Configuration

### Selecting an instance

`kernel.cu` has a block of `#include "settings_KC*_*fl_*.cu"` lines with **exactly one** uncommented.
To run another instance, comment the active line and uncomment the one you want, then rebuild. Each
settings file defines:

- `FACILITIES_LOCATIONS` and `OBJECTIVES` (2 for `2fl`, 3 for `3fl`);
- `POPULATION_SIZE`, which **must be a power of two** (bitonic sort), and `ITERATIONS`;
- `EXCHANGE_MUTATION_PROBABILITY` and `TRANSPOSITION_MUTATION_PROBABILITY`;
- the distance and flow matrices, copied from the matching `.dat` file (nothing is read at runtime).

| Settings file | `POPULATION_SIZE` | `ITERATIONS` |
|---|---|---|
| KC10-2fl-1rl, 3rl, 4rl, 5rl | 64 | 70 |
| KC10-2fl-1uni, 2rl | 16 | 70 |
| KC10-2fl-2uni | 4 | 70 |
| KC10-2fl-3uni | 128 | 25 |
| KC20-2fl-1rl, 1uni, 2uni, 3uni | 64 | 300 |
| KC30-3fl-1rl, 1uni, 2uni | 32 | 70 |

The header comments of some settings files name the wrong `.dat` file, and
`settings_KC20_2fl_1rl.cu` contains the matrices of another instance (see
[Known issues](#known-issues-of-the-original-code)).

### Debug output

`general_dev_settings.cu` holds `TIMES`, `DEV_MODE` and the `PRINT_*` flags. Most prints require
`DEV_MODE true` **and** the specific flag. `PRINT_FIRST_POPULATION_WITH_FITNESS` (on by default) prints
the initial population with its fitness.

---

## Output

The final population of every run (`TIMES` runs per execution) is **appended** to
`result_KCX_Yfl_Z_nsga2_greedy_2opt.txt` in the working directory, as a Python/JavaScript-style
dictionary literal:

```
{
'0361948752': [5925064, 2282788],
'5134062879': [1665490, 5884156],
...
},
```

The key is the permutation (genes written without separators) and the value, its objective values. With
more than 10 facilities the key is ambiguous, because some genes have two digits; the console output
separates the genes with spaces. The execution time is printed at the end (`Time Spent`).

The population converges, so it normally holds the same solution several times — in a run of
KC10-2fl-1rl, 64 individuals but 38 distinct solutions. **Each distinct solution is reported once**, in
the file and on the console, keeping its first position in the population; the algorithm and the
population itself are untouched.

---

## Result analysis

- **`comparative_results_kcX_datasets.xlsx`**: one tab per instance with the obtained fronts (NSGA-II,
  NSGA-II + Greedy 2-opt, initial population and, for KC10, the Pareto optimal front) and their chart,
  plus the *Distance Metric* tab.
- **`mQAPMetrics/distance_metric_*.js`**: Node.js scripts with the obtained fronts pasted in. For each
  run they compute the mean Euclidean distance from every obtained solution to the closest point of the
  `.PO` front, then the mean and standard deviation over the runs, for NSGA-II and for NSGA-II + Greedy
  2-opt. Run them with `node distance_metric_<instance>.js`.
- **`mQAPMetrics/3D_plot-*.js`**: 3D plots of the 3-objective fronts with LightningChart JS
  (`npm install @arction/lcjs @arction/xydata`).

**The workbook in the rewritten branches.** The copy in this branch holds the four series of 2019 per tab
(NSGA-II, NSGA-II + Greedy 2-opt, initial population and, for KC10, the published optimal front). The
branches that rewrite the program carry their own copy, with their results added next to the originals:

- [`develop_with_claude_opus_5`](https://github.com/apupiales/cuda_mqap/tree/develop_with_claude_opus_5), and the two population branches after
  it: a **green** series per tab, run with the population and the iterations of each experiment.
- [`develop_large_population_multiblock`](https://github.com/apupiales/cuda_mqap/tree/develop_large_population_multiblock): besides the green
  one, a **red** series with the same iterations and the maximum population of that branch, P = 65536. It
  reaches more of the published optimal front on every KC10 instance — 114 of the 130 optimal points of
  KC10-2fl-3uni against 66 of the green series, 39 of 49 against 26 on KC10-2fl-5rl, 46 of 58 against 38
  on KC10-2fl-1rl. Its *Distance Metric* tab also carries the gamma distance of that population over 100
  runs per KC10 instance (26,891.28 → 3,409.13 on KC10-2fl-5rl; 0.00 on 1uni, 2rl and 2uni, where every
  solution found lands on the optimal front), and its README documents the runs and the distances.

---

## Measured parallelization (GPU and CPU use)

One of the goals of the project is to get as much as possible out of the GPU. This section measures how
much of it each version actually uses, with the same workload: the original implementation of this branch
and the three branches that rewrite it (see [Repository branches](#repository-branches)). Every number
comes from `benchmarks/run_comparison.ps1`, so the measurement can be repeated.

### Methodology

- **Same workload for the four versions**: the instance, the population and the number of generations that
  `kernel.cu` has compiled in — KC10-2fl-1rl, P = 64, 70 generations, one run. The script reads them from
  the active `settings_*.cu`, so changing the instance there changes the whole comparison.
- **Two runs per version**: a clean one, which gives the wall time and the CPU time of the process
  (`System.Diagnostics.Process`), and one under Nsight Systems (`nsys profile --trace=cuda`), which gives
  the CUDA trace that everything else is derived from.
- **Active window**: from the start of the first kernel to the end of the last one. The percentages are
  computed over this window, so the fixed cost of starting the process and creating the CUDA context
  (about 100 ms, the same for every version) does not distort the comparison.
- **GPU computing**: share of the window in which a kernel is running. This is the parallel fraction of
  the algorithm in the sense of Amdahl's law; the rest of the window is the GPU waiting for the host.
- **Occupancy**: time-weighted average of the thread slots in use, `Σ min(threads, slots) × duration`
  over `window × slots`, with `slots = SMs × threads per SM` (30 × 1024 = 30,720 on the RTX 2060 used).
  It is not the per-SM achieved occupancy that Nsight Compute reports; it answers a simpler question, how
  full the GPU was on average.
- **Useful threads**: the share of the launched threads that pass the guard of their kernel. It matters
  for the original, which launches blocks of `dim3(32, 32)` = 1024 threads to multiply matrices of
  n × n = 10 × 10, so only 100 of every 1024 threads do any work.
- Measured on an RTX 2060 (6 GB, 30 SMs, sm_75), CUDA 13.4, Windows 11; the four versions compiled with
  `-O3 -arch=sm_75 -std=c++17`.

### Same workload: KC10-2fl-1rl, P = 64, 70 generations

| Metric | Original (this branch) | Optimized | p512 | multiblock |
|---|---|---|---|---|
| Wall time (no profiler) | 3.706 s | 0.171 s | 0.138 s | 0.143 s |
| CPU used by the process | 3.688 s (100 % of one core) | 0.156 s | 0.109 s | 0.109 s |
| Active window of the algorithm | 4,807 ms | 9.8 ms | 9.9 ms | 10.9 ms |
| **GPU computing (kernels / window)** | **6.5 %** | **57.8 %** | **64.3 %** | **60.0 %** |
| Host ↔ device transfers | 4.71 % | 0.08 % | 0.08 % | 0.07 % |
| GPU idle, waiting for the host | 88.8 % | 42.2 % | 35.6 % | 39.9 % |
| Kernel launches | 87,417 (1,249/gen.) | 214 (3/gen.) | 214 | 214 |
| `cudaMemcpy` | 80,539 (1,151/gen.) | 6 | 6 | 6 |
| `cudaDeviceSynchronize` | 68,334 (976/gen.) | 2 | 2 | 2 |
| Average occupancy of the thread slots | 4.6 % (2.0 % useful) | 2.4 % | 2.8 % | 2.7 % |
| Useful threads / launched threads | 10.0 % | 100 % | 100 % | 100 % |
| Work executed (thread·second) | 28,283 | 7.4 | 8.6 | 9.2 |

### How far each version fills the GPU

| Version and population | GPU computing | Average GPU occupancy | Window |
|---|---|---|---|
| Original, P = 64 | 6.5 % | 4.6 % (2.0 % useful) | 4,807 ms |
| Optimized, P = 64 | 57.8 % | 2.4 % | 9.8 ms |
| Optimized, P = 256 (its cap) | 85.2 % | 6.1 % | 28.4 ms |
| p512, P = 512 (its cap) | 86.6 % | 17.5 % | 32.7 ms |
| multiblock, P = 4096 | 93.1 % | 37.7 % | 162 ms |
| multiblock, P = 65536 (its cap) | 99.9 % | 94.2 % | 6,779 ms |

### Reading of the results

**"How much of the algorithm is parallelized" is not what separates the versions.** The original already
has nearly every phase in kernels: fitness, dominance matrix, rank, crowding, binary tournament, mutations
and the acceptance step of the 2-opt are all `__global__` functions. What stays on the host is the
*orchestration*: the loops of the greedy 2-opt and the peeling of the Pareto fronts run on the CPU,
launching one kernel per step. So the useful axes are the other two.

**Effective parallelization (Amdahl's law, measured): 6.5 % → 99.9 %.** In the original the GPU is idle
about 88 % of the time, because there are 976 synchronizations and 1,151 copies per generation: every step
of the 2-opt recomputes the fitness of the whole population with three dense matrix products, copies the
result back to the host and synchronizes. The rewrite leaves 3 launches and 0 synchronizations per
generation, with the whole NSGA-II inside the GPU.

**Use of the GPU: 2.0 % → 94.2 %.** The 4.6 % of the original is apparent: only 10 % of the threads it
launches pass the guard, so the useful occupancy is 2.0 %, and it executes 28,283 thread·second against
7.4 of the rewritten version — about 3,800 times more work for the same result.

**With P = 64 the limit is no longer the GPU but the size of the problem.** Even in the rewritten versions
the GPU stays nearly empty (2.4 %): 64 individuals cannot fill 30,720 thread slots. That is what the
large-population branches are for: `p512` reaches 17.5 % with P = 512, `multiblock` 37.7 % with P = 4096
and 94.2 % with P = 65536, where the program finally becomes compute bound (99.9 % of the window with the
GPU computing, and the CPU down to 35 % of the wall time, spent waiting inside `cudaDeviceSynchronize`).

### Reproducing the measurement

Requires Visual Studio, the CUDA Toolkit (it ships `nsys`) and Python. From a clone of the repository:

```
git worktree add ../cuda_mqap_opt        develop_with_claude_opus_5
git worktree add ../cuda_mqap_p512       develop_p512_single_block
git worktree add ../cuda_mqap_multiblock develop_large_population_multiblock

powershell -ExecutionPolicy Bypass -File benchmarksun_comparison.ps1
```

It builds the four versions, runs the eight cases twice each and prints the tables above. Use
`-Optimized`, `-P512` and `-Multiblock` if the other source trees are somewhere else, `-Arch` for a GPU
that is not Turing, and `-SkipBuild` to reuse the binaries. The binaries, the `.nsys-rep` traces and the
CSV reports are left in `benchmarks/results/` (ignored by git). `benchmarks/analyze.py` can be run again
on its own over an existing run directory; `SMS` and `THREADS_PER_SM` at the top of that file describe the
GPU and have to match the one used.

### Caveats

- The three rewritten branches share the same kernels up to P = 256, which is why their numbers at P = 64
  are the same; they only differ above that size.
- The percentages come from the profiled run. Nsight Systems adds overhead per launch, which penalizes the
  87,417 launches of the original: measured over its clean wall time, its GPU-computing share is 8.4 %
  instead of 6.5 %.
- One run per case. Repeating the whole measurement moves the short cases by a few points (the two runs
  taken here differ by up to 4 points at P = 64), so the small numbers are orders of magnitude, not exact
  values.
- The occupancy above is a proxy computed from the launch geometry, not the achieved occupancy per SM that
  Nsight Compute measures.

---

## Known issues of the original code

A review of this version (compute-sanitizer, Nsight Systems and consistency checks) found these problems.
They are documented here and **not changed in this branch**. The minimal fixes are in commit `3f3a187`,
and the complete rework is in the branch `develop_with_claude_opus_5`.

| # | Issue | Effect |
|---|---|---|
| B1 | `cudaMalloc(&d_state, sizeof(curandState))` reserves 1 state, but `curand_setup` initializes up to 8,192 | Out-of-bounds writes in GPU memory. With CUDA 13.4, **running without the Greedy 2-opt produces impossible fitness values** (negative or in the billions); `compute-sanitizer --tool memcheck` reports it |
| B2 | `binaryTournament` is launched with `NSGA2_POPULATION_SIZE` blocks over arrays of `POPULATION_SIZE` elements | Out-of-bounds reads and writes |
| B3 | `settings_KC20_2fl_1rl.cu` contains the matrices of **KC20-2fl-2rl** | Results labeled KC20-2fl-1rl belong to KC20-2fl-2rl |
| B4 | The tournament RNG is seeded with `time(NULL)` every generation | Consecutive generations within the same second repeat the adversaries |
| B5 | `shufflePopulationGenes` shares curand states between blocks | Correlated random numbers and a biased shuffle |
| B6 | Crowding: `(unsigned int)HUGE_VALF`, missed front end, possible division by zero | Undefined behavior in corner cases |
| B7 | Per-generation `cudaMalloc` without `cudaFree` | Memory leaks (783 allocations in a KC10 run) |
| B8 | The greedy loop visits the pairs (i, j) and (j, i) | Twice the local search work |
| B9 | ~0.8–0.9 MB of arrays on the host stack | Close to the 1 MB Windows stack (8 MB reserved in the project) |
| B10 | `DEV_MODE \|\| PRINT_*` instead of `&&`, wrong `sizeof` | Unexpected debug output |
| B11 | The last iteration keeps only the first front plus stale rows | The final population mixes non-dominated and stale solutions |
| B12 | Solutions outside the current front can win the crowding sort | Rare wrong selections |

Performance: in a KC10-2fl-1rl run (70 iterations, 3.6 s on an RTX 2060) the GPU is busy only ~9 % of
the time. Nearly all of it goes into kernel launches, debug copies and per-kernel synchronizations
(87,510 launches, 80,558 `cudaMemcpy`).

---

## Credits and citations

### Author

**Andrés Pupiales Arévalo** — <apupiales@gmail.com> — <https://github.com/apupiales>. Project started in May 2019.

### Test suites (third-party data)

The files in `mQAPData/` are **not part of this program's source code** and are not covered by its
license. They are redistributed unmodified, for academic and research use, with credit to their authors
(details in [`mQAPData/README.txt`](mQAPData/README.txt)):

- **Instances (`.dat`)**: mQAP test suite of **Joshua D. Knowles and David W. Corne**, generated with
  their instance generators `makeQAPuni` and `makeQAPrl` ((C) J. Knowles, 2002). The original page is no
  longer online; [archived copy](https://web.archive.org/web/2019/http://www.cs.bham.ac.uk/~jdk/mQAP/).
- **Pareto optimal fronts (`.PO`)**: enumeration of the Pareto optima of the ten-facility instances by
  **Gary Lamont**, published on the same page.
- **Copy used**: [fredizzimo/keyboardlayout](https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData)
  (Fred Sundvik, 2015). The MIT license of that repository does not cover this data, whose authors are
  the ones listed above.
- **Terms**: no explicit license was published for the instances. The original page offers the generators
  as free software *"for academic or educational use"* and asks to contact the author for commercial use;
  the same criterion applies to the instances.

Please cite the test suite when you use it:

> J. D. Knowles and D. W. Corne. *Instance Generators and Test Suites for the Multiobjective Quadratic
> Assignment Problem*. In Evolutionary Multi-Criterion Optimization (EMO 2003), Lecture Notes in Computer
> Science, vol. 2632, pp. 295–310. Springer, 2003.

```bibtex
@InProceedings{Knowles2003mQAP,
  author    = {Joshua D. Knowles and David W. Corne},
  title     = {Instance Generators and Test Suites for the Multiobjective Quadratic Assignment Problem},
  booktitle = {Evolutionary Multi-Criterion Optimization (EMO 2003)},
  series    = {Lecture Notes in Computer Science},
  volume    = {2632},
  pages     = {295--310},
  publisher = {Springer},
  year      = {2003}
}
```

Related work by the same authors (landscape analysis of the mQAP):

> J. D. Knowles and D. W. Corne. *Towards Landscape Analyses to Inform the Design of a Hybrid Local
> Search for the Multiobjective Quadratic Assignment Problem*. In A. Abraham, J. Ruiz-del-Solar and
> M. Köppen (eds.), Soft Computing Systems: Design, Management and Applications, pp. 271–279. IOS Press,
> Amsterdam, 2002.

### Algorithmic references

- The adaptation of the Greedy 2-opt criterion to several objectives follows the work referenced in
  `kernel.cu`: <https://arxiv.org/abs/1109.1276>.
- The metric scripts use [LightningChart JS](https://lightningchart.com/js-charts/) (`@arction/lcjs`) for the 3D plots.

---

## License

Copyright (C) 2019-2026 Andrés Pupiales Arévalo.

This program is distributed under the **GNU General Public License v3** (see [`LICENSE`](LICENSE)).
The header of `kernel.cu` states *"version 2 of the License, or (at your option) any later version"*,
which allows distribution under version 3. The program is distributed in the hope that it will be
useful, but **without any warranty**.

NVIDIA's CUDA Toolkit (the `nvcc` compiler, the `cudart` runtime and the cuRAND library) is not part of
this repository: it is proprietary NVIDIA software, distributed under its own license (CUDA EULA). The
data in `mQAPData/` is third-party data (see [Test suites](#test-suites-third-party-data)).
