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
11. [Known issues of the original code](#known-issues-of-the-original-code)
12. [Credits and citations](#credits-and-citations)
13. [License](#license)

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
