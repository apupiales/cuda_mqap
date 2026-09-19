# cuda_mqap — NSGA-II + Adapted Greedy 2-opt in CUDA for the mQAP

**English** | [Español](LEEME.md)

GPU-parallel implementation (CUDA C++) of the multiobjective evolutionary algorithm **NSGA-II**,
combined with an **adapted Greedy 2-opt** local search, to solve instances of the
**multiobjective Quadratic Assignment Problem** (mQAP).

The whole algorithm runs on the GPU: fitness evaluation, non-dominated sorting, crowding distance,
selection, mutation and local search. Each generation takes **3 kernel launches with no host
synchronization**, and **several independent runs can execute concurrently** in a single call to
the program.

---

## Contents

1. [Features](#features)
2. [The problem: mQAP](#the-problem-mqap)
3. [The algorithm](#the-algorithm)
4. [Project architecture](#project-architecture)
5. [GPU design](#gpu-design)
6. [Requirements](#requirements)
7. [Build](#build)
8. [Usage](#usage)
9. [Experiments and metrics](#experiments-and-metrics)
10. [Tests and validation](#tests-and-validation)
11. [Performance](#performance)
12. [Improvements over the original version](#improvements-over-the-original-version)
13. [Population size limits and GPU resources](#population-size-limits-and-gpu-resources)
14. [Limitations and future work](#limitations-and-future-work)
15. [Troubleshooting](#troubleshooting)
16. [Credits and license](#credits-and-license)

---

## Features

**Algorithm**
- Complete NSGA-II: fast non-dominated sorting, crowding distance and elitist (μ + λ) selection.
- Binary tournament selection, exchange mutation and transposition mutation (reversal of a segment).
- Greedy 2-opt adapted to several objectives: in each generation the improvement criterion is chosen at
  random, either the sum of all objectives or a single objective.
- Instances with 2 or 3 objectives (flow matrices) and up to 64 facilities.

**GPU performance**
- **O(n²)** fitness per chromosome (one *warp* per chromosome, with the matrices in *shared memory*),
  instead of three O(n³) dense matrix products.
- Complete NSGA-II **inside a single block per run**: fast non-dominated sorting with dominator counters
  and front lists (no dominance matrix, so P up to 512 on any GPU) and bitonic sorts in *shared memory*.
- Greedy 2-opt with **O(n) incremental (delta) evaluation** of each swap; the local search of the whole
  offspring is a single kernel.
- Persistent Philox random states, initialized only once.
- **Independent runs in parallel** (`--runs R`) to use the whole GPU in experiment campaigns.

**Engineering**
- Strict host/device separation: `main.cpp` contains no CUDA code, and kernels are exposed through launcher functions.
- Instances read from the `.dat` files at runtime; parameters are passed on the command line.
- `CUDA_CHECK`/`CUDA_CHECK_KERNEL` error checking and RAII memory management (`DeviceBuffer<T>`).
- **Plug and play in Visual Studio 2026**: clone, open `cuda_mqap.slnx`, press F5. The CUDA version,
  C++ toolset and GPU architectures adapt to the machine. Also `CMakeLists.txt` with `ctest`.
- Test suite that compares every kernel with an independent CPU implementation, plus the `--verify`
  option, which validates the results of every run.
- Reproducible results through a seed (`--seed`).

---

## The problem: mQAP

`n` facilities must be assigned to `n` locations. A solution is a permutation `p`, where `p[i]` is the
location of facility `i`. Each objective `k` has its own flow matrix `Fk`, and all of them share the
distance matrix `D`. The `m` costs are minimized simultaneously:

```
cost_k(p) = Σ_i Σ_j  Fk[i][j] · D[p(i)][p(j)]          k = 1..m   (m = 2 or 3)
```

This expression is equivalent to the matrix formulation `Trace(Fk · X · Dᵀ · Xᵀ)` used by the original
version, where `X` is the permutation matrix; the tests verify that equivalence. Because the objectives
conflict, the result is not a single solution but an approximation of the **Pareto front**: the set of
non-dominated solutions.

### Instances (`mQAPData/`)

The instances are Knowles and Corne's mQAP test suite
([archived page](https://web.archive.org/web/2019/http://www.cs.bham.ac.uk/~jdk/mQAP/)), taken from the copy at
<https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData>. They are third-party data, not
covered by this project's license: see [Third-party data](#third-party-data).

| File | Content |
|---|---|
| `KC<n>-<m>fl-<type>.dat` | Header (`facilities = 10 objectives = 2 …` or `facilities: 10 objectives: 2 …`), the `n×n` distance matrix and `m` `n×n` flow matrices |
| `KC10-2fl-*.PO` | Published Pareto optimal front: each line holds a 1-based permutation and its `m` costs |

`rl` instances have *real-like* distances and flows, and `uni` instances have uniform values.
There are 23 instances with n = 10, 20 and 30, and the optimal fronts of the 8 KC10 instances.

---

## The algorithm

```mermaid
flowchart TD
    A[Random initial population<br/>2P permutations · Fisher-Yates] --> B[Fitness of the 2P solutions]
    B --> C{{"NSGA-II survival (Rt = Pt ∪ Qt)<br/>fronts · crowding · best P"}}
    C -->|last iteration?| Z[Final non-dominated front]
    C --> D["Reproduction<br/>Pt+1 = survivors<br/>Qt+1 = binary tournament winners + mutations"]
    D --> E["Adapted Greedy 2-opt on Qt+1<br/>(leaves the fitness updated)"]
    E --> C
```

Each generation does the following:

1. **NSGA-II survival** over the `2P` solutions of `Rt = Pt ∪ Qt`:
   - *Non-dominated sorting*: rank 1 for the first Pareto front, rank 2 for the next one, and so on.
   - *Crowding distance* of each front. For each objective the members of the front are sorted; the
     extremes get ∞ and the interior points add `(f[next] − f[previous]) / (max − min)`, with the maximum
     and minimum taken over the whole population.
   - The best `P` solutions by (rank ascending, crowding descending) are selected.
2. **Reproduction**. The `P` survivors form `Pt+1`. For each of them a **binary tournament** is held
   against another survivor chosen at random: the lower rank wins and, on a tie, the larger crowding.
   The winner is copied and mutated:
   - **Exchange mutation**: two random genes are swapped; it is applied twice.
   - **Transposition mutation**: the segment between two random positions is reversed.
3. **Adapted Greedy 2-opt** on each offspring. All pairs of positions `(r < s)` are visited in order and
   a swap is kept if it does not worsen the criterion of the generation, chosen at random for each run and
   generation: the sum of all objectives or a single objective `k`. The idea of adapting the criterion
   comes from <https://arxiv.org/ftp/arxiv/papers/1109/1109.1276.pdf>.

Parameters:

| Parameter | Where | Default |
|---|---|---|
| Population size `P` | `--population` | 64 (power of 2 between 16 and 512) |
| Generations | `--iterations` | 70 |
| Independent runs | `--runs` | 1 |
| Seed | `--seed` | random (printed) |
| Exchange mutations per child | `include/config.h` (`kExchangeMutations`) | 2 |
| Exchange / transposition probability | `include/config.h` | 1.0 / 1.0 |

---

## Project architecture

```
cuda_mqap/
├── include/
│   ├── config.h            Limits (n, P, objectives) and operator parameters
│   ├── cuda_check.cuh      CUDA_CHECK / CUDA_CHECK_KERNEL
│   ├── device_buffer.cuh   DeviceBuffer<T>: GPU memory with RAII
│   ├── device_common.cuh   Shared __device__ functions (per-warp cost, 2-opt delta, bitonic sort)
│   ├── instance.h          Instance struct, loadInstance(), reference CPU cost()
│   ├── kernels.cuh         Declaration of the kernel launchers and of the memory layout
│   └── solver.h            SolverOptions, Solution, RunResult, solve()
├── src/
│   ├── main.cpp            Command line, result file and --verify (host only)
│   ├── instance.cpp        .dat parser and validation (including fitness overflow)
│   ├── solver.cu           Host orchestration: allocations, generation loop and result collection
│   ├── fitness.cu          Fitness kernel
│   ├── nsga2.cu            NSGA-II survival kernel
│   ├── operators.cu        RNG, initial population, tournament and mutations
│   └── local_search.cu     Greedy 2-opt kernel
├── tests/test_kernels.cu   Tests of every kernel against CPU references
├── scripts/run_experiments.ps1   Experiment campaign with the parameters of each instance
├── mQAPData/               Instances (.dat) and optimal fronts (.PO)
├── mQAPMetrics/            Node.js metric and 3D plot scripts
├── comparative_results_kcX_datasets.xlsx   Comparative results
├── cuda_mqap.slnx, cuda_mqap.vcxproj, test_kernels.vcxproj   Visual Studio solution and projects
├── cuda_mqap.props, cuda_toolkit.props   Shared settings and CUDA version detection
├── .vsconfig, .gitattributes             Visual Studio components and line endings
└── CMakeLists.txt
```

**Host/device separation.** The program is organized in three layers:

| Layer | Files | Responsibility |
|---|---|---|
| Application (host) | `main.cpp`, `instance.cpp` | Arguments, instance loading, result output and CPU verification |
| Orchestration (host) | `solver.cu` | Allocation of all memory once, launch sequence and final copy of the results |
| Kernels (device) | `fitness.cu`, `nsga2.cu`, `operators.cu`, `local_search.cu` | `template<int OBJ>` kernels (instantiated for 2 and 3 objectives) and their launchers |

The kernels live in the `mqap::detail` namespace. Outside their `.cu` file only the launchers declared in
`kernels.cuh` are visible (`launchFitness`, `launchSurvival`, `launchReproduce`, `launchGreedy2Opt`…),
and each one checks its launch with `CUDA_CHECK_KERNEL()`.

---

## GPU design

### Memory layout

For `R` runs, population `P`, `n` facilities and `OBJ` objectives:

| Buffer | Type and shape | Description |
|---|---|---|
| `genes` (×2, double buffer) | `short [R][2P][n]` | Rows `[0, P)`: survivors; rows `[P, 2P)`: offspring |
| `fitness` (×2) | `unsigned int [R][2P][OBJ]` | Cost of each objective |
| `survivorIndex / Rank / Crowding` | `[R][P]` | Survival result, ordered by (rank, −crowding) |
| `rng` | `curandStatePhilox4_32_10_t [R][2P]` | Persistent random states |
| `flow`, `dist` | `int [OBJ][n][n]`, `int [n][n]` | Instance matrices |

All memory is allocated **once** with `DeviceBuffer<T>` and released automatically. Between
generations only the pointers of the double buffer are swapped.

### Kernels

| Kernel | Grid × block | Parallelism | Techniques |
|---|---|---|---|
| `fitnessKernel<OBJ>` | `(⌈2P/4⌉, R)` × 128 | 1 warp per chromosome | `F` and `D` in *shared memory* (coalesced load); consecutive `F` reads per lane (no bank conflicts); reduction with `__shfl_down_sync` |
| `survivalKernel<OBJ>` | `R` × `2P` | 1 block per run, 1 thread per individual | Fast non-dominated sorting: dominator counts and a shared list of each front (O(N²) work, O(N) memory), bitonic sort of 64-bit keys `(rank, fitness)` and `(rank, −crowding)` in *shared memory* |
| `reproduceKernel<OBJ>` | `(⌈P/128⌉, R)` × 128 | 1 thread per offspring | Philox state in registers; tournament, mutations and copy in a single pass |
| `greedy2OptKernel<OBJ>` | `(⌈P/4⌉, R)` × 128 | 1 warp per offspring | Matrices in *shared memory*; O(n) delta split across the 32 lanes; warp-uniform criterion (no divergence) |
| `initPopulationKernel` | `(⌈2P/128⌉, R)` × 128 | 1 thread per chromosome | Unbiased Fisher-Yates |
| `rngInitKernel` | `⌈R·2P/128⌉` × 128 | 1 thread per state | One independent Philox subsequence per thread |

*Shared memory* per block:
- **Fitness and 2-opt:** `(OBJ + 1)·n²·4 + 4·n·2` bytes, e.g. 14.6 KB for n = 30 and 3 objectives.
  When more than 48 KB are needed, the device's maximum *opt-in* is requested automatically
  (`cudaFuncAttributeMaxDynamicSharedMemorySize`), which allows n = 60 with 3 objectives on Turing.
- **Survival:** grows linearly with P, ~30 KB with P = 512.

### Incremental 2-opt evaluation

Swapping positions `r` and `s` of `p` only changes the cost terms in which `r` or `s` appear:

```
Δ(r,s) = (F_rr − F_ss)(D_{ps ps} − D_{pr pr}) + (F_rs − F_sr)(D_{ps pr} − D_{pr ps})
       + Σ_{k≠r,s} [ (F_kr − F_ks)(D_{pk ps} − D_{pk pr}) + (F_rk − F_sk)(D_{ps pk} − D_{pr pk}) ]
```

Each lane of the warp computes part of the sum and the result is reduced with `__shfl_down_sync`.
Each of the `n(n−1)/2` evaluations therefore costs O(n) instead of recomputing the full fitness.
The accumulators are 64-bit.

### Synchronization

- All kernels are launched on the *default stream*, which already guarantees their order, so
  `cudaDeviceSynchronize` is not used during the run.
- The host only waits at the end (`cudaEventSynchronize`), to measure the time and copy the results.
- Inside the kernels, `__syncthreads()` only separates phases that share *shared memory*, and
  `__syncwarp()` makes the swap applied by the 2-opt visible to the whole warp.
- In Debug builds, `MQAP_SYNC_CHECK` makes `CUDA_CHECK_KERNEL()` synchronize after every kernel, so an
  execution error is reported at the launch that caused it.

---

## Requirements

- NVIDIA GPU with *compute capability* ≥ 7.5 (GeForce RTX 20xx or newer) and an up-to-date driver.
- **Visual Studio 2026** with the *Desktop development with C++* workload. When the solution is opened,
  Visual Studio reads `.vsconfig` and offers to install any missing component.
- **CUDA Toolkit 12.x or 13.x** (≥ 11.8), installed **after** Visual Studio so that its *Visual Studio
  Integration* is added to it. The project detects the installed version automatically; it was developed
  with CUDA 13.4.
- Alternative without the IDE: CMake ≥ 3.24 + Ninja (both included with Visual Studio).

---

## Build

### Open in Visual Studio 2026 (plug and play)

```
git clone https://github.com/apupiales/cuda_mqap.git
cd cuda_mqap
git checkout develop_with_claude_opus_5
start cuda_mqap.slnx
```

1. Visual Studio opens the solution with its two projects: `cuda_mqap` (the program, startup project) and
   `test_kernels` (the tests).
2. Select `Release | x64` and press **F5** (or Ctrl+F5). The program runs on
   `mQAPData\KC10-2fl-1rl.dat --verify` with the repository root as working directory. Edit the arguments
   in *Project → Properties → Debugging*.
3. To run the tests: right-click `test_kernels` → *Set as Startup Project* → Ctrl+F5.
4. The executables are written to `build\x64\<Configuration>\`.

Nothing depends on the machine where the project was created:

| What | How it adapts |
|---|---|
| CUDA version | `cuda_toolkit.props` takes it from `CUDA_PATH` (e.g. `...\CUDA\v12.6` → `CUDA 12.6.props`). To use another installed version: `set CudaVersion=12.6` before opening VS, or `msbuild /p:CudaVersion=12.6` |
| Missing CUDA integration | The build stops with a message that explains how to fix it (instead of "unknown item type CudaCompile") |
| C++ toolset | `$(DefaultPlatformToolset)` of the Visual Studio that opens it (v145 in VS 2026); Windows SDK `10.0` (the latest installed) |
| GPU | Native code for `sm_75`, `sm_80`, `sm_86` and `sm_89`, plus PTX that the driver compiles for newer GPUs (RTX 50xx) |
| Paths | All relative to the repository (`$(MSBuildThisFileDirectory)`); outputs in `build\` (ignored by git) |
| Line endings | `.gitattributes` keeps CRLF for the Visual Studio files |

The common configuration is in `cuda_mqap.props`: C++17, `/W4`, `-lineinfo` in Release, and `-G` plus
`MQAP_SYNC_CHECK` in Debug.

### CMake

```
cmake -S . -B build/cmake -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake
ctest --test-dir build/cmake --output-on-failure
```

By default it builds `sm_75`, `sm_80`, `sm_86` and `sm_89` plus PTX; for your GPU only use
`-DCMAKE_CUDA_ARCHITECTURES=native`. In Visual Studio you can also use *File → Open → Folder*.

### nvcc directly

From an *x64 Native Tools* console:

```
nvcc -O3 -arch=sm_75 -std=c++17 -Iinclude src\main.cpp src\instance.cpp src\solver.cu src\fitness.cu ^
     src\nsga2.cu src\operators.cu src\local_search.cu -o cuda_mqap.exe
```

---

## Usage

```
cuda_mqap <instance.dat> [options]
  --population P   population size, power of two in [16, 512] (default 64)
  --iterations N   generations (default 70)
  --runs R         independent runs executed concurrently (default 1)
  --seed S         random seed (default: random, printed in the output)
  --output FILE    result file, appended (default result_<instance>_nsga2_greedy_2opt.txt)
  --verify         check the final populations on the CPU
  --quiet          do not print the final solutions
```

Examples:

```
:: One run with verification
build\x64\Release\cuda_mqap.exe mQAPData\KC10-2fl-1rl.dat --verify

:: 30 independent runs in parallel, reproducible
build\x64\Release\cuda_mqap.exe mQAPData\KC20-2fl-1rl.dat --iterations 300 --runs 30 --seed 2026 --quiet

:: 3-objective instance
build\x64\Release\cuda_mqap.exe mQAPData\KC30-3fl-1rl.dat --population 32 --runs 10
```

Console output (abridged):

```
Instance KC10-2fl-1rl: n = 10, objectives = 2 | population = 64, iterations = 70, runs = 1, seed = 42

FINAL SOLUTION (run 0, 64 non-dominated)
0 3 6 1 9 4 8 7 5 2 5925064 2282788
5 1 3 4 0 6 2 8 7 9 1665490 5884156
5 1 6 3 0 2 8 9 7 4 1869616 4670952
...
Verification: OK

Results appended to result_KC10-2fl-1rl_nsga2_greedy_2opt.txt
Time Spent: 0.148970 s (GPU 13.494 ms)
```

### Result file

It keeps the original format, so the `mQAPMetrics` scripts still work. Each run appends a block with
the non-dominated (rank 1) solutions of the final population: the key is the permutation and the value,
its costs.

```
{
'0361948752': [5925064, 2282788],
'5134062879': [1665490, 5884156],
'5163028974': [1869616, 4670952],
},
```

At the end of a run the population may contain repeated solutions; in the file, duplicate keys collapse
when it is read as a dictionary, just as in the original version.

### Verification (`--verify`)

Independently recomputes on the CPU every solution of the final population of every run, checking:
- that the permutation is valid;
- that the fitness matches the CPU `cost()` exactly;
- that rank 1 corresponds exactly to the non-dominated solutions (and that every solution with rank > 1
  is dominated by some survivor).

If any check fails, the exit code is 1.

---

## Experiments and metrics

`scripts/run_experiments.ps1` runs the campaign of `comparative_results_kcX_datasets.xlsx` with the
population and iterations each instance used in the original version:

```
.\scripts\run_experiments.ps1 -Runs 30                              # all instances
.\scripts\run_experiments.ps1 -Runs 10 -Seed 2026 -Instances KC10-2fl-1rl,KC30-3fl-1rl
```

| Instances | P | Generations |
|---|---|---|
| KC10-2fl-1rl, 3rl, 4rl, 5rl | 64 | 70 |
| KC10-2fl-1uni, 2rl, 2uni ¹ | 16 | 70 |
| KC10-2fl-3uni | 128 | 25 |
| KC20-2fl-1rl, 1uni, 2uni, 3uni | 64 | 300 |
| KC30-3fl-1rl, 1uni, 2uni | 32 | 70 |

¹ KC10-2fl-2uni used P = 4; the minimum is now 16.

The results are saved to `results\result_<instance>_nsga2_greedy_2opt.txt`. On the RTX 2060 the whole
campaign (15 instances × 3 runs) takes about 2 seconds.

**Metrics (`mQAPMetrics/`)** — Node.js scripts that contain the obtained fronts, copied from the result files:
- `distance_metric_*.js`: generational distance, i.e. the mean Euclidean distance from each obtained
  solution to the closest point of the optimal `.PO` front. It reports the mean and standard deviation
  over runs, for NSGA-II and for NSGA-II + Greedy 2-opt.
- `3D_plot-*.js`: 3D plots of the fronts of the 3-objective instances, with LightningChart JS
  (`@arction/lcjs`).

### Results in the Excel workbook

On 2026-09-19 the results of this version were added to `comparative_results_kcX_datasets.xlsx`:

- **Instance tabs (KC10-\*, KC20-\*):** each tab has a new block to the right of the existing ones, with
  20 or 10 genes and 2 objectives per row, and a **green** series in its chart:
  *"CUDA NSGA-II Paralelo + Greedy 2opt, N iteraciones (optimizado con claude)"*. Each tab's own population
  and iterations were used, with `--verify`.
  - KC10 tabs show the first of 100 concurrent runs; KC20 tabs, a single run.
  - A note below each block records the command, seed and time.
  - KC10-2fl-2uni ran with P = 16 (the original series used P = 2).
- **Distance Metric:** columns F–G (mean and standard deviation) and column H of the second table, with the
  gamma distance of this version over 100 runs per KC10 instance. It is computed exactly like
  `mQAPMetrics/distance_metric_*.js` and truncated to 2 decimals, like the existing values.

**Quality versus the original Greedy 2-opt.** The results are mixed:

| Instance | Metric | Original | This version |
|---|---|---|---|
| KC10-2fl-1rl | gamma distance (lower is better), 100 runs | 1,484.66 | **850.56** |
| KC10-2fl-3rl | same | 22,541.34 | **20,531.69** |
| KC10-2fl-4rl | same | 12,399.20 | **7,515.98** |
| KC10-2fl-5rl | same | 32,418.89 | **26,891.28** |
| KC10-2fl-3uni | same | 381.15 | 376.23 |
| KC10-2fl-1uni | same | **79.32** | 192.02 |
| KC10-2fl-2rl | same | **4,451.70** | 10,941.55 |
| KC10-2fl-2uni | same (different P, not comparable) | 1,346.43 | 532.56 |
| KC20-2fl-1uni | hypervolume, 1 run (higher is better) | 3.5249·10¹⁰ | 3.5253·10¹⁰ |
| KC20-2fl-1rl | same | **6.3518·10¹³** | 6.3061·10¹³ (−0.7 %) |
| KC20-2fl-2uni | same | **8.4911·10⁹** | 7.8799·10⁹ (−7.2 %) |
| KC20-2fl-3uni | same | **7.6649·10¹⁰** | 7.4938·10¹⁰ (−2.2 %) |

The KC20 figures come from a single run of each version, so they do not support statistical conclusions.

**Workbook corrections (2026-09-19)**
- **KC20-2fl-3uni:** the NSGA-II, Greedy 2opt and hidden Pareto optimal chart series pointed to the
  KC20-2fl-1uni tab, and cell A1 read "KC20-2fl-1uni Pareto Optimal". They now use the tab's own data.
- **KC20-2fl-1rl:** the original series held results of instance **KC20-2fl-2rl**, because the former
  `settings_KC20_2fl_1rl.cu` contained those matrices (bug B3). They were re-run on the correct instance,
  P = 64 and 300 iterations, using the original code (`kernel.cu` of `ec882da`) with **only** the memory
  fixes B1 and B2:
  - NSGA-II: the `greedy2Opt` call commented out (2.8 s);
  - NSGA-II + Greedy 2opt: 104.5 s;
  - initial population: the one from the Greedy run.

  All 320 rows of the tab have fitness equal to their cost on KC20-2fl-1rl. The notes are in X67 and CS67,
  and the old data remains in the git history.
- **2019 data:** in all 12 tabs, every row of the NSGA-II, Greedy and initial population blocks has exactly
  the cost of its permutation, so the original results are internally consistent.
- **Open issue:** in the second column of Distance Metric, the NSGA-II value for KC10-2fl-2uni (27,629.49)
  does not match the current data in `mQAPMetrics/distance_metric_KC10_2fl_2uni.js` (15,195.66).

> **Warning about the original code.** With CUDA 13.4, `kernel.cu` of `ec882da` run without the Greedy
> 2-opt produces impossible fitness values (negative, or in the billions). The cause is the out-of-bounds
> writes of `curand_setup` (bug B1), which corrupt the NSGA-II buffers. To reproduce the original version,
> use commit `3f3a187` or apply at least fixes B1 and B2, and check it with
> `compute-sanitizer --tool memcheck`.

---

## Tests and validation

`test_kernels` (the `test_kernels` project in Visual Studio, or `ctest`) compares every kernel with an
independent CPU implementation:

| Test | What it checks |
|---|---|
| Instance loading | All 23 `.dat` files load (in both header formats) and `n`/`m` match the file name |
| `.PO` optimal fronts | The **374 published optimal solutions** have exactly their published cost, both on the CPU and on the GPU |
| Fitness | The kernel matches the original version's literal `Trace(F·X·Dᵀ·Xᵀ)` on KC10, KC20 and KC30, with several runs |
| NSGA-II survival | Ranks, crowding and selection match a CPU NSGA-II for P = 16, 64, 256 and 512, with 2 and 3 objectives |
| Greedy 2-opt | The resulting permutation is **identical** to that of a CPU greedy that recomputes the full cost (n = 10, 30 and 60, the latter with more than 48 KB of *shared memory*) |
| Reproduction | Survivors and their fitness are copied correctly and the children are valid permutations |
| Initial population | Every permutation is valid and shuffled |

```
build\x64\Release\test_kernels.exe mQAPData
```

Additional validation performed with `compute-sanitizer` on the program and on the tests:

```
compute-sanitizer --tool memcheck --leak-check full build\x64\Release\cuda_mqap.exe mQAPData\KC30-3fl-1rl.dat --population 32 --iterations 5 --runs 2 --verify
compute-sanitizer --tool racecheck  ...
compute-sanitizer --tool synccheck  ...
compute-sanitizer --tool initcheck  ...
```

Result: 0 errors, 0 leaks and 0 *hazards*.

---

## Performance

GeForce RTX 2060 (sm_75, 30 SMs), Release builds, CUDA 13.4. The "Fixed original" column is the
previous monolithic code (`kernel.cu`, commit `3f3a187`) with its memory errors fixed.

| Case | Fixed original | This version | Speedup (wall time) |
|---|---|---|---|
| KC10-2fl-1rl, P=64, 70 gen., 1 run | 2.2 s | 0.15 s (15 ms GPU) | ~15× |
| KC10-2fl-1rl, P=64, 70 gen., 10 runs | 23.7 s | 0.12 s (21 ms GPU) | ~200× |
| KC20-2fl-1rl, P=64, 300 gen., 1 run | 48.8 s | 0.15 s (55 ms GPU) | ~325× |
| KC30-3fl-1rl, P=32, 70 gen., 1 run | 42.4 s | 0.14 s (40 ms GPU) | ~300× |
| KC30-3fl-1rl, P=32, 70 gen., 30 runs | ~21 min (estimated) | 0.23 s (135 ms GPU) | ~5,500× |

In this version the wall time is dominated by the creation of the CUDA context (~0.1 s), so the GPU
time better reflects the cost of the algorithm.

Nsight Systems profile (KC10-2fl-1rl, 70 generations, 1 run):

| Metric | Original (`ec882da`) | This version |
|---|---|---|
| Total time | 3.64 s | 0.15 s |
| Kernel launches | 87,510 | 214 |
| `cudaMemcpy` | 80,558 | 6 |
| `cudaDeviceSynchronize` | 68,335 | 0 |
| `cudaMalloc` / `cudaFree` | 21,507 / 20,724 (783 leaks) | 11 / 11 |
| Total kernel time | ~340 ms | ~6.4 ms |

**Solution quality.** Measured as the fraction of the optimal `.PO` front points found exactly and as
normalized IGD, with the same parameters in both versions:

| Instance | Fixed original | This version |
|---|---|---|
| KC10-2fl-1rl (P=64, 70 gen.) | 68.4 % · IGD 0.0053 | 68.4 % · IGD 0.0054 |
| KC10-2fl-3uni (P=128, 25 gen.) | 44.2 % · IGD 0.0057 | 45.0 % · IGD 0.0055 |

On these two instances the quality is equivalent. Over the whole campaign of the Excel workbook the results
are mixed: see [Results in the Excel workbook](#results-in-the-excel-workbook).

---

## Improvements over the original version

### Fixed bugs

| # | Bug in the original version | Fix |
|---|---|---|
| B1 | 1 `curandState` was allocated but up to 8,192 were initialized, writing out of bounds in GPU memory (with CUDA 13.4 it corrupts the results of the variant without Greedy) | Philox states sized per thread (`[R][2P]`) and persistent |
| B2 | The binary tournament used 2P blocks over arrays of size P (out-of-bounds accesses) | One thread per offspring, with a bounds guard |
| B3 | `settings_KC20_2fl_1rl.cu` contained the KC20-2fl-2rl matrices | Instances are read directly from the `.dat` files |
| B4 | The tournament seed was `time(NULL)` in every generation, so ~19 consecutive generations repeated adversaries | Persistent RNG with a single 64-bit seed |
| B5 | The initial shuffle used curand states shared between blocks and was biased | Fisher-Yates with one state per chromosome |
| B6 | In the crowding distance, `(unsigned)HUGE_VALF` (undefined behavior), a mis-detected front end and a possible division by zero | Crowding rewritten with a real ∞ and a range check |
| B7 | 11 GPU allocations per generation never released | RAII `DeviceBuffer<T>`; all allocations are made once |
| B8 | The greedy evaluated every pair twice ((i,j) and (j,i)) | Each pair `r < s` is evaluated once |
| B9 | ~0.9 MB of debug arrays on the host stack (1 MB on Windows) | Removed |
| B10 | `DEV_MODE \|\| PRINT_*` instead of `&&`, and a wrong `sizeof` | Removed together with the debug code |
| B11 | The last iteration output only the first front mixed with stale rows | The output is exactly the non-dominated front of the final population |
| B12 | Solutions outside the front could win the crowding sort | Selection by a composite key (rank, −crowding) |

### Optimizations

| Area | Before | Now |
|---|---|---|
| Fitness | 3 dense O(n³) matrix products, 32×32-thread blocks (≈10 % useful with n = 10), uncoalesced accesses and serialized constant memory | O(n²), one warp per chromosome, matrices in *shared memory*, *shuffle* reduction |
| NSGA-II | Host loop per front, with ~10 kernels and copies per front; bitonic sort with 2-thread blocks (28 launches per sort) | A single kernel per generation, everything in *shared memory* |
| Greedy 2-opt | ~50 API calls per evaluated pair (full fitness, `cudaMalloc`/`cudaFree`, copies) | One launch per generation, O(n) delta |
| Launch configuration | 13 kernels with 1 thread per block (1/32 SIMT efficiency) | 1 thread or 1 warp per element, 128-thread blocks |
| Transfers | Always-on debug copies (~1,150 per generation) | Only 6 copies at the end of the run |
| Synchronization | `cudaDeviceSynchronize` after every kernel | None during the run |
| Scalability | Serial runs (`TIMES` loop) | Concurrent `--runs R` (`blockIdx.y = run`) |

### Engineering

- Monolithic `kernel.cu` (2,081 lines) → modules with host/device separation.
- 15 `settings_*.cu` files recompiled per instance → instance and parameters at runtime.
- Ignored errors → `CUDA_CHECK` / `CUDA_CHECK_KERNEL` that abort with file and line.
- Unversioned Visual Studio project (excluded by `.gitignore`) → versioned `cuda_mqap.slnx` + `CMakeLists.txt`.
- No tests → `test_kernels` + `--verify` + `compute-sanitizer`.

### Behavior differences

- The greedy 2-opt evaluates each pair once. With the "all objectives" criterion it compares the exact
  sum of the changes instead of averages truncated to integers.
- The result file contains only the non-dominated solutions of the final population.
- The minimum population is 16 (KC10-2fl-2uni used 4) and it must be a power of 2.
- The tournament adversary is chosen uniformly among the P survivors.

---

## Population size limits and GPU resources

**In this branch the maximum population is P = 512 on any GPU, for all 15 instances.** The NSGA-II
survival of each run is still done by a single block of 2P threads, but it no longer keeps a dominance
matrix in shared memory. It uses the fast non-dominated sorting scheme of NSGA-II instead:
- every thread counts its dominators once;
- for each front, the members are listed in shared memory and the remaining threads subtract the members
  that dominated them.

The work is still O(N²) (N = 2P), but the shared memory grows linearly with P: about 30 KB with P = 512,
which fits in the default 48 KB of every GPU. The limit is now the number of threads of a block (1024), so
P ≤ 512. Larger populations need the multi-block design of the branch `develop_large_population_multiblock`.

### Measured on the RTX 2060

- **P = 512 works on all 15 instances**, with the iterations of each tab of the workbook and `--verify`.
- **For P ≤ 256 the results are identical to `develop_with_claude_opus_5`**: with the same seed, the result
  files are byte for byte the same (KC10, KC20 and KC30, 1 and 30 runs), and the GPU time is lower.

| Instance (iterations) | develop, P = 256 | this branch, P = 256 | this branch, P = 512 |
|---|---|---|---|
| KC10-2fl-1rl (70), 1 run | 30 ms | 19 ms | 26 ms |
| KC10-2fl-1rl (70), 30 runs | 128 ms | 86 ms | 153 ms |
| KC20-2fl-1rl (300), 1 run | 157 ms | 95 ms | 177 ms |
| KC20-2fl-1rl (300), 30 runs | 1,531 ms | 1,407 ms | 2,809 ms |
| KC30-3fl-1rl (70), 1 run | 64 ms | 61 ms | 113 ms |
| KC30-3fl-1rl (70), 30 runs | 1,013 ms | 1,001 ms | 2,001 ms |

Shared memory of the survival kernel, `S(P) = 2P·(18 + 4·OBJ) + 16` bytes:

| P | Threads per block | 2 objectives | 3 objectives | Before (dominance matrix, 3 objectives) |
|---|---|---|---|---|
| 128 | 256 | 6.5 KB | 7.5 KB | 15 KB |
| 256 | 512 | 13 KB | 15 KB | 47 KB |
| **512** | 1024 | 26 KB | **30 KB** | 160 KB (did not fit) |
| 1024 | 2048 (not allowed) | — | — | — |

### Effect of the population size on quality

KC10 instances, 70 iterations, 100 runs per case. The first figure is the gamma distance to the
published Pareto optimal front, mean ± standard deviation (lower is better; computed like
`mQAPMetrics/distance_metric_*.js`). The percentage is the average share of the optimal front found per run.

| Instance | P = 64 | P = 256 | P = 512 |
|---|---|---|---|
| KC10-2fl-1rl | 965.91 ± 679.81 · 68 % | 782.85 ± 452.71 · 75 % | 814.86 ± 509.29 · 77 % |
| KC10-2fl-2rl | 4,182.16 ± 5,728.91 · 93 % | 234.13 ± 1,096.94 · 99 % | 54.88 ± 546.02 · 100 % |
| KC10-2fl-3rl | 21,228.30 ± 3,031.55 · 53 % | 17,904.42 ± 1,742.56 · 62 % | 17,375.47 ± 1,862.39 · 65 % |
| KC10-2fl-4rl | 7,320.49 ± 2,151.26 · 53 % | 5,028.91 ± 1,180.20 · 61 % | 4,332.74 ± 541.21 · 64 % |
| KC10-2fl-5rl | 29,054.12 ± 12,426.16 · 50 % | 16,866.45 ± 6,464.29 · 62 % | 14,516.44 ± 6,799.74 · 65 % |
| KC10-2fl-1uni | 17.88 ± 46.44 · 85 % | 27.71 ± 56.61 · 90 % | 16.26 ± 44.87 · 91 % |
| KC10-2fl-3uni | 347.64 ± 58.36 · 31 % | 160.14 ± 38.27 · 68 % | 112.40 ± 31.24 · 72 % |

A larger population explores more of the front: the share of the optimal front found grows with P on
every instance. Each doubling of P roughly doubles the time, so compare it with running more
generations or more runs (`--runs`) for the same time budget.

### How to compute the limit for another GPU

P must be a power of 2 and at least 16. The maximum P is the largest one that meets:

1. **Threads:** 2P ≤ maximum threads per block (1024 on all current GPUs), so P ≤ 512.
2. **Shared memory of the survival kernel:** `S(P) = 2P·(18 + 4·OBJ) + 16` bytes ≤ 48 KB. It holds for
   every P ≤ 512 (30 KB at most), so it no longer depends on the GPU.
3. **Code cap:** P ≤ 512 (`kMaxPopulation` in `include/config.h`).

The VRAM only determines how many concurrent runs fit:

```
max runs       = min(65 535, free VRAM / memory per run)
memory per run = 2P·(4n + 8·OBJ + 64) + 8P bytes        (n = number of facilities)
```

With P = 512 a run takes 124 KB on KC10, 164 KB on KC20 and 212 KB on KC30. On the RTX 2060 (5 GB free)
about 24,700 concurrent runs of KC30 fit (calculated, not executed).

| GPU | VRAM | Max. P (this branch) | Concurrent KC30 runs with P = 512 (calculated) |
|---|---|---|---|
| RTX 2060 (measured) | 6 GB | **512** | ~24,700 |
| RTX 3070 Laptop | 8 GB | **512** | ~33,600 |
| RTX 3080 | 10–12 GB | **512** | ~42,000–50,000 |
| RTX 4080 / 4090 | 16 / 24 GB | **512** | 65,535 (program maximum) |
| RTX 5080 / 5090 | 16 / 32 GB | **512** | 65,535 |

The run counts use 85 % of the VRAM of each GPU (except the RTX 2060, whose free memory was measured).

---

## Limitations and future work

**Current limits:**
- n ≤ 64. The available *shared memory* also matters: with 3 objectives, n ≤ 63 on GPUs with 64 KB *opt-in*.
- P is a power of 2 between 16 and 512, because survival uses a block of 2P threads (at most 1024) and bitonic sort
  (see [Population size limits and GPU resources](#population-size-limits-and-gpu-resources)).
- Only 2 or 3 objectives are supported (the kernels are instantiated for those values).
- Costs are stored as 32-bit integers; the loader rejects instances that could overflow them.

**Possible improvements:**
- Island model with migration between the concurrent runs.
- CUDA Graphs to capture a generation; with 3 kernels per generation, the expected benefit is small.
- Nsight Compute analysis of the `survivalKernel`, which is latency bound because it is a single block.
- More crossover operators and variants of the 2-opt criterion.

---

## Troubleshooting

| Symptom | Cause and solution |
|---|---|
| `CUDA Toolkit X.Y Visual Studio integration not found` | The CUDA Toolkit was installed before Visual Studio, or without its *Visual Studio Integration*: re-run the CUDA installer (custom install → Visual Studio Integration). With several toolkits installed, choose one with `CudaVersion` (see [Open in Visual Studio 2026](#open-in-visual-studio-2026-plug-and-play)) |
| `no kernel image is available for execution on the device` | The GPU is older than `sm_75`, or the driver is too old to JIT-compile the PTX: update the driver or add the architecture in `cuda_mqap.props` (`CodeGeneration`) |
| Visual Studio asks to install components when opening the solution | It comes from `.vsconfig`: accept to install the C++ workload and the Windows SDK |
| `population must be a power of two in [16, 512]` | Use 16, 32, 64, 128, 256 or 512 |
| `instance too large: … shared memory` | The instance does not fit in the block's *shared memory* (see limits) |
| `costs may overflow 32-bit fitness values` | The instance could overflow the 32-bit fitness |
| Very slow execution in Debug | Expected: Debug compiles device code with `-G` and synchronizes after every kernel. Use Release to measure |
| `[CUDA] … at <file>:<line>` | CUDA error with its exact location; for more detail, run under `compute-sanitizer` |

---

## Credits and license

### Credits

- **Author:** Andrés Pupiales Arévalo — <apupiales@gmail.com> — <https://github.com/apupiales>. Project started in May 2019.
- **mQAP instances:** J. Knowles and D. Corne; Pareto optimal fronts by G. Lamont (see [Third-party data](#third-party-data)).
- **Refactoring and optimization of the current version:** done with the assistance of Claude (Anthropic).

### License

Copyright (C) 2019-2026 Andrés Pupiales Arévalo.

This program is free software: you can redistribute it and/or modify it under the terms of the
**GNU General Public License** as published by the Free Software Foundation, either **version 3** of the
License, or (at your option) any later version (`SPDX-License-Identifier: GPL-3.0-or-later`). It is
distributed in the hope that it will be useful, but **without any warranty**. See the full text in
[`LICENSE`](LICENSE).

Every source file carries the corresponding header.

**About CUDA.** NVIDIA's CUDA Toolkit (the `nvcc` compiler, the `cudart` runtime and the cuRAND headers)
**is not part of this repository**: it is proprietary NVIDIA software, distributed under its own license
(CUDA EULA), and it is required to build and run the program. The GPL v3 license covers only the code
of this project.

### Third-party data

The files in `mQAPData/` are **not covered by the GPL v3** license of this project: they are
third-party benchmark data, redistributed unmodified for academic and research use.

- **Instances (`.dat`):** mQAP test suite by Joshua Knowles and David Corne, generated with their
  generators `makeQAPuni`/`makeQAPrl` ((C) J. Knowles, 2002). The original page is no longer online;
  [archived copy](https://web.archive.org/web/2019/http://www.cs.bham.ac.uk/~jdk/mQAP/).
- **Pareto optimal fronts (`.PO`):** enumeration of the ten-facility instances by Gary Lamont, published
  on the same page.
- **Copy used:** [fredizzimo/keyboardlayout](https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData)
  (identical files). The MIT license of that repository does not cover this data, whose authors are the ones above.
- **Terms:** no explicit license is published. The original page offers the generators as free software
  *"for academic or educational use"* and asks to contact the author for commercial use; apply the
  same criterion to the instances.
- **Citation:** J. D. Knowles and D. W. Corne, *Instance Generators and Test Suites for the Multiobjective
  Quadratic Assignment Problem*, EMO 2003, LNCS 2632, pp. 295–310, Springer, 2003.

Details and BibTeX entry in [`mQAPData/README.txt`](mQAPData/README.txt).
