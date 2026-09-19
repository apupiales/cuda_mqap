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
13. [Limitations and future work](#limitations-and-future-work)
14. [Troubleshooting](#troubleshooting)
15. [Credits and license](#credits-and-license)

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
- Complete NSGA-II **inside a single block per run**: the dominance matrix is packed in bits, the fronts
  are extracted with `__ballot_sync`/`__popc` and the bitonic sorts run in *shared memory*.
- Greedy 2-opt with **O(n) incremental (delta) evaluation** of each swap; the local search of the whole
  offspring is a single kernel.
- Persistent Philox random states, initialized only once.
- **Independent runs in parallel** (`--runs R`) to use the whole GPU in experiment campaigns.

**Engineering**
- Strict host/device separation: `main.cpp` contains no CUDA code, and kernels are exposed through launcher functions.
- Instances read from the `.dat` files at runtime; parameters are passed on the command line.
- `CUDA_CHECK`/`CUDA_CHECK_KERNEL` error checking and RAII memory management (`DeviceBuffer<T>`).
- Versioned Visual Studio solution (`cuda_mqap.slnx`) and `CMakeLists.txt` with `ctest`.
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
| Population size `P` | `--population` | 64 (power of 2 between 16 and 256) |
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
├── cuda_mqap.slnx, cuda_mqap.vcxproj, test_kernels.vcxproj, cuda_mqap.props   Visual Studio
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
| `survivalKernel<OBJ>` | `R` × `2P` | 1 block per run, 1 thread per individual | Bit-packed dominance (`2P × 2P/32` words), fronts with `__ballot_sync` + `__popc`, bitonic sort of 64-bit keys `(rank, fitness)` and `(rank, −crowding)` in *shared memory* |
| `reproduceKernel<OBJ>` | `(⌈P/128⌉, R)` × 128 | 1 thread per offspring | Philox state in registers; tournament, mutations and copy in a single pass |
| `greedy2OptKernel<OBJ>` | `(⌈P/4⌉, R)` × 128 | 1 warp per offspring | Matrices in *shared memory*; O(n) delta split across the 32 lanes; warp-uniform criterion (no divergence) |
| `initPopulationKernel` | `(⌈2P/128⌉, R)` × 128 | 1 thread per chromosome | Unbiased Fisher-Yates |
| `rngInitKernel` | `⌈R·2P/128⌉` × 128 | 1 thread per state | One independent Philox subsequence per thread |

*Shared memory* per block:
- **Fitness and 2-opt:** `(OBJ + 1)·n²·4 + 4·n·2` bytes, e.g. 14.6 KB for n = 30 and 3 objectives.
  When more than 48 KB are needed, the device's maximum *opt-in* is requested automatically
  (`cudaFuncAttributeMaxDynamicSharedMemorySize`), which allows n = 60 with 3 objectives on Turing.
- **Survival:** up to ~46 KB with P = 256.

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

- NVIDIA GPU with *compute capability* ≥ 7.5. The projects build for `sm_75` (GeForce RTX 20xx);
  for other GPUs, add their architecture (see [Build](#build)).
- **CUDA Toolkit 13.4**. The `.vcxproj` files import `CUDA 13.4.props`; with another version, change that
  line in both `.vcxproj` files.
- Windows: Visual Studio 2026 (toolset v145) with the CUDA integration. Alternative: CMake ≥ 3.24 + Ninja,
  both included with Visual Studio.

---

## Build

### Visual Studio

1. Open `cuda_mqap.slnx`.
2. Select `Release | x64` and build the solution.
3. The executables are written to `build\x64\Release\` (`cuda_mqap.exe` and `test_kernels.exe`).

The `cuda_mqap` project already includes debugging arguments (`mQAPData\KC10-2fl-1rl.dat --verify`) and
its working directory is the repository root, so F5 works directly. The common configuration is in
`cuda_mqap.props`:
- Architecture `compute_75,sm_75`.
- C++17 and `/W4`.
- `-lineinfo` in Release.
- `-G` and `MQAP_SYNC_CHECK` in Debug.

### CMake

```
cmake -S . -B build/cmake -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake
ctest --test-dir build/cmake --output-on-failure
```

For other architectures: `-DCMAKE_CUDA_ARCHITECTURES="75;86;89"`. In Visual Studio you can also use
*File → Open → Folder*.

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
  --population P   population size, power of two in [16, 256] (default 64)
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

---

## Tests and validation

`test_kernels` (the `test_kernels` project in Visual Studio, or `ctest`) compares every kernel with an
independent CPU implementation:

| Test | What it checks |
|---|---|
| Instance loading | All 23 `.dat` files load (in both header formats) and `n`/`m` match the file name |
| `.PO` optimal fronts | The **374 published optimal solutions** have exactly their published cost, both on the CPU and on the GPU |
| Fitness | The kernel matches the original version's literal `Trace(F·X·Dᵀ·Xᵀ)` on KC10, KC20 and KC30, with several runs |
| NSGA-II survival | Ranks, crowding and selection match a CPU NSGA-II for P = 16, 64 and 256, with 2 and 3 objectives |
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

The speedup does not change the quality of the obtained front.

---

## Improvements over the original version

### Fixed bugs

| # | Bug in the original version | Fix |
|---|---|---|
| B1 | 1 `curandState` was allocated but up to 8,192 were initialized, writing out of bounds in GPU memory | Philox states sized per thread (`[R][2P]`) and persistent |
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

## Limitations and future work

**Current limits:**
- n ≤ 64. The available *shared memory* also matters: with 3 objectives, n ≤ 63 on GPUs with 64 KB *opt-in*.
- P is a power of 2 between 16 and 256, because survival uses a block of 2P threads and bitonic sort.
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
| Visual Studio cannot find `CUDA 13.4.props` | Another toolkit version is installed: change `CUDA 13.4` in the `.vcxproj` files to yours |
| `no kernel image is available for execution on the device` | The GPU is not `sm_75`: add its architecture in `cuda_mqap.props` (`CodeGeneration`) or in `CMAKE_CUDA_ARCHITECTURES` |
| `population must be a power of two in [16, 256]` | Use 16, 32, 64, 128 or 256 |
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
