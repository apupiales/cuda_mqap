# (cuda_mqap) Parallel Implementation of NSGA-2 + Adapted Greedy 2opt in CUDA to solve mQAP

Program to solve instances of multiobjective quadratic assignment problems (mQAP) in CUDA C++.

What it includes:

1. Creation of the initial population (Fisher-Yates shuffle, one thread per chromosome).
2. Fitness calculation for each objective: `sum_i sum_j Fk[i][j] * D[p(i)][p(j)]`, which is equal to
   `Trace(Fk * X * DT * XT)` and is computed in O(n^2) by one warp per chromosome.
3. Parallel NSGA-II (one block per run, everything in shared memory):
   - [3.1] Dominance matrix (packed in bits).
   - [3.2] Total dominance (`__popc`).
   - [3.3] Pareto fronts and rank.
   - [3.4] Crowding distance (bitonic sort by rank and fitness for each objective).
   - [3.5] Selection of the next population by rank and crowding distance.
4. Binary tournament selection.
5. Mutation (exchange and transposition).
6. Adapted greedy 2opt with O(n) delta evaluation of every swap.

Every generation is three kernel launches (survival, reproduction, greedy 2opt) with no host
synchronization, and several independent runs are executed concurrently (`--runs`).

## Project layout

```
include/   config.h (limits), cuda_check.cuh (CUDA_CHECK), device_buffer.cuh (RAII),
           instance.h, solver.h, kernels.cuh (kernel launchers), device_common.cuh (device helpers)
src/       main.cpp (command line), instance.cpp (.dat parser), solver.cu (host orchestration),
           fitness.cu, nsga2.cu, operators.cu, local_search.cu (kernels)
tests/     test_kernels.cu (every kernel against a CPU reference)
scripts/   run_experiments.ps1 (instances and parameters of comparative_results_kcX_datasets.xlsx)
mQAPData/  instances (.dat) and published Pareto optimal fronts (.PO)
```

## Build

Requirements: CUDA Toolkit 13.4 and an NVIDIA GPU (the projects target `sm_75`, GeForce RTX 20xx;
add your architecture to `CodeGeneration` in `cuda_mqap.props` or to `CMAKE_CUDA_ARCHITECTURES`).

- **Visual Studio**: open `cuda_mqap.slnx` and build `Release|x64`. The executables are written to
  `build\x64\<Configuration>\`. Debug builds synchronize after every kernel (`MQAP_SYNC_CHECK`) to
  report errors at the failing launch.
- **CMake** (also from Visual Studio with *Open Folder*):

  ```
  cmake -S . -B build/cmake -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build build/cmake
  ctest --test-dir build/cmake --output-on-failure
  ```

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

Example: `build\x64\Release\cuda_mqap.exe mQAPData\KC20-2fl-1rl.dat --iterations 300 --runs 10 --verify`

The result file keeps the original format (one `{ ... },` block per run with the non-dominated
solutions: `'permutation': [f1, f2],`), so the scripts in `mQAPMetrics` can be used as before.

`--verify` recomputes on the CPU the fitness of every solution of the final population, checks that
every permutation is valid and that the rank 1 solutions are exactly the non-dominated ones.

To reproduce the experiments of `comparative_results_kcX_datasets.xlsx` (same population and
iterations per instance as the former `settings_*.cu` files):

```
.\scripts\run_experiments.ps1 -Runs 30
```

## Tests

`test_kernels` checks every kernel against an independent CPU implementation: fitness against the
literal `Trace(F*X*DT*XT)` and against the published `.PO` fronts, NSGA-II ranks, crowding and
selection, greedy 2opt against a CPU greedy that recomputes the full cost, reproduction and
initialization. Run it from the repository root (`test_kernels mQAPData`) or with `ctest`. It is also
worth running the program under `compute-sanitizer` (memcheck, racecheck, synccheck, initcheck).

## Performance

GeForce RTX 2060 (sm_75), Release builds. The previous version is `kernel.cu` at commit `3f3a187`
(the monolithic implementation with its memory errors fixed).

| Instance | Previous version | This version |
|---|---|---|
| KC10-2fl-1rl, P=64, 70 iterations, 1 run | 2.2 s | 0.15 s wall (15 ms GPU) |
| KC10-2fl-1rl, P=64, 70 iterations, 10 runs | 23.7 s | 0.12 s wall (21 ms GPU) |
| KC20-2fl-1rl, P=64, 300 iterations, 1 run | 48.8 s | 0.15 s wall (55 ms GPU) |
| KC30-3fl-1rl, P=32, 70 iterations, 1 run | 42.4 s | 0.14 s wall (40 ms GPU) |
| KC30-3fl-1rl, P=32, 70 iterations, 30 runs | ~21 min (estimated, 30 x 42.4 s) | 0.23 s wall (135 ms GPU) |

The wall time of this version is dominated by the creation of the CUDA context (~0.1 s).

Quality is unchanged: over 10 to 30 runs, the fraction of the published Pareto optimal front found
was 68.4 % in both versions for KC10-2fl-1rl, and 44.2 % (previous) vs 45.0 % (this version) for
KC10-2fl-3uni with P=128 and 25 iterations.

## Differences with the previous implementation

- The instance is read at runtime from the `.dat` file (both header formats are supported) instead of
  being compiled from a `settings_*.cu` file. The former `settings_KC20_2fl_1rl.cu` contained the
  matrices of KC20-2fl-2rl.
- The greedy 2opt visits every pair of positions once (the previous loop visited (i, j) and (j, i)),
  and with the "all objectives" criterion it compares the exact sum of the objective changes instead
  of averages truncated to integers.
- The final result contains only the non-dominated (rank 1) solutions of the final population.
- The minimum population is 16 (the former KC10-2fl-2uni setting used 4).
