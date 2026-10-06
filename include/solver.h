/*
 * solver.h
 *
 * NSGA-II + adapted greedy 2-opt for the mQAP on the GPU.
 *
 * Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <vector>

#include "instance.h"

namespace mqap {

struct SolverOptions {
    int population = 64;             // P (power of two in [kMinPopulation, kMaxPopulation]).
    int iterations = 70;             // Generations of the genetic algorithm.
    int runs = 1;                    // Independent runs executed concurrently on the GPU.
    unsigned long long seed = 0;     // Seed of the random number generator.
    // How much of the population the greedy 2-opt improves. 1.0f and 1 are what the original version
    // does, every offspring of every generation; lower values trade local search for diversity, which is
    // what the small instances lose when the population is large. include/best_configuration.h holds the
    // value measured best for each instance, and the README explains the trade-off.
    float greedyRate = 1.0f;         // Fraction of the offspring improved, in [0, 1]; 0 disables it.
    int greedyPeriod = 1;            // The local search runs on the generations multiple of this.
    // Records the front of every generation in SolveStats::trace, to study how the search converges.
    // It copies the survivors to the host once per generation, so it synchronizes with the device and
    // the measured time is no longer comparable with a normal run.
    bool trace = false;
    int traceMaxPoints = 4096;       // Points kept per run and generation; the rest are not recorded.
    int traceEvery = 1;              // Generations between two recorded fronts; the last one is always
                                     // recorded. It bounds the size of the trace when the fronts are
                                     // large (3 objectives with a big population).
    // Keeps the initial population, the 2P random permutations of every run with their fitness, in
    // SolveStats::initialPopulation. It is a device-to-device copy queued before the first generation and
    // brought to the host after the timer stops, so the run neither synchronizes nor changes.
    bool recordInitial = false;
};

struct Solution {
    std::vector<short> permutation;  // permutation[i] = location of facility i.
    std::vector<unsigned int> fitness; // One value per objective.
    int rank = 0;                    // NSGA-II rank (1 = non-dominated).
};

struct RunResult {
    std::vector<Solution> population;  // Final population Pt (P solutions).
    std::vector<Solution> paretoFront; // Rank 1 solutions of the final population, each one once.
};

// One distinct non-dominated solution of one generation (SolverOptions::trace).
struct TracePoint {
    int run = 0;
    int generation = 0;                // 0 = survival of the initial population, iterations = final front.
    std::vector<unsigned int> fitness; // One value per objective.
};

struct SolveStats {
    float gpuMilliseconds = 0.0f;    // Time of the whole algorithm on the device.
    std::vector<TracePoint> trace;   // Front of every generation, when SolverOptions::trace is set.
    int traceTruncated = 0;          // Generations whose front did not fit in traceMaxPoints.
    std::vector<std::vector<Solution>> initialPopulation; // [run][2P], when SolverOptions::recordInitial is set.
    // Work done, over all runs, so that algorithms with different structures can be given the same budget:
    // full O(n^2) evaluations of a permutation (the initial population and every offspring), and O(n)
    // evaluations of one swap by the greedy 2-opt. Counted on the host from the deterministic gate.
    long long fullEvaluations = 0;
    long long swapEvaluations = 0;
};

// Fills SolveStats::fullEvaluations and swapEvaluations for a run with these options on n facilities.
void countWork(int n, const SolverOptions& options, SolveStats& stats);

// The same algorithm on the CPU with OpenMP (solver_cpu.cpp), the baseline of the GPU version. The
// time it reports in SolveStats::gpuMilliseconds is the CPU time of the whole algorithm.
std::vector<RunResult> solveCpu(const Instance& instance, const SolverOptions& options, SolveStats* stats = nullptr);

// Throws std::invalid_argument when the options or the instance are not supported.
std::vector<RunResult> solve(const Instance& instance, const SolverOptions& options, SolveStats* stats = nullptr);

} // namespace mqap
