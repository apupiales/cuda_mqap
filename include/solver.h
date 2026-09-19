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
};

struct Solution {
    std::vector<short> permutation;  // permutation[i] = location of facility i.
    std::vector<unsigned int> fitness; // One value per objective.
    int rank = 0;                    // NSGA-II rank (1 = non-dominated).
};

struct RunResult {
    std::vector<Solution> population;  // Final population Pt (P solutions).
    std::vector<Solution> paretoFront; // Rank 1 solutions of the final population.
};

struct SolveStats {
    float gpuMilliseconds = 0.0f;    // Time of the whole algorithm on the device.
};

// Throws std::invalid_argument when the options or the instance are not supported.
std::vector<RunResult> solve(const Instance& instance, const SolverOptions& options, SolveStats* stats = nullptr);

} // namespace mqap
