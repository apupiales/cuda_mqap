/*
 * config.h
 *
 * Compile-time limits and genetic operator settings.
 * Instance size, population size, iterations and runs are runtime options (see solver.h).
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

namespace mqap {

// Largest supported number of facilities/locations. The effective limit also depends on the
// shared memory available per block (flow + distance matrices), checked at runtime.
constexpr int kMaxFacilities = 64;

// Population size (P) limits. P must be a power of two. Up to kSingleBlockMaxPopulation the NSGA-II
// survival of each run runs in a single block of 2P threads (bitonic sorts in shared memory); above
// it, the multi-block survival of nsga2_multiblock.cu is used. Survivor indices and ranks are int,
// so the cap is set by the cost of the O(N^2) dominance counting, not by the index type.
constexpr int kMinPopulation = 16;
constexpr int kSingleBlockMaxPopulation = 256;
constexpr int kMaxPopulation = 65536;

// Number of objectives supported by the kernels (they are instantiated for these values).
constexpr int kMinObjectives = 2;
constexpr int kMaxObjectives = 3;

// Chromosomes (one warp each) per block in the fitness and greedy 2-opt kernels.
constexpr int kWarpsPerBlock = 4;

// Threads per block for kernels that use one thread per chromosome.
constexpr int kThreadsPerBlock = 128;

// Mutation settings (applied to every tournament winner).
constexpr int kExchangeMutations = 2;
constexpr float kExchangeMutationProbability = 1.0f;
constexpr float kTranspositionMutationProbability = 1.0f;

} // namespace mqap
