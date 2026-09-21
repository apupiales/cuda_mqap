/*
 * kernels.cuh
 *
 * Host launchers of the device kernels. Every launcher checks the launch with CUDA_CHECK_KERNEL()
 * and runs on the default stream, so consecutive launches are ordered without synchronizing.
 *
 * Memory layout (R independent runs, population P, n facilities, OBJ objectives):
 *   genes    short        [R][2P][n]   rows [0, P) survivors, rows [P, 2P) offspring
 *   fitness  unsigned int [R][2P][OBJ]
 *   survivor arrays (index/rank/crowding)  [R][P]
 *   rng      RngState     [R][2P]
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

#include <curand_kernel.h>

namespace mqap {

using RngState = curandStatePhilox4_32_10_t;

class SurvivalWorkspace; // survival_workspace.cuh

// Shared memory (bytes) needed by the fitness and greedy 2-opt kernels.
size_t matricesSharedMemory(int n, int objectives);

// One independent Philox subsequence per state.
void launchRngInit(RngState* states, int count, unsigned long long seed);

// Random permutations (Fisher-Yates) for every row of every run.
void launchInitPopulation(RngState* rng, short* genes, int rows, int n, int runs);

// Fitness of every row: one warp per chromosome, matrices in shared memory.
template <int OBJ>
void launchFitness(const short* genes, unsigned int* fitness, const int* flow, const int* dist,
                   int rows, int n, int runs);

// NSGA-II survival: non-dominated sorting, crowding distance and selection of the best P of the
// 2P individuals of each run. Writes, for each run, the P survivors ordered by (rank ascending,
// crowding descending). Uses one block of 2P threads per run, or the multi-block survival when the
// workspace says so (P > kSingleBlockMaxPopulation, or forced).
template <int OBJ>
void launchSurvival(const unsigned int* fitness, int population, int runs,
                    short* survivorIndex, short* survivorRank, float* survivorCrowding,
                    SurvivalWorkspace* workspace = nullptr);

// Multi-block NSGA-II survival (any population up to kMaxPopulation); see nsga2_multiblock.cu.
template <int OBJ>
void launchSurvivalMultiblock(const unsigned int* fitness, int population, int runs,
                              short* survivorIndex, short* survivorRank, float* survivorCrowding,
                              SurvivalWorkspace& workspace);

// Builds the next population of each run:
//   rows [0, P)  : survivors (with their fitness),
//   rows [P, 2P) : binary tournament winners after exchange and transposition mutations.
// Also draws the greedy 2-opt criterion of the generation for each run.
template <int OBJ>
void launchReproduce(const short* genes, const unsigned int* fitness,
                     short* nextGenes, unsigned int* nextFitness,
                     const short* survivorIndex, const short* survivorRank, const float* survivorCrowding,
                     RngState* rng, int* greedyType, int population, int n, int runs);

// Adapted greedy 2-opt on rows [P, 2P) of each run with O(n) delta evaluation. Leaves the
// improved permutations and their fitness in place. greedyType[run]: 0 = sum of all objectives,
// k = objective k only.
template <int OBJ>
void launchGreedy2Opt(short* genes, unsigned int* fitness, const int* flow, const int* dist,
                      const int* greedyType, int population, int n, int runs);

} // namespace mqap
