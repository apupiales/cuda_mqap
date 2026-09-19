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
 */
#pragma once

#include <curand_kernel.h>

namespace mqap {

using RngState = curandStatePhilox4_32_10_t;

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
// 2P individuals of each run (one block of 2P threads per run). Writes, for each run, the P
// survivors ordered by (rank ascending, crowding descending).
template <int OBJ>
void launchSurvival(const unsigned int* fitness, int population, int runs,
                    short* survivorIndex, short* survivorRank, float* survivorCrowding);

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
