/*
 * config.h
 *
 * Compile-time limits and genetic operator settings.
 * Instance size, population size, iterations and runs are runtime options (see solver.h).
 */
#pragma once

namespace mqap {

// Largest supported number of facilities/locations. The effective limit also depends on the
// shared memory available per block (flow + distance matrices), checked at runtime.
constexpr int kMaxFacilities = 64;

// Population size (P) limits. P must be a power of two: the NSGA-II survival kernel runs in a
// single block of 2P threads and uses bitonic sorts in shared memory.
constexpr int kMinPopulation = 16;
constexpr int kMaxPopulation = 256;

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
