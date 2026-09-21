/*
 * operators.cu
 *
 * Random number states, initial population, binary tournament selection and mutations.
 * Every thread loads its curand state into registers, uses it and stores it back, so the states
 * are initialized only once per execution.
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
#include "kernels.cuh"

#include "config.h"
#include "cuda_check.cuh"

namespace mqap {

namespace detail {

__global__ void rngInitKernel(RngState* states, int count, unsigned long long seed) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        curand_init(seed, id, 0, &states[id]);
    }
}

// One thread per chromosome (blockIdx.y = run). Unbiased Fisher-Yates shuffle.
__global__ void initPopulationKernel(RngState* rng, short* genes, int rows, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const size_t runRow = static_cast<size_t>(blockIdx.y) * rows + row;
    RngState state = rng[runRow];
    short* p = genes + runRow * n;
    for (int k = 0; k < n; k++) {
        p[k] = static_cast<short>(k);
    }
    for (int k = n - 1; k > 0; k--) {
        const int r = curand(&state) % (k + 1);
        const short tmp = p[k];
        p[k] = p[r];
        p[r] = tmp;
    }
    rng[runRow] = state;
}

// One thread per offspring i in [0, P) (blockIdx.y = run).
template <int OBJ>
__global__ void reproduceKernel(const short* __restrict__ genes, const unsigned int* __restrict__ fitness,
                                short* __restrict__ nextGenes, unsigned int* __restrict__ nextFitness,
                                const int* __restrict__ survivorIndex, const int* __restrict__ survivorRank,
                                const float* __restrict__ survivorCrowding,
                                RngState* rng, int* greedyType, int population, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= population) {
        return;
    }
    const int run = blockIdx.y;
    const int rows = 2 * population;
    const size_t runBase = static_cast<size_t>(run) * rows;
    const int* index = survivorIndex + static_cast<size_t>(run) * population;
    const int* rank = survivorRank + static_cast<size_t>(run) * population;
    const float* crowding = survivorCrowding + static_cast<size_t>(run) * population;

    // Survivor i becomes row i of the next population (Pt+1).
    const size_t source = runBase + index[i];
    const size_t target = runBase + i;
    for (int k = 0; k < n; k++) {
        nextGenes[target * n + k] = genes[source * n + k];
    }
#pragma unroll
    for (int o = 0; o < OBJ; o++) {
        nextFitness[target * OBJ + o] = fitness[source * OBJ + o];
    }

    RngState state = rng[runBase + i];

    // Binary tournament against a random survivor: lower rank wins, then higher crowding.
    const int adversary = curand(&state) % population;
    const bool iWins = rank[i] < rank[adversary] ||
                       (rank[i] == rank[adversary] && crowding[i] > crowding[adversary]);
    const int winner = iWins ? i : adversary;

    short child[kMaxFacilities];
    const short* parent = genes + (runBase + index[winner]) * n;
    for (int k = 0; k < n; k++) {
        child[k] = parent[k];
    }

    // Exchange mutation: swap two random genes.
    for (int m = 0; m < kExchangeMutations; m++) {
        if (curand_uniform(&state) <= kExchangeMutationProbability) {
            const int a = curand(&state) % n;
            const int b = curand(&state) % n;
            const short tmp = child[a];
            child[a] = child[b];
            child[b] = tmp;
        }
    }

    // Transposition mutation: reverse the genes between two random positions.
    if (curand_uniform(&state) <= kTranspositionMutationProbability) {
        int lo = curand(&state) % n;
        int hi = curand(&state) % n;
        if (lo > hi) {
            const int tmp = lo;
            lo = hi;
            hi = tmp;
        }
        while (lo < hi) {
            const short tmp = child[lo];
            child[lo++] = child[hi];
            child[hi--] = tmp;
        }
    }

    // The offspring fitness is computed by the greedy 2-opt kernel.
    const size_t offspring = runBase + population + i;
    for (int k = 0; k < n; k++) {
        nextGenes[offspring * n + k] = child[k];
    }

    // Greedy 2-opt criterion of this generation: 0 = all objectives, k = objective k.
    if (i == 0) {
        greedyType[run] = curand(&state) % (OBJ + 1);
    }

    rng[runBase + i] = state;
}

} // namespace detail

void launchRngInit(RngState* states, int count, unsigned long long seed) {
    detail::rngInitKernel<<<(count + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(states, count, seed);
    CUDA_CHECK_KERNEL();
}

void launchInitPopulation(RngState* rng, short* genes, int rows, int n, int runs) {
    const dim3 grid((rows + kThreadsPerBlock - 1) / kThreadsPerBlock, runs);
    detail::initPopulationKernel<<<grid, kThreadsPerBlock>>>(rng, genes, rows, n);
    CUDA_CHECK_KERNEL();
}

template <int OBJ>
void launchReproduce(const short* genes, const unsigned int* fitness,
                     short* nextGenes, unsigned int* nextFitness,
                     const int* survivorIndex, const int* survivorRank, const float* survivorCrowding,
                     RngState* rng, int* greedyType, int population, int n, int runs) {
    const dim3 grid((population + kThreadsPerBlock - 1) / kThreadsPerBlock, runs);
    detail::reproduceKernel<OBJ><<<grid, kThreadsPerBlock>>>(genes, fitness, nextGenes, nextFitness,
                                                     survivorIndex, survivorRank, survivorCrowding,
                                                     rng, greedyType, population, n);
    CUDA_CHECK_KERNEL();
}

template void launchReproduce<2>(const short*, const unsigned int*, short*, unsigned int*,
                                 const int*, const int*, const float*, RngState*, int*, int, int, int);
template void launchReproduce<3>(const short*, const unsigned int*, short*, unsigned int*,
                                 const int*, const int*, const float*, RngState*, int*, int, int, int);

} // namespace mqap
