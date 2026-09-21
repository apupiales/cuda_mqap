/*
 * nsga2_multiblock.cu
 *
 * NSGA-II survival split across many blocks, for populations that do not fit in one block
 * (P > kSingleBlockMaxPopulation, up to kMaxPopulation). Per run, with N = 2P:
 *   1. countDominatorsKernel: number of individuals that dominate each one, O(N^2) comparisons
 *      with the fitness read in shared memory tiles.
 *   2. peelFrontsKernel (cooperative launch, grid-wide synchronization): the individuals without
 *      remaining dominators form the next front and are appended to a front list; the others
 *      subtract the members of that front that dominated them. Repeated until everyone is ranked.
 *   3. Crowding distance: for each objective, a segmented radix sort (CUB) by (rank, fitness)
 *      leaves each front contiguous; boundaries get infinity, interior points add
 *      (f[next] - f[prev]) / (max - min) with max/min over the whole population, exactly like the
 *      single-block kernel.
 *   4. Selection: segmented sort by (rank ascending, crowding descending); the first P survive.
 * Every buffer is O(N) per run, so the VRAM needed grows linearly with P; the time grows with N^2.
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

#include <algorithm>
#include <climits>

#include <cooperative_groups.h>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <math_constants.h>

#include "config.h"
#include "cuda_check.cuh"
#include "survival_workspace.cuh"

namespace mqap {

namespace detail {

namespace cg = cooperative_groups;

constexpr int kMultiblockThreads = 256;

template <int OBJ>
__device__ inline bool dominatesValues(const unsigned int* a, const unsigned int* b) {
    bool lessOrEqual = true;
    bool less = false;
#pragma unroll
    for (int o = 0; o < OBJ; o++) {
        lessOrEqual &= a[o] <= b[o];
        less |= a[o] < b[o];
    }
    return lessOrEqual && less;
}

// Grid (ceil(N / 256), R). Also resets rank and crowding.
template <int OBJ>
__global__ void countDominatorsKernel(const unsigned int* __restrict__ fitness, int total,
                                      int* __restrict__ dominators, int* __restrict__ rank,
                                      float* __restrict__ crowding) {
    __shared__ unsigned int tile[kMultiblockThreads * OBJ];
    const int run = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int* runFitness = fitness + static_cast<size_t>(run) * total * OBJ;

    unsigned int mine[OBJ];
#pragma unroll
    for (int o = 0; o < OBJ; o++) {
        mine[o] = (i < total) ? runFitness[i * OBJ + o] : 0u;
    }

    int count = 0;
    for (int base = 0; base < total; base += kMultiblockThreads) {
        const int j = base + threadIdx.x;
        if (j < total) {
#pragma unroll
            for (int o = 0; o < OBJ; o++) {
                tile[threadIdx.x * OBJ + o] = runFitness[j * OBJ + o];
            }
        }
        __syncthreads();
        const int limit = min(kMultiblockThreads, total - base);
        if (i < total) {
            for (int k = 0; k < limit; k++) {
                count += dominatesValues<OBJ>(&tile[k * OBJ], mine);
            }
        }
        __syncthreads();
    }

    if (i < total) {
        const size_t item = static_cast<size_t>(run) * total + i;
        dominators[item] = count;
        rank[item] = 0;
        crowding[item] = 0.0f;
    }
}

// Cooperative launch: every block is resident, grid.sync() separates the phases of each front.
// frontSize[0][*] and flags[0] must be zero at launch.
template <int OBJ>
__global__ void peelFrontsKernel(const unsigned int* __restrict__ fitness, int total, int runs,
                                 int* __restrict__ dominators, int* __restrict__ rank,
                                 int* __restrict__ frontList, int* __restrict__ frontSize, int* flags) {
    cg::grid_group grid = cg::this_grid();
    const long long items = static_cast<long long>(runs) * total;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    const long long first = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    int parity = 0;
    for (int front = 1;; front++) {
        int* size = frontSize + parity * runs;

        // Phase A: individuals without remaining dominators join the front.
        for (long long item = first; item < items; item += stride) {
            if (rank[item] == 0 && dominators[item] == 0) {
                const int run = static_cast<int>(item / total);
                rank[item] = front;
                const int position = atomicAdd(&size[run], 1);
                frontList[static_cast<size_t>(run) * total + position] = static_cast<int>(item - static_cast<long long>(run) * total);
            }
        }
        grid.sync();

        // Phase B: the others subtract the members of the front that dominated them.
        for (long long item = first; item < items; item += stride) {
            if (rank[item] == 0) {
                const int run = static_cast<int>(item / total);
                const unsigned int* runFitness = fitness + static_cast<size_t>(run) * total * OBJ;
                const unsigned int* mine = fitness + static_cast<size_t>(item) * OBJ;
                const int* members = frontList + static_cast<size_t>(run) * total;
                const int count = size[run];
                int removed = 0;
                for (int k = 0; k < count; k++) {
                    removed += dominatesValues<OBJ>(runFitness + static_cast<size_t>(members[k]) * OBJ, mine);
                }
                dominators[item] -= removed;
                flags[parity] = 1;
            }
        }
        // Reset the buffers of the next front (nobody uses them during this phase).
        for (long long run = first; run < runs; run += stride) {
            frontSize[(1 - parity) * runs + run] = 0;
        }
        if (first == 0) {
            flags[1 - parity] = 0;
        }
        grid.sync();

        if (flags[parity] == 0) {
            break;
        }
        parity ^= 1;
    }
}

// Grid (ceil(N / 256), R): keys (rank, fitness of objective o), local indices, min/max per run.
template <int OBJ>
__global__ void crowdingKeysKernel(const unsigned int* __restrict__ fitness, int total, int objective,
                                   const int* __restrict__ rank, unsigned long long* __restrict__ keys,
                                   int* __restrict__ values, unsigned int* minFitness, unsigned int* maxFitness) {
    const int run = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const size_t item = static_cast<size_t>(run) * total + i;
    const unsigned int value = fitness[item * OBJ + objective];
    keys[item] = (static_cast<unsigned long long>(static_cast<unsigned int>(rank[item])) << 32) | value;
    values[item] = i;
    atomicMin(&minFitness[run], value);
    atomicMax(&maxFitness[run], value);
}

// Grid (ceil(N / 256), R): thread q handles sorted position q of its run.
template <int OBJ>
__global__ void crowdingAccumulateKernel(const unsigned int* __restrict__ fitness, int total, int objective,
                                         const int* __restrict__ rank, const int* __restrict__ sorted,
                                         const unsigned int* __restrict__ minFitness,
                                         const unsigned int* __restrict__ maxFitness, float* __restrict__ crowding) {
    const int run = blockIdx.y;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= total) {
        return;
    }
    const size_t base = static_cast<size_t>(run) * total;
    const int* order = sorted + base;
    const int id = order[q];
    const int r = rank[base + id];
    const bool first = (q == 0) || rank[base + order[q - 1]] != r;
    const bool last = (q == total - 1) || rank[base + order[q + 1]] != r;
    const float range = static_cast<float>(maxFitness[run] - minFitness[run]);
    if (first || last) {
        crowding[base + id] = CUDART_INF_F;
    } else if (range > 0.0f) {
        crowding[base + id] += static_cast<float>(fitness[(base + order[q + 1]) * OBJ + objective] -
                                                  fitness[(base + order[q - 1]) * OBJ + objective]) / range;
    }
}

// Grid (ceil(N / 256), R): keys (rank ascending, crowding descending).
__global__ void selectionKeysKernel(int total, const int* __restrict__ rank, const float* __restrict__ crowding,
                                    unsigned long long* __restrict__ keys, int* __restrict__ values) {
    const int run = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const size_t item = static_cast<size_t>(run) * total + i;
    keys[item] = (static_cast<unsigned long long>(static_cast<unsigned int>(rank[item])) << 32) |
                 (0xffffffffu - __float_as_uint(crowding[item]));
    values[item] = i;
}

// Grid (ceil(P / 256), R): the first P sorted individuals of each run survive.
__global__ void survivorsKernel(int total, int population, const int* __restrict__ sorted,
                                const int* __restrict__ rank, const float* __restrict__ crowding,
                                int* __restrict__ survivorIndex, int* __restrict__ survivorRank,
                                float* __restrict__ survivorCrowding) {
    const int run = blockIdx.y;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= population) {
        return;
    }
    const size_t base = static_cast<size_t>(run) * total;
    const int id = sorted[base + q];
    const size_t out = static_cast<size_t>(run) * population + q;
    survivorIndex[out] = id;
    survivorRank[out] = rank[base + id];
    survivorCrowding[out] = crowding[base + id];
}

__global__ void segmentOffsetsKernel(int* offsets, int runs, int total) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r <= runs) {
        offsets[r] = r * total;
    }
}

int bitsFor(int value) {
    int bits = 0;
    while ((1 << bits) <= value) {
        bits++;
    }
    return bits;
}

void segmentedSort(SurvivalWorkspace& ws, int endBit) {
    const int items = ws.runs * ws.total;
    size_t bytes = ws.sortTempBytes;
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
        ws.sortTemp.get(), bytes, ws.keysIn.get(), ws.keysOut.get(), ws.valuesIn.get(), ws.valuesOut.get(),
        items, ws.runs, ws.offsets.get(), ws.offsets.get() + 1, 0, endBit));
    CUDA_CHECK_KERNEL();
}

template <int OBJ>
int cooperativeBlocks(int items) {
    static int maxBlocks = 0;
    if (maxBlocks == 0) {
        int device = 0;
        int sms = 0;
        int perSm = 0;
        int cooperative = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
        if (!cooperative) {
            std::fprintf(stderr, "The GPU does not support cooperative launches (needed for P > %d)\n",
                         kSingleBlockMaxPopulation);
            std::exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, peelFrontsKernel<OBJ>, kMultiblockThreads, 0));
        maxBlocks = sms * perSm;
    }
    return std::max(1, std::min(maxBlocks, (items + kMultiblockThreads - 1) / kMultiblockThreads));
}

} // namespace detail

SurvivalWorkspace::SurvivalWorkspace(int population_, int runs_, bool forceMultiblock)
    : population(population_), runs(runs_), total(2 * population_),
      multiblock_(forceMultiblock || population_ > kSingleBlockMaxPopulation) {
    if (!multiblock_) {
        return;
    }
    const size_t items = static_cast<size_t>(runs) * total;
    dominators = DeviceBuffer<int>(items);
    rank = DeviceBuffer<int>(items);
    crowding = DeviceBuffer<float>(items);
    frontList = DeviceBuffer<int>(items);
    frontSize = DeviceBuffer<int>(2 * static_cast<size_t>(runs));
    flags = DeviceBuffer<int>(2);
    keysIn = DeviceBuffer<unsigned long long>(items);
    keysOut = DeviceBuffer<unsigned long long>(items);
    valuesIn = DeviceBuffer<int>(items);
    valuesOut = DeviceBuffer<int>(items);
    minFitness = DeviceBuffer<unsigned int>(runs);
    maxFitness = DeviceBuffer<unsigned int>(runs);
    offsets = DeviceBuffer<int>(static_cast<size_t>(runs) + 1);

    detail::segmentOffsetsKernel<<<(runs + 1 + 255) / 256, 256>>>(offsets.get(), runs, total);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, sortTempBytes, keysIn.get(), keysOut.get(), valuesIn.get(), valuesOut.get(),
        static_cast<int>(items), runs, offsets.get(), offsets.get() + 1));
    sortTemp = DeviceBuffer<unsigned char>(std::max<size_t>(sortTempBytes, 1));
}

size_t SurvivalWorkspace::deviceBytes() const {
    return dominators.size() * sizeof(int) + rank.size() * sizeof(int) + crowding.size() * sizeof(float) +
           frontList.size() * sizeof(int) + frontSize.size() * sizeof(int) + flags.size() * sizeof(int) +
           (keysIn.size() + keysOut.size()) * sizeof(unsigned long long) +
           (valuesIn.size() + valuesOut.size()) * sizeof(int) +
           (minFitness.size() + maxFitness.size()) * sizeof(unsigned int) + offsets.size() * sizeof(int) +
           sortTemp.size();
}

template <int OBJ>
void launchSurvivalMultiblock(const unsigned int* fitness, int population, int runs,
                              int* survivorIndex, int* survivorRank, float* survivorCrowding,
                              SurvivalWorkspace& ws) {
    using detail::kMultiblockThreads;
    const int total = 2 * population;
    const dim3 grid((total + kMultiblockThreads - 1) / kMultiblockThreads, runs);
    const int rankBits = detail::bitsFor(total);

    // 1. Dominator counts.
    detail::countDominatorsKernel<OBJ><<<grid, kMultiblockThreads>>>(fitness, total, ws.dominators.get(),
                                                                    ws.rank.get(), ws.crowding.get());
    CUDA_CHECK_KERNEL();

    // 2. Fronts (cooperative launch).
    CUDA_CHECK(cudaMemsetAsync(ws.frontSize.get(), 0, ws.frontSize.size() * sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(ws.flags.get(), 0, ws.flags.size() * sizeof(int)));
    int* dominators = ws.dominators.get();
    int* rank = ws.rank.get();
    int* frontList = ws.frontList.get();
    int* frontSize = ws.frontSize.get();
    int* flags = ws.flags.get();
    int totalArg = total;
    int runsArg = runs;
    void* args[] = {&fitness, &totalArg, &runsArg, &dominators, &rank, &frontList, &frontSize, &flags};
    const int blocks = detail::cooperativeBlocks<OBJ>(runs * total);
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(detail::peelFrontsKernel<OBJ>),
                                           blocks, kMultiblockThreads, args, 0, nullptr));
    CUDA_CHECK_KERNEL();

    // 3. Crowding distance, one segmented sort per objective.
    for (int o = 0; o < OBJ; o++) {
        CUDA_CHECK(cudaMemsetAsync(ws.minFitness.get(), 0xff, ws.minFitness.size() * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemsetAsync(ws.maxFitness.get(), 0, ws.maxFitness.size() * sizeof(unsigned int)));
        detail::crowdingKeysKernel<OBJ><<<grid, kMultiblockThreads>>>(fitness, total, o, ws.rank.get(), ws.keysIn.get(),
                                                                     ws.valuesIn.get(), ws.minFitness.get(),
                                                                     ws.maxFitness.get());
        CUDA_CHECK_KERNEL();
        detail::segmentedSort(ws, 32 + rankBits);
        detail::crowdingAccumulateKernel<OBJ><<<grid, kMultiblockThreads>>>(fitness, total, o, ws.rank.get(),
                                                                           ws.valuesOut.get(), ws.minFitness.get(),
                                                                           ws.maxFitness.get(), ws.crowding.get());
        CUDA_CHECK_KERNEL();
    }

    // 4. Selection by (rank ascending, crowding descending).
    detail::selectionKeysKernel<<<grid, kMultiblockThreads>>>(total, ws.rank.get(), ws.crowding.get(),
                                                              ws.keysIn.get(), ws.valuesIn.get());
    CUDA_CHECK_KERNEL();
    detail::segmentedSort(ws, 32 + 32);
    const dim3 survivorsGrid((population + kMultiblockThreads - 1) / kMultiblockThreads, runs);
    detail::survivorsKernel<<<survivorsGrid, kMultiblockThreads>>>(total, population, ws.valuesOut.get(), ws.rank.get(),
                                                                   ws.crowding.get(), survivorIndex, survivorRank,
                                                                   survivorCrowding);
    CUDA_CHECK_KERNEL();
}

template void launchSurvivalMultiblock<2>(const unsigned int*, int, int, int*, int*, float*, SurvivalWorkspace&);
template void launchSurvivalMultiblock<3>(const unsigned int*, int, int, int*, int*, float*, SurvivalWorkspace&);

} // namespace mqap
