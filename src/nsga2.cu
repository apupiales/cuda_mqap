/*
 * nsga2.cu
 *
 * NSGA-II survival (Rt = Pt U Qt  ->  Pt+1) of each run in a single block of 2P threads:
 *   1. Dominance matrix packed in bits (shared memory) and fast non-dominated sorting with
 *      __popc / __ballot_sync: rank 1 is the first Pareto front.
 *   2. Crowding distance of every front: one bitonic sort per objective by (rank, fitness),
 *      which leaves each front contiguous. Boundaries get infinity, interior points add
 *      (f[next] - f[prev]) / (max - min) with max/min taken over the whole population.
 *   3. Selection of the best P individuals by (rank ascending, crowding descending).
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

#include <climits>
#include <math_constants.h>

#include "config.h"
#include "cuda_check.cuh"
#include "device_common.cuh"
#include "survival_workspace.cuh"

namespace mqap {

namespace detail {

size_t survivalSharedMemory(int total, int objectives) {
    const int words = total / 32;
    return static_cast<size_t>(total) * sizeof(unsigned long long)       // sKey
         + static_cast<size_t>(total) * objectives * sizeof(unsigned int) // sFit
         + static_cast<size_t>(total) * words * sizeof(unsigned int)      // sDominatedBy
         + static_cast<size_t>(total) * sizeof(float)                     // sCrowding
         + static_cast<size_t>(words) * sizeof(unsigned int)              // sFrontMask
         + static_cast<size_t>(total) * sizeof(short) * 2;                // sRank, sIndex
}

// blockIdx.x = run, blockDim.x = 2P (power of two, multiple of 32).
template <int OBJ>
__global__ void survivalKernel(const unsigned int* __restrict__ fitness, int population,
                               int* __restrict__ survivorIndex, int* __restrict__ survivorRank,
                               float* __restrict__ survivorCrowding) {
    const int total = blockDim.x;
    const int words = total / 32;

    extern __shared__ unsigned long long smem64[];
    unsigned long long* sKey = smem64;                                                // total
    unsigned int* sFit = reinterpret_cast<unsigned int*>(sKey + total);               // total * OBJ
    unsigned int* sDominatedBy = sFit + total * OBJ;                                  // total * words
    float* sCrowding = reinterpret_cast<float*>(sDominatedBy + total * words);        // total
    unsigned int* sFrontMask = reinterpret_cast<unsigned int*>(sCrowding + total);    // words
    short* sRank = reinterpret_cast<short*>(sFrontMask + words);                      // total
    short* sIndex = sRank + total;                                                    // total
    __shared__ int sRemaining;
    __shared__ unsigned int sMin;
    __shared__ unsigned int sMax;

    const int run = blockIdx.x;
    const int i = threadIdx.x;
    const int lane = i & 31;
    const int warp = i >> 5;

    const unsigned int* runFitness = fitness + static_cast<size_t>(run) * total * OBJ;
#pragma unroll
    for (int o = 0; o < OBJ; o++) {
        sFit[i * OBJ + o] = runFitness[i * OBJ + o];
    }
    sRank[i] = 0;
    sCrowding[i] = 0.0f;
    if (i == 0) {
        sRemaining = total;
    }
    __syncthreads();

    // 1. Row i of the dominance matrix: bit j is set when j dominates i.
    for (int w = 0; w < words; w++) {
        unsigned int bits = 0;
        for (int b = 0; b < 32; b++) {
            const int j = w * 32 + b;
            bool lessOrEqual = true;
            bool less = false;
#pragma unroll
            for (int o = 0; o < OBJ; o++) {
                lessOrEqual &= sFit[j * OBJ + o] <= sFit[i * OBJ + o];
                less |= sFit[j * OBJ + o] < sFit[i * OBJ + o];
            }
            bits |= static_cast<unsigned int>(lessOrEqual && less) << b;
        }
        sDominatedBy[i * words + w] = bits;
    }
    __syncthreads();

    // Peel the fronts: individuals without remaining dominators form the next front.
    for (short front = 1; sRemaining > 0; front++) {
        int dominators = 0;
        for (int w = 0; w < words; w++) {
            dominators += __popc(sDominatedBy[i * words + w]);
        }
        const bool inFront = (sRank[i] == 0) && (dominators == 0);
        const unsigned int mask = __ballot_sync(kFullWarpMask, inFront);
        if (lane == 0) {
            sFrontMask[warp] = mask;
        }
        __syncthreads();
        if (inFront) {
            sRank[i] = front;
            atomicSub(&sRemaining, 1);
        }
        for (int w = 0; w < words; w++) {
            sDominatedBy[i * words + w] &= ~sFrontMask[w];
        }
        __syncthreads();
    }

    // 2. Crowding distance. Thread q handles sorted position q.
    for (int o = 0; o < OBJ; o++) {
        if (i == 0) {
            sMin = UINT_MAX;
            sMax = 0;
        }
        __syncthreads();
        const unsigned int value = sFit[i * OBJ + o];
        atomicMin(&sMin, value);
        atomicMax(&sMax, value);
        sKey[i] = (static_cast<unsigned long long>(static_cast<unsigned short>(sRank[i])) << 32) | value;
        sIndex[i] = static_cast<short>(i);
        __syncthreads();
        blockBitonicSort(sKey, sIndex, total);

        const float range = static_cast<float>(sMax - sMin);
        const int id = sIndex[i];
        const short rank = sRank[id];
        const bool first = (i == 0) || sRank[sIndex[i - 1]] != rank;
        const bool last = (i == total - 1) || sRank[sIndex[i + 1]] != rank;
        if (first || last) {
            sCrowding[id] = CUDART_INF_F;
        } else if (range > 0.0f) {
            sCrowding[id] += static_cast<float>(sFit[sIndex[i + 1] * OBJ + o] - sFit[sIndex[i - 1] * OBJ + o]) / range;
        }
        __syncthreads();
    }

    // 3. Survivors: rank ascending, crowding descending (crowding >= 0, so its bits are ordered).
    sKey[i] = (static_cast<unsigned long long>(static_cast<unsigned short>(sRank[i])) << 32)
            | (0xffffffffu - __float_as_uint(sCrowding[i]));
    sIndex[i] = static_cast<short>(i);
    __syncthreads();
    blockBitonicSort(sKey, sIndex, total);

    if (i < population) {
        const size_t out = static_cast<size_t>(run) * population + i;
        const int id = sIndex[i];
        survivorIndex[out] = id;
        survivorRank[out] = sRank[id];
        survivorCrowding[out] = sCrowding[id];
    }
}

} // namespace detail

template <int OBJ>
void launchSurvival(const unsigned int* fitness, int population, int runs,
                    int* survivorIndex, int* survivorRank, float* survivorCrowding,
                    SurvivalWorkspace* workspace) {
    if (workspace != nullptr && workspace->multiblock()) {
        launchSurvivalMultiblock<OBJ>(fitness, population, runs, survivorIndex, survivorRank, survivorCrowding, *workspace);
        return;
    }
    const int total = 2 * population;
    detail::survivalKernel<OBJ><<<runs, total, detail::survivalSharedMemory(total, OBJ)>>>(
        fitness, population, survivorIndex, survivorRank, survivorCrowding);
    CUDA_CHECK_KERNEL();
}

template void launchSurvival<2>(const unsigned int*, int, int, int*, int*, float*, SurvivalWorkspace*);
template void launchSurvival<3>(const unsigned int*, int, int, int*, int*, float*, SurvivalWorkspace*);

} // namespace mqap
