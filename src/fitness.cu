/*
 * fitness.cu
 *
 * Fitness of every chromosome: cost_k(p) = sum_i sum_j Fk[i][j] * D[p[i]][p[j]], which is equal to
 * Trace(Fk * X * DT * XT) with X the permutation matrix, in O(n^2) instead of O(n^3).
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
#include "device_common.cuh"

namespace mqap {

namespace detail {

// One warp per chromosome, kWarpsPerBlock chromosomes per block, blockIdx.y = run.
template <int OBJ>
__global__ void fitnessKernel(const short* __restrict__ genes, unsigned int* __restrict__ fitness,
                              const int* __restrict__ flow, const int* __restrict__ dist,
                              int rows, int n) {
    extern __shared__ int smem[];
    int* sFlow = smem;                                   // OBJ * n * n
    int* sDist = sFlow + OBJ * n * n;                    // n * n
    short* sPerm = reinterpret_cast<short*>(sDist + n * n); // kWarpsPerBlock * n

    loadMatricesToShared<OBJ>(flow, dist, sFlow, sDist, n);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarpsPerBlock + warp;
    const size_t runRow = static_cast<size_t>(blockIdx.y) * rows + row;
    short* p = sPerm + warp * n;

    if (row < rows) {
        for (int k = lane; k < n; k += 32) {
            p[k] = genes[runRow * n + k];
        }
    }
    __syncthreads();
    if (row >= rows) {
        return;
    }

#pragma unroll
    for (int o = 0; o < OBJ; o++) {
        const long long value = warpCost(sFlow + o * n * n, sDist, p, n, lane);
        if (lane == 0) {
            fitness[runRow * OBJ + o] = static_cast<unsigned int>(value);
        }
    }
}

} // namespace detail

size_t matricesSharedMemory(int n, int objectives) {
    return static_cast<size_t>(objectives * n * n + n * n) * sizeof(int)
         + static_cast<size_t>(kWarpsPerBlock) * n * sizeof(short);
}

template <int OBJ>
void launchFitness(const short* genes, unsigned int* fitness, const int* flow, const int* dist,
                   int rows, int n, int runs) {
    const size_t smem = matricesSharedMemory(n, OBJ);
    static bool attributeSet = false;
    if (!attributeSet) {
        // Allow more than the default 48 KB of dynamic shared memory (large instances).
        int maxOptin = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
        CUDA_CHECK(cudaFuncSetAttribute(detail::fitnessKernel<OBJ>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxOptin));
        attributeSet = true;
    }
    const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock, runs);
    detail::fitnessKernel<OBJ><<<grid, 32 * kWarpsPerBlock, smem>>>(genes, fitness, flow, dist, rows, n);
    CUDA_CHECK_KERNEL();
}

template void launchFitness<2>(const short*, unsigned int*, const int*, const int*, int, int, int);
template void launchFitness<3>(const short*, unsigned int*, const int*, const int*, int, int, int);

} // namespace mqap
