/*
 * device_common.cuh
 *
 * Device helpers shared by several kernels.
 */
#pragma once

#include <cuda_runtime.h>

namespace mqap {

constexpr unsigned int kFullWarpMask = 0xffffffffu;

// Copies the flow matrices and the distance matrix to shared memory (whole block, coalesced).
template <int OBJ>
__device__ inline void loadMatricesToShared(const int* __restrict__ flow, const int* __restrict__ dist,
                                            int* sFlow, int* sDist, int n) {
    for (int k = threadIdx.x; k < OBJ * n * n; k += blockDim.x) {
        sFlow[k] = flow[k];
    }
    for (int k = threadIdx.x; k < n * n; k += blockDim.x) {
        sDist[k] = dist[k];
    }
}

// Warp-wide sum, broadcast to every lane.
__device__ inline long long warpSum(long long value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return __shfl_sync(kFullWarpMask, value, 0);
}

// cost(p) = sum_i sum_j F[i][j] * D[p[i]][p[j]], computed by the 32 lanes of a warp.
// F[k] is read with consecutive k per lane (no shared memory bank conflicts).
__device__ inline long long warpCost(const int* F, const int* D, const short* p, int n, int lane) {
    long long acc = 0;
    for (int k = lane; k < n * n; k += 32) {
        const int i = k / n;
        const int j = k - i * n;
        acc += static_cast<long long>(F[k]) * D[p[i] * n + p[j]];
    }
    return warpSum(acc);
}

// Change of the cost when positions r and s of p are swapped, in O(n):
//   (F_rr - F_ss)(D_{ps ps} - D_{pr pr}) + (F_rs - F_sr)(D_{ps pr} - D_{pr ps})
//   + sum_{k != r,s} (F_kr - F_ks)(D_{pk ps} - D_{pk pr}) + (F_rk - F_sk)(D_{ps pk} - D_{pr pk})
__device__ inline long long warpSwapDelta(const int* F, const int* D, const short* p,
                                          int n, int r, int s, int lane) {
    const int pr = p[r];
    const int ps = p[s];
    long long delta = 0;
    for (int k = lane; k < n; k += 32) {
        if (k == r || k == s) {
            continue;
        }
        const int pk = p[k];
        delta += static_cast<long long>(F[k * n + r] - F[k * n + s]) * (D[pk * n + ps] - D[pk * n + pr])
               + static_cast<long long>(F[r * n + k] - F[s * n + k]) * (D[ps * n + pk] - D[pr * n + pk]);
    }
    delta = warpSum(delta);
    return delta
        + static_cast<long long>(F[r * n + r] - F[s * n + s]) * (D[ps * n + ps] - D[pr * n + pr])
        + static_cast<long long>(F[r * n + s] - F[s * n + r]) * (D[ps * n + pr] - D[pr * n + ps]);
}

// Ascending bitonic sort of (key, index) pairs in shared memory by the whole block.
// Requires blockDim.x == count and count a power of two. Returns synchronized.
__device__ inline void blockBitonicSort(unsigned long long* key, short* index, int count) {
    const int t = threadIdx.x;
    for (int k = 2; k <= count; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int partner = t ^ j;
            if (partner > t) {
                const bool ascending = (t & k) == 0;
                if ((key[t] > key[partner]) == ascending) {
                    const unsigned long long tmpKey = key[t];
                    key[t] = key[partner];
                    key[partner] = tmpKey;
                    const short tmpIndex = index[t];
                    index[t] = index[partner];
                    index[partner] = tmpIndex;
                }
            }
            __syncthreads();
        }
    }
}

} // namespace mqap
