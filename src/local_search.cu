/*
 * local_search.cu
 *
 * Adapted greedy 2-opt for the mQAP (see https://arxiv.org/ftp/arxiv/papers/1109/1109.1276.pdf).
 * Each warp improves one offspring: every pair of positions (r < s) is visited once in order and
 * the swap is kept when it does not worsen the criterion of the generation (sum of all objectives
 * or a single objective). The effect of a swap is evaluated in O(n) with warpSwapDelta, so the
 * whole local search of all offspring of all runs is a single kernel launch.
 */
#include "kernels.cuh"

#include "config.h"
#include "cuda_check.cuh"
#include "device_common.cuh"

namespace mqap {

namespace detail {

// One warp per offspring (rows [P, 2P)), blockIdx.y = run.
template <int OBJ>
__global__ void greedy2OptKernel(short* __restrict__ genes, unsigned int* __restrict__ fitness,
                                 const int* __restrict__ flow, const int* __restrict__ dist,
                                 const int* __restrict__ greedyType, int population, int n) {
    extern __shared__ int smem[];
    int* sFlow = smem;                                      // OBJ * n * n
    int* sDist = sFlow + OBJ * n * n;                       // n * n
    short* sPerm = reinterpret_cast<short*>(sDist + n * n); // kWarpsPerBlock * n

    loadMatricesToShared<OBJ>(flow, dist, sFlow, sDist, n);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int local = blockIdx.x * kWarpsPerBlock + warp;
    const size_t row = static_cast<size_t>(blockIdx.y) * 2 * population + population + local;
    short* p = sPerm + warp * n;

    if (local < population) {
        for (int k = lane; k < n; k += 32) {
            p[k] = genes[row * n + k];
        }
    }
    __syncthreads();
    if (local >= population) {
        return;
    }

    const int type = greedyType[blockIdx.y];
    long long cost[OBJ];
#pragma unroll
    for (int o = 0; o < OBJ; o++) {
        cost[o] = warpCost(sFlow + o * n * n, sDist, p, n, lane);
    }

    for (int r = 0; r < n - 1; r++) {
        for (int s = r + 1; s < n; s++) {
            long long delta[OBJ];
            long long sum = 0;
#pragma unroll
            for (int o = 0; o < OBJ; o++) {
                delta[o] = warpSwapDelta(sFlow + o * n * n, sDist, p, n, r, s, lane);
                sum += delta[o];
            }
            // Uniform across the warp: every lane holds the same deltas.
            const long long criterion = (type == 0) ? sum : delta[type - 1];
            if (criterion <= 0) {
                if (lane == 0) {
                    const short tmp = p[r];
                    p[r] = p[s];
                    p[s] = tmp;
                }
                __syncwarp();
#pragma unroll
                for (int o = 0; o < OBJ; o++) {
                    cost[o] += delta[o];
                }
            }
        }
    }

    for (int k = lane; k < n; k += 32) {
        genes[row * n + k] = p[k];
    }
    if (lane < OBJ) {
        fitness[row * OBJ + lane] = static_cast<unsigned int>(cost[lane]);
    }
}

} // namespace detail

template <int OBJ>
void launchGreedy2Opt(short* genes, unsigned int* fitness, const int* flow, const int* dist,
                      const int* greedyType, int population, int n, int runs) {
    const size_t smem = matricesSharedMemory(n, OBJ);
    static bool attributeSet = false;
    if (!attributeSet) {
        int maxOptin = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
        CUDA_CHECK(cudaFuncSetAttribute(detail::greedy2OptKernel<OBJ>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxOptin));
        attributeSet = true;
    }
    const dim3 grid((population + kWarpsPerBlock - 1) / kWarpsPerBlock, runs);
    detail::greedy2OptKernel<OBJ><<<grid, 32 * kWarpsPerBlock, smem>>>(genes, fitness, flow, dist, greedyType, population, n);
    CUDA_CHECK_KERNEL();
}

template void launchGreedy2Opt<2>(short*, unsigned int*, const int*, const int*, const int*, int, int, int);
template void launchGreedy2Opt<3>(short*, unsigned int*, const int*, const int*, const int*, int, int, int);

} // namespace mqap
