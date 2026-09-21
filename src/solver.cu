/*
 * solver.cu
 *
 * Host orchestration. All device memory is allocated once; every generation is 3 kernel
 * launches (survival, reproduction, greedy 2-opt) on the default stream without host
 * synchronization. The results are copied back only at the end.
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
#include "solver.h"

#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "config.h"
#include "cuda_check.cuh"
#include "device_buffer.cuh"
#include "kernels.cuh"

namespace mqap {

namespace {

void validate(const Instance& instance, const SolverOptions& options) {
    const int p = options.population;
    if (p < kMinPopulation || p > kMaxPopulation || (p & (p - 1)) != 0) {
        throw std::invalid_argument("population must be a power of two in [" + std::to_string(kMinPopulation) +
                                    ", " + std::to_string(kMaxPopulation) + "]");
    }
    if (options.iterations < 0) {
        throw std::invalid_argument("iterations must be >= 0");
    }
    if (options.runs < 1 || options.runs > 65535) {
        throw std::invalid_argument("runs must be in [1, 65535]");
    }
    int maxOptin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (matricesSharedMemory(instance.n, instance.objectives) > static_cast<size_t>(maxOptin)) {
        throw std::invalid_argument("instance too large: the flow and distance matrices do not fit in the "
                                    "shared memory of one block (" + std::to_string(maxOptin) + " bytes)");
    }
}

template <int OBJ>
std::vector<RunResult> solveImpl(const Instance& instance, const SolverOptions& options, SolveStats* stats) {
    const int n = instance.n;
    const int population = options.population;
    const int rows = 2 * population;
    const int runs = options.runs;
    const size_t totalRows = static_cast<size_t>(runs) * rows;

    DeviceBuffer<int> flow(instance.flow.size());
    DeviceBuffer<int> dist(instance.dist.size());
    flow.copyFromHost(instance.flow);
    dist.copyFromHost(instance.dist);

    // Double buffered population: current -> next every generation.
    DeviceBuffer<short> genesA(totalRows * n);
    DeviceBuffer<short> genesB(totalRows * n);
    DeviceBuffer<unsigned int> fitnessA(totalRows * OBJ);
    DeviceBuffer<unsigned int> fitnessB(totalRows * OBJ);
    DeviceBuffer<short> survivorIndex(static_cast<size_t>(runs) * population);
    DeviceBuffer<short> survivorRank(static_cast<size_t>(runs) * population);
    DeviceBuffer<float> survivorCrowding(static_cast<size_t>(runs) * population);
    DeviceBuffer<int> greedyType(runs);
    DeviceBuffer<RngState> rng(totalRows);

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    short* genes = genesA.get();
    short* nextGenes = genesB.get();
    unsigned int* fitness = fitnessA.get();
    unsigned int* nextFitness = fitnessB.get();

    launchRngInit(rng.get(), static_cast<int>(totalRows), options.seed);
    launchInitPopulation(rng.get(), genes, rows, n, runs);
    launchFitness<OBJ>(genes, fitness, flow.get(), dist.get(), rows, n, runs);

    for (int iteration = 0; iteration <= options.iterations; iteration++) {
        // Rt (2P) -> best P by rank and crowding distance.
        launchSurvival<OBJ>(fitness, population, runs, survivorIndex.get(), survivorRank.get(), survivorCrowding.get());
        if (iteration == options.iterations) {
            break;
        }
        // Pt+1 = survivors, Qt+1 = mutated tournament winners improved by greedy 2-opt.
        launchReproduce<OBJ>(genes, fitness, nextGenes, nextFitness,
                             survivorIndex.get(), survivorRank.get(), survivorCrowding.get(),
                             rng.get(), greedyType.get(), population, n, runs);
        launchGreedy2Opt<OBJ>(nextGenes, nextFitness, flow.get(), dist.get(), greedyType.get(), population, n, runs);
        std::swap(genes, nextGenes);
        std::swap(fitness, nextFitness);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    if (stats != nullptr) {
        CUDA_CHECK(cudaEventElapsedTime(&stats->gpuMilliseconds, start, stop));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Final population: the survivors of the last survival step.
    std::vector<short> hostGenes(totalRows * n);
    std::vector<unsigned int> hostFitness(totalRows * OBJ);
    CUDA_CHECK(cudaMemcpy(hostGenes.data(), genes, hostGenes.size() * sizeof(short), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostFitness.data(), fitness, hostFitness.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    const std::vector<short> hostIndex = survivorIndex.toHost();
    const std::vector<short> hostRank = survivorRank.toHost();

    std::vector<RunResult> results(runs);
    for (int run = 0; run < runs; run++) {
        // The front is what the program prints and writes, so a solution appears in it only once.
        // Repetitions are common: the population converges and many survivors are copies of each other.
        std::set<std::vector<short>> seen;
        for (int i = 0; i < population; i++) {
            const size_t source = static_cast<size_t>(run) * rows + hostIndex[static_cast<size_t>(run) * population + i];
            Solution solution;
            solution.permutation.assign(hostGenes.begin() + source * n, hostGenes.begin() + (source + 1) * n);
            solution.fitness.assign(hostFitness.begin() + source * OBJ, hostFitness.begin() + (source + 1) * OBJ);
            solution.rank = hostRank[static_cast<size_t>(run) * population + i];
            if (solution.rank == 1 && seen.insert(solution.permutation).second) {
                results[run].paretoFront.push_back(solution);
            }
            results[run].population.push_back(std::move(solution));
        }
    }
    return results;
}

} // namespace

std::vector<RunResult> solve(const Instance& instance, const SolverOptions& options, SolveStats* stats) {
    validate(instance, options);
    switch (instance.objectives) {
    case 2:
        return solveImpl<2>(instance, options, stats);
    case 3:
        return solveImpl<3>(instance, options, stats);
    default:
        throw std::invalid_argument("only 2 and 3 objective instances are supported");
    }
}

} // namespace mqap
