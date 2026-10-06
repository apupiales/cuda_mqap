/*
 * solver_cpu.cpp
 *
 * The same algorithm as solver.cu, on the CPU with OpenMP, as the baseline the GPU version is measured
 * against. Every step mirrors a kernel: Fisher-Yates initialization, the O(n^2) fitness, the NSGA-II
 * survival by dominator counting and front peeling (O(N^2), like the multi-block kernels), the crowding
 * distance with the range taken over the whole population, the binary tournament with two exchange
 * mutations and one transposition, and the greedy 2-opt with the O(n) swap delta, the pair traversal of
 * kGreedyFullPairs and the same deterministic gate (greedyApplies). Only the random generator differs
 * (xoshiro256** instead of Philox), so for one seed the runs differ, but not in distribution.
 *
 * The loops over individuals are parallel; the runs are processed one after the other, each with all
 * the threads, which is what keeps every core busy at any population.
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

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "config.h"
#include "gate.h"

namespace mqap {

namespace {

// xoshiro256** seeded through splitmix64: one independent state per row, like the Philox states.
struct Rng {
    std::uint64_t s[4];

    static std::uint64_t splitmix(std::uint64_t& x) {
        std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    Rng(std::uint64_t seed = 0, std::uint64_t stream = 0) {
        std::uint64_t x = seed ^ (0xD1B54A32D192ED03ULL * (stream + 1));
        for (std::uint64_t& v : s) {
            v = splitmix(x);
        }
    }

    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    std::uint32_t next() {
        const std::uint64_t result = rotl(s[1] * 5, 7) * 9;
        const std::uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return static_cast<std::uint32_t>(result >> 32);
    }

    // Uniform in (0, 1], like curand_uniform.
    float uniform() { return (static_cast<float>(next() >> 8) + 1.0f) / 16777216.0f; }
};

struct Problem {
    int n;
    int m;
    const int* flow;   // [m][n][n]
    const int* dist;   // [n][n]
};

long long fullCost(const Problem& pb, const short* p, int o) {
    const int* F = pb.flow + static_cast<size_t>(o) * pb.n * pb.n;
    long long acc = 0;
    for (int i = 0; i < pb.n; i++) {
        const int* Fi = F + i * pb.n;
        const int* Di = pb.dist + p[i] * pb.n;
        for (int j = 0; j < pb.n; j++) {
            acc += static_cast<long long>(Fi[j]) * Di[p[j]];
        }
    }
    return acc;
}

// Equation of warpSwapDelta, sequentially.
long long swapDelta(const Problem& pb, const short* p, int r, int s, int o) {
    const int n = pb.n;
    const int* F = pb.flow + static_cast<size_t>(o) * n * n;
    const int* D = pb.dist;
    const int pr = p[r];
    const int ps = p[s];
    long long delta = static_cast<long long>(F[r * n + r] - F[s * n + s]) * (D[ps * n + ps] - D[pr * n + pr])
                    + static_cast<long long>(F[r * n + s] - F[s * n + r]) * (D[ps * n + pr] - D[pr * n + ps]);
    for (int k = 0; k < n; k++) {
        if (k == r || k == s) {
            continue;
        }
        const int pk = p[k];
        delta += static_cast<long long>(F[k * n + r] - F[k * n + s]) * (D[pk * n + ps] - D[pk * n + pr])
               + static_cast<long long>(F[r * n + k] - F[s * n + k]) * (D[ps * n + pk] - D[pr * n + pk]);
    }
    return delta;
}

bool dominates(const unsigned int* a, const unsigned int* b, int m) {
    bool lessOrEqual = true;
    bool less = false;
    for (int o = 0; o < m; o++) {
        lessOrEqual &= a[o] <= b[o];
        less |= a[o] < b[o];
    }
    return lessOrEqual && less;
}

// NSGA-II survival of one run: rows [0, 2P) -> the best P by (rank, -crowding).
void survival(const unsigned int* fitness, int population, int m, std::vector<int>& index, std::vector<int>& rankOut,
              std::vector<float>& crowdingOut) {
    const int total = 2 * population;
    std::vector<int> dominators(total);
    std::vector<int> rank(total, 0);
    std::vector<float> crowding(total, 0.0f);

#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < total; i++) {
        int count = 0;
        for (int j = 0; j < total; j++) {
            count += dominates(fitness + j * m, fitness + i * m, m);
        }
        dominators[i] = count;
    }

    // Front peeling: the unranked individuals without remaining dominators form the next front; the rest
    // subtract the members of that front that dominate them.
    int remaining = total;
    std::vector<int> front;
    for (int level = 1; remaining > 0; level++) {
        front.clear();
        for (int i = 0; i < total; i++) {
            if (rank[i] == 0 && dominators[i] == 0) {
                front.push_back(i);
            }
        }
        for (int i : front) {
            rank[i] = level;
        }
        remaining -= static_cast<int>(front.size());
        const int members = static_cast<int>(front.size());
#pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < total; i++) {
            if (rank[i] != 0) {
                continue;
            }
            int count = 0;
            for (int f = 0; f < members; f++) {
                count += dominates(fitness + front[f] * m, fitness + i * m, m);
            }
            dominators[i] -= count;
        }
    }

    // Crowding distance: each front contiguous after sorting by (rank, f_o); the range is the whole population.
    std::vector<int> order(total);
    for (int o = 0; o < m; o++) {
        unsigned int lo = std::numeric_limits<unsigned int>::max();
        unsigned int hi = 0;
        for (int i = 0; i < total; i++) {
            lo = std::min(lo, fitness[i * m + o]);
            hi = std::max(hi, fitness[i * m + o]);
        }
        const float range = static_cast<float>(hi - lo);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return rank[a] != rank[b] ? rank[a] < rank[b] : fitness[a * m + o] < fitness[b * m + o];
        });
        for (int q = 0; q < total; q++) {
            const int id = order[q];
            const bool first = q == 0 || rank[order[q - 1]] != rank[id];
            const bool last = q == total - 1 || rank[order[q + 1]] != rank[id];
            if (first || last) {
                crowding[id] = std::numeric_limits<float>::infinity();
            } else if (range > 0.0f) {
                crowding[id] += static_cast<float>(fitness[order[q + 1] * m + o] - fitness[order[q - 1] * m + o]) / range;
            }
        }
    }

    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return rank[a] != rank[b] ? rank[a] < rank[b] : crowding[a] > crowding[b];
    });
    for (int i = 0; i < population; i++) {
        index[i] = order[i];
        rankOut[i] = rank[order[i]];
        crowdingOut[i] = crowding[order[i]];
    }
}

} // namespace

std::vector<RunResult> solveCpu(const Instance& instance, const SolverOptions& options, SolveStats* stats) {
    const int population = options.population;
    if (population < kMinPopulation || population > kMaxPopulation || (population & (population - 1)) != 0) {
        throw std::invalid_argument("population must be a power of two in [16, 65536]");
    }
    if (instance.objectives < kMinObjectives || instance.objectives > kMaxObjectives) {
        throw std::invalid_argument("only 2 and 3 objective instances are supported");
    }
    if (stats != nullptr) {
        countWork(instance.n, options, *stats);
    }

    const Problem pb{instance.n, instance.objectives, instance.flow.data(), instance.dist.data()};
    const int n = pb.n;
    const int m = pb.m;
    const int rows = 2 * population;
    const auto begin = std::chrono::steady_clock::now();

    std::vector<RunResult> results(options.runs);
    for (int run = 0; run < options.runs; run++) {
        std::vector<short> genes(static_cast<size_t>(rows) * n);
        std::vector<short> nextGenes(genes.size());
        std::vector<unsigned int> fitness(static_cast<size_t>(rows) * m);
        std::vector<unsigned int> nextFitness(fitness.size());
        std::vector<Rng> rng(rows);
        for (int r = 0; r < rows; r++) {
            rng[r] = Rng(options.seed, static_cast<std::uint64_t>(run) * rows + r);
        }

        // Initial population (Fisher-Yates) and its fitness.
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            short* p = genes.data() + static_cast<size_t>(r) * n;
            for (int k = 0; k < n; k++) {
                p[k] = static_cast<short>(k);
            }
            for (int k = n - 1; k > 0; k--) {
                std::swap(p[k], p[rng[r].next() % (k + 1)]);
            }
            for (int o = 0; o < m; o++) {
                fitness[static_cast<size_t>(r) * m + o] = static_cast<unsigned int>(fullCost(pb, p, o));
            }
        }
        if (options.recordInitial && stats != nullptr) {
            if (stats->initialPopulation.size() != static_cast<size_t>(options.runs)) {
                stats->initialPopulation.assign(options.runs, {});
            }
            for (int r = 0; r < rows; r++) {
                Solution s;
                s.permutation.assign(genes.begin() + static_cast<size_t>(r) * n, genes.begin() + static_cast<size_t>(r + 1) * n);
                s.fitness.assign(fitness.begin() + static_cast<size_t>(r) * m, fitness.begin() + static_cast<size_t>(r + 1) * m);
                stats->initialPopulation[run].push_back(std::move(s));
            }
        }

        std::vector<int> index(population);
        std::vector<int> rank(population);
        std::vector<float> crowding(population);
        int greedyType = 0;
        const int every = options.traceEvery > 0 ? options.traceEvery : 1;

        for (int generation = 0; generation <= options.iterations; generation++) {
            survival(fitness.data(), population, m, index, rank, crowding);

            if (options.trace && stats != nullptr && (generation % every == 0 || generation == options.iterations)) {
                std::set<std::vector<unsigned int>> seen;
                for (int i = 0; i < population && rank[i] == 1; i++) {
                    std::vector<unsigned int> values(fitness.begin() + static_cast<size_t>(index[i]) * m,
                                                     fitness.begin() + static_cast<size_t>(index[i] + 1) * m);
                    if (!seen.insert(values).second) {
                        continue;
                    }
                    if (static_cast<int>(seen.size()) > options.traceMaxPoints) {
                        stats->traceTruncated++;
                        break;
                    }
                    stats->trace.push_back(TracePoint{run, generation, values});
                }
            }
            if (generation == options.iterations) {
                break;
            }

            // Reproduction: survivors to rows [0, P), mutated tournament winners to rows [P, 2P).
#pragma omp parallel for schedule(static)
            for (int i = 0; i < population; i++) {
                std::copy_n(genes.begin() + static_cast<size_t>(index[i]) * n, n, nextGenes.begin() + static_cast<size_t>(i) * n);
                std::copy_n(fitness.begin() + static_cast<size_t>(index[i]) * m, m, nextFitness.begin() + static_cast<size_t>(i) * m);

                Rng& state = rng[i];
                const int adversary = static_cast<int>(state.next() % population);
                const bool iWins = rank[i] < rank[adversary] || (rank[i] == rank[adversary] && crowding[i] > crowding[adversary]);
                const int winner = iWins ? i : adversary;
                short* child = nextGenes.data() + static_cast<size_t>(population + i) * n;
                std::copy_n(genes.begin() + static_cast<size_t>(index[winner]) * n, n, child);

                for (int k = 0; k < kExchangeMutations; k++) {
                    if (state.uniform() <= kExchangeMutationProbability) {
                        const int a = static_cast<int>(state.next() % n);
                        const int b = static_cast<int>(state.next() % n);
                        std::swap(child[a], child[b]);
                    }
                }
                if (state.uniform() <= kTranspositionMutationProbability) {
                    int lo = static_cast<int>(state.next() % n);
                    int hi = static_cast<int>(state.next() % n);
                    if (lo > hi) {
                        std::swap(lo, hi);
                    }
                    std::reverse(child + lo, child + hi + 1);
                }
                if (i == 0) {
                    greedyType = static_cast<int>(state.next() % (m + 1));
                }
            }

            // Greedy 2-opt on the offspring, which also gives them their fitness.
#pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < population; i++) {
                short* p = nextGenes.data() + static_cast<size_t>(population + i) * n;
                long long cost[kMaxObjectives];
                for (int o = 0; o < m; o++) {
                    cost[o] = fullCost(pb, p, o);
                }
                if (greedyApplies(options.seed, run, i, generation, options.greedyRate, options.greedyPeriod)) {
                    for (int r = 0; r < n - 1; r++) {
                        for (int s = kGreedyFullPairs ? 1 : r + 1; s < n; s++) {
                            if (kGreedyFullPairs && r == s) {
                                continue;
                            }
                            long long delta[kMaxObjectives];
                            long long sum = 0;
                            for (int o = 0; o < m; o++) {
                                delta[o] = swapDelta(pb, p, r, s, o);
                                sum += delta[o];
                            }
                            const long long criterion = greedyType == 0 ? sum : delta[greedyType - 1];
                            if (criterion <= 0) {
                                std::swap(p[r], p[s]);
                                for (int o = 0; o < m; o++) {
                                    cost[o] += delta[o];
                                }
                            }
                        }
                    }
                }
                for (int o = 0; o < m; o++) {
                    nextFitness[static_cast<size_t>(population + i) * m + o] = static_cast<unsigned int>(cost[o]);
                }
            }
            std::swap(genes, nextGenes);
            std::swap(fitness, nextFitness);
        }

        // Final population: the survivors of the last survival step; the front without repetitions.
        std::set<std::vector<short>> seen;
        for (int i = 0; i < population; i++) {
            Solution solution;
            solution.permutation.assign(genes.begin() + static_cast<size_t>(index[i]) * n,
                                        genes.begin() + static_cast<size_t>(index[i] + 1) * n);
            solution.fitness.assign(fitness.begin() + static_cast<size_t>(index[i]) * m,
                                    fitness.begin() + static_cast<size_t>(index[i] + 1) * m);
            solution.rank = rank[i];
            if (solution.rank == 1 && seen.insert(solution.permutation).second) {
                results[run].paretoFront.push_back(solution);
            }
            results[run].population.push_back(std::move(solution));
        }
    }

    if (stats != nullptr) {
        stats->gpuMilliseconds = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
    }
    return results;
}

} // namespace mqap
