/*
 * test_kernels.cu
 *
 * Checks every kernel against an independent CPU reference.
 * Usage: test_kernels [path/to/mQAPData]   (default: mQAPData)
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "config.h"
#include "device_buffer.cuh"
#include "instance.h"
#include "kernels.cuh"

using namespace mqap;

namespace {

int failures = 0;

#define EXPECT(condition, ...)                                   \
    do {                                                         \
        if (!(condition)) {                                      \
            if (failures++ < 20) {                               \
                std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
                std::printf(__VA_ARGS__);                        \
                std::printf("\n");                               \
            }                                                    \
        }                                                        \
    } while (0)

std::vector<short> randomPermutations(int count, int n, std::mt19937& rng) {
    std::vector<short> genes(static_cast<size_t>(count) * n);
    for (int c = 0; c < count; c++) {
        std::iota(genes.begin() + static_cast<size_t>(c) * n, genes.begin() + static_cast<size_t>(c + 1) * n, static_cast<short>(0));
        std::shuffle(genes.begin() + static_cast<size_t>(c) * n, genes.begin() + static_cast<size_t>(c + 1) * n, rng);
    }
    return genes;
}

Instance randomInstance(int n, int objectives, std::mt19937& rng) {
    Instance instance;
    instance.name = "random";
    instance.n = n;
    instance.objectives = objectives;
    instance.dist.resize(n * n);
    instance.flow.resize(static_cast<size_t>(objectives) * n * n);
    for (int a = 0; a < n; a++) {
        for (int b = 0; b < n; b++) {
            instance.dist[a * n + b] = (a == b) ? 0 : static_cast<int>(rng() % 100);
        }
    }
    for (int& f : instance.flow) {
        f = static_cast<int>(rng() % 100);
    }
    return instance;
}

// Literal Trace(Fk * X * DT * XT) of the original program, with DT = matrix of the .dat file.
long long originalTraceFitness(const Instance& instance, const short* p, int objective) {
    const int n = instance.n;
    std::vector<long long> x(n * n, 0), a(n * n, 0), b(n * n, 0);
    for (int j = 0; j < n; j++) {
        x[j * n + p[j]] = 1;
    }
    const int* flow = instance.flow.data() + static_cast<size_t>(objective) * n * n;
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            long long s = 0;
            for (int q = 0; q < n; q++) {
                s += static_cast<long long>(flow[i * n + q]) * x[q * n + k];
            }
            a[i * n + k] = s;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            long long s = 0;
            for (int q = 0; q < n; q++) {
                const int fileDist = instance.dist[k * n + q]; // file[q][k] = dist[k][q]
                s += a[i * n + q] * fileDist;
            }
            b[i * n + k] = s;
        }
    }
    long long trace = 0;
    for (int i = 0; i < n; i++) {
        for (int q = 0; q < n; q++) {
            trace += b[i * n + q] * x[i * n + q];
        }
    }
    return trace;
}

template <int OBJ>
void testFitness(const Instance& instance, int rows, int runs) {
    std::mt19937 rng(11);
    const int n = instance.n;
    const std::vector<short> genes = randomPermutations(rows * runs, n, rng);
    DeviceBuffer<short> dGenes(genes.size());
    DeviceBuffer<unsigned int> dFitness(genes.size() / n * OBJ);
    DeviceBuffer<int> dFlow(instance.flow.size());
    DeviceBuffer<int> dDist(instance.dist.size());
    dGenes.copyFromHost(genes);
    dFlow.copyFromHost(instance.flow);
    dDist.copyFromHost(instance.dist);
    launchFitness<OBJ>(dGenes.get(), dFitness.get(), dFlow.get(), dDist.get(), rows, n, runs);
    const std::vector<unsigned int> fitness = dFitness.toHost();
    for (int c = 0; c < rows * runs; c++) {
        for (int o = 0; o < OBJ; o++) {
            const long long expected = originalTraceFitness(instance, &genes[static_cast<size_t>(c) * n], o);
            EXPECT(expected == fitness[static_cast<size_t>(c) * OBJ + o], "%s fitness row %d obj %d: %u != %lld",
                   instance.name.c_str(), c, o, fitness[static_cast<size_t>(c) * OBJ + o], expected);
        }
    }
}

// CPU NSGA-II survival: fronts, crowding (same float operations) and the best P by (rank, -crowding).
struct CpuSurvivor {
    int rank;
    float crowding;
};

std::vector<CpuSurvivor> cpuSurvival(const std::vector<unsigned int>& fit, int total, int objectives, int population) {
    auto dominates = [&](int a, int b) {
        bool le = true, lt = false;
        for (int o = 0; o < objectives; o++) {
            le &= fit[a * objectives + o] <= fit[b * objectives + o];
            lt |= fit[a * objectives + o] < fit[b * objectives + o];
        }
        return le && lt;
    };
    std::vector<int> rank(total, 0);
    int remaining = total;
    for (int front = 1; remaining > 0; front++) {
        std::vector<int> members;
        for (int i = 0; i < total; i++) {
            if (rank[i] != 0) continue;
            bool dominated = false;
            for (int j = 0; j < total && !dominated; j++) {
                dominated = rank[j] == 0 && dominates(j, i);
            }
            if (!dominated) members.push_back(i);
        }
        for (int i : members) rank[i] = front;
        remaining -= static_cast<int>(members.size());
    }
    std::vector<float> crowding(total, 0.0f);
    for (int o = 0; o < objectives; o++) {
        std::vector<int> order(total);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (rank[a] != rank[b]) return rank[a] < rank[b];
            return fit[a * objectives + o] < fit[b * objectives + o];
        });
        unsigned int lo = UINT32_MAX, hi = 0;
        for (int i = 0; i < total; i++) {
            lo = std::min(lo, fit[i * objectives + o]);
            hi = std::max(hi, fit[i * objectives + o]);
        }
        const float range = static_cast<float>(hi - lo);
        for (int q = 0; q < total; q++) {
            const int id = order[q];
            const bool first = q == 0 || rank[order[q - 1]] != rank[id];
            const bool last = q == total - 1 || rank[order[q + 1]] != rank[id];
            if (first || last) {
                crowding[id] = INFINITY;
            } else if (range > 0.0f) {
                crowding[id] += static_cast<float>(fit[order[q + 1] * objectives + o] - fit[order[q - 1] * objectives + o]) / range;
            }
        }
    }
    std::vector<int> order(total);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (rank[a] != rank[b]) return rank[a] < rank[b];
        return crowding[a] > crowding[b];
    });
    std::vector<CpuSurvivor> survivors;
    for (int q = 0; q < population; q++) {
        survivors.push_back({rank[order[q]], crowding[order[q]]});
    }
    return survivors;
}

template <int OBJ>
void testSurvival(int population, int runs) {
    std::mt19937 rng(100 + population * OBJ);
    const int total = 2 * population;
    // Distinct values per objective, so the sort order (and the crowding) is unique.
    std::vector<unsigned int> fitness(static_cast<size_t>(runs) * total * OBJ);
    for (int r = 0; r < runs; r++) {
        for (int o = 0; o < OBJ; o++) {
            std::vector<unsigned int> values(total);
            std::iota(values.begin(), values.end(), 1000u);
            std::shuffle(values.begin(), values.end(), rng);
            for (int i = 0; i < total; i++) {
                fitness[(static_cast<size_t>(r) * total + i) * OBJ + o] = values[i] * 7u + (rng() % 7u);
            }
        }
    }
    DeviceBuffer<unsigned int> dFitness(fitness.size());
    DeviceBuffer<short> dIndex(static_cast<size_t>(runs) * population);
    DeviceBuffer<short> dRank(static_cast<size_t>(runs) * population);
    DeviceBuffer<float> dCrowding(static_cast<size_t>(runs) * population);
    dFitness.copyFromHost(fitness);
    launchSurvival<OBJ>(dFitness.get(), population, runs, dIndex.get(), dRank.get(), dCrowding.get());
    const std::vector<short> index = dIndex.toHost();
    const std::vector<short> rank = dRank.toHost();
    const std::vector<float> crowding = dCrowding.toHost();

    for (int r = 0; r < runs; r++) {
        const std::vector<unsigned int> runFitness(fitness.begin() + static_cast<size_t>(r) * total * OBJ,
                                                   fitness.begin() + static_cast<size_t>(r + 1) * total * OBJ);
        const std::vector<CpuSurvivor> expected = cpuSurvival(runFitness, total, OBJ, population);
        std::vector<int> seen(total, 0);
        for (int q = 0; q < population; q++) {
            const size_t k = static_cast<size_t>(r) * population + q;
            EXPECT(index[k] >= 0 && index[k] < total && !seen[index[k]]++, "survival P=%d: invalid index", population);
            EXPECT(rank[k] == expected[q].rank, "survival P=%d OBJ=%d run %d pos %d: rank %d != %d",
                   population, OBJ, r, q, rank[k], expected[q].rank);
            const bool same = (std::isinf(expected[q].crowding) && std::isinf(crowding[k])) ||
                              crowding[k] == expected[q].crowding;
            EXPECT(same, "survival P=%d OBJ=%d run %d pos %d: crowding %f != %f",
                   population, OBJ, r, q, crowding[k], expected[q].crowding);
        }
    }
}

// CPU greedy 2-opt with full cost recomputation (no delta formula).
std::vector<short> cpuGreedy(const Instance& instance, std::vector<short> p, int type) {
    const int n = instance.n;
    std::vector<long long> current(instance.objectives);
    for (int o = 0; o < instance.objectives; o++) current[o] = cost(instance, p.data(), o);
    for (int r = 0; r < n - 1; r++) {
        for (int s = r + 1; s < n; s++) {
            std::swap(p[r], p[s]);
            std::vector<long long> candidate(instance.objectives);
            long long sum = 0;
            for (int o = 0; o < instance.objectives; o++) {
                candidate[o] = cost(instance, p.data(), o);
                sum += candidate[o] - current[o];
            }
            const long long criterion = type == 0 ? sum : candidate[type - 1] - current[type - 1];
            if (criterion <= 0) {
                current = candidate;
            } else {
                std::swap(p[r], p[s]);
            }
        }
    }
    return p;
}

template <int OBJ>
void testGreedy(int n, int population, int runs) {
    std::mt19937 rng(7 * n + OBJ);
    const Instance instance = randomInstance(n, OBJ, rng);
    const int rows = 2 * population;
    const std::vector<short> genes = randomPermutations(rows * runs, n, rng);
    std::vector<int> types(runs);
    for (int r = 0; r < runs; r++) types[r] = r % (OBJ + 1);

    DeviceBuffer<short> dGenes(genes.size());
    DeviceBuffer<unsigned int> dFitness(static_cast<size_t>(rows) * runs * OBJ);
    DeviceBuffer<int> dFlow(instance.flow.size());
    DeviceBuffer<int> dDist(instance.dist.size());
    DeviceBuffer<int> dTypes(runs);
    dGenes.copyFromHost(genes);
    dFlow.copyFromHost(instance.flow);
    dDist.copyFromHost(instance.dist);
    dTypes.copyFromHost(types);
    launchGreedy2Opt<OBJ>(dGenes.get(), dFitness.get(), dFlow.get(), dDist.get(), dTypes.get(), population, n, runs);
    const std::vector<short> out = dGenes.toHost();
    const std::vector<unsigned int> fitness = dFitness.toHost();

    for (int r = 0; r < runs; r++) {
        for (int c = 0; c < rows; c++) {
            const size_t row = static_cast<size_t>(r) * rows + c;
            const std::vector<short> before(genes.begin() + row * n, genes.begin() + (row + 1) * n);
            const std::vector<short> after(out.begin() + row * n, out.begin() + (row + 1) * n);
            if (c < population) {
                EXPECT(before == after, "greedy modified a survivor row");
                continue;
            }
            EXPECT(after == cpuGreedy(instance, before, types[r]), "greedy n=%d OBJ=%d type %d: differs from CPU",
                   n, OBJ, types[r]);
            for (int o = 0; o < OBJ; o++) {
                EXPECT(cost(instance, after.data(), o) == fitness[row * OBJ + o], "greedy fitness mismatch");
            }
        }
    }
}

template <int OBJ>
void testReproduce(int n, int population, int runs) {
    std::mt19937 rng(5 + n);
    const int rows = 2 * population;
    const std::vector<short> genes = randomPermutations(rows * runs, n, rng);
    std::vector<unsigned int> fitness(static_cast<size_t>(rows) * runs * OBJ);
    for (unsigned int& f : fitness) f = rng();
    std::vector<short> index(static_cast<size_t>(runs) * population);
    std::vector<short> rank(index.size());
    std::vector<float> crowding(index.size());
    for (int r = 0; r < runs; r++) {
        std::vector<short> ids(rows);
        std::iota(ids.begin(), ids.end(), static_cast<short>(0));
        std::shuffle(ids.begin(), ids.end(), rng);
        for (int q = 0; q < population; q++) {
            index[static_cast<size_t>(r) * population + q] = ids[q];
            rank[static_cast<size_t>(r) * population + q] = static_cast<short>(1 + q / 8);
            crowding[static_cast<size_t>(r) * population + q] = static_cast<float>(rng() % 100);
        }
    }
    DeviceBuffer<short> dGenes(genes.size()), dNext(genes.size());
    DeviceBuffer<unsigned int> dFitness(fitness.size()), dNextFitness(fitness.size());
    DeviceBuffer<short> dIndex(index.size()), dRank(rank.size());
    DeviceBuffer<float> dCrowding(crowding.size());
    DeviceBuffer<int> dTypes(runs);
    DeviceBuffer<RngState> dRng(static_cast<size_t>(rows) * runs);
    dGenes.copyFromHost(genes);
    dFitness.copyFromHost(fitness);
    dIndex.copyFromHost(index);
    dRank.copyFromHost(rank);
    dCrowding.copyFromHost(crowding);
    launchRngInit(dRng.get(), rows * runs, 1234);
    launchReproduce<OBJ>(dGenes.get(), dFitness.get(), dNext.get(), dNextFitness.get(), dIndex.get(), dRank.get(),
                         dCrowding.get(), dRng.get(), dTypes.get(), population, n, runs);
    const std::vector<short> next = dNext.toHost();
    const std::vector<unsigned int> nextFitness = dNextFitness.toHost();
    const std::vector<int> types = dTypes.toHost();

    for (int r = 0; r < runs; r++) {
        EXPECT(types[r] >= 0 && types[r] <= OBJ, "greedy type out of range");
        for (int q = 0; q < rows; q++) {
            const size_t row = static_cast<size_t>(r) * rows + q;
            std::vector<short> p(next.begin() + row * n, next.begin() + (row + 1) * n);
            if (q < population) {
                const size_t source = static_cast<size_t>(r) * rows + index[static_cast<size_t>(r) * population + q];
                EXPECT(std::equal(p.begin(), p.end(), genes.begin() + source * n), "survivor genes not copied");
                for (int o = 0; o < OBJ; o++) {
                    EXPECT(nextFitness[row * OBJ + o] == fitness[source * OBJ + o], "survivor fitness not copied");
                }
            }
            std::sort(p.begin(), p.end());
            for (int k = 0; k < n; k++) {
                EXPECT(p[k] == k, "reproduce produced an invalid permutation");
            }
        }
    }
}

void testInitPopulation(int n, int rows, int runs) {
    DeviceBuffer<RngState> dRng(static_cast<size_t>(rows) * runs);
    DeviceBuffer<short> dGenes(static_cast<size_t>(rows) * runs * n);
    launchRngInit(dRng.get(), rows * runs, 99);
    launchInitPopulation(dRng.get(), dGenes.get(), rows, n, runs);
    const std::vector<short> genes = dGenes.toHost();
    int identity = 0;
    for (int c = 0; c < rows * runs; c++) {
        std::vector<short> p(genes.begin() + static_cast<size_t>(c) * n, genes.begin() + static_cast<size_t>(c + 1) * n);
        bool isIdentity = true;
        for (int k = 0; k < n; k++) isIdentity &= p[k] == k;
        identity += isIdentity;
        std::sort(p.begin(), p.end());
        for (int k = 0; k < n; k++) {
            EXPECT(p[k] == k, "initial population: invalid permutation");
        }
    }
    EXPECT(identity == 0, "initial population not shuffled");
}

// Every instance file loads, and its header matches the file name (KC<n>-<k>fl-...).
void testLoadAllInstances(const std::string& data) {
    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(data)) {
        if (entry.path().extension() != ".dat") continue;
        const Instance instance = loadInstance(entry.path().string());
        const size_t dash = instance.name.find('-');
        const int n = std::stoi(instance.name.substr(2, dash - 2));
        const int objectives = instance.name[dash + 1] - '0';
        EXPECT(instance.n == n && instance.objectives == objectives, "%s: header n=%d k=%d",
               instance.name.c_str(), instance.n, instance.objectives);
        count++;
    }
    EXPECT(count > 0, "no .dat files found in %s", data.c_str());
}

// The published Pareto optimal solutions (.PO: 1-based permutation followed by the fitness values)
// must have exactly their published fitness, on the CPU reference and on the GPU kernel.
void testParetoOptimalFiles(const std::string& data) {
    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(data)) {
        if (entry.path().extension() != ".PO") continue;
        std::filesystem::path datPath = entry.path();
        const Instance instance = loadInstance(datPath.replace_extension(".dat").string());
        const int n = instance.n;
        const int k = instance.objectives;
        std::ifstream in(entry.path());
        std::vector<short> genes;
        std::vector<long long> published;
        long long value = 0;
        std::vector<long long> line;
        while (in >> value) {
            line.push_back(value);
            if (static_cast<int>(line.size()) == n + k) {
                for (int i = 0; i < n; i++) genes.push_back(static_cast<short>(line[i] - 1));
                for (int o = 0; o < k; o++) published.push_back(line[n + o]);
                line.clear();
            }
        }
        const int rows = static_cast<int>(genes.size()) / n;
        EXPECT(rows > 0 && line.empty(), "%s: unexpected format", instance.name.c_str());
        DeviceBuffer<short> dGenes(genes.size());
        DeviceBuffer<unsigned int> dFitness(static_cast<size_t>(rows) * k);
        DeviceBuffer<int> dFlow(instance.flow.size());
        DeviceBuffer<int> dDist(instance.dist.size());
        dGenes.copyFromHost(genes);
        dFlow.copyFromHost(instance.flow);
        dDist.copyFromHost(instance.dist);
        if (k == 2) {
            launchFitness<2>(dGenes.get(), dFitness.get(), dFlow.get(), dDist.get(), rows, n, 1);
        } else {
            launchFitness<3>(dGenes.get(), dFitness.get(), dFlow.get(), dDist.get(), rows, n, 1);
        }
        const std::vector<unsigned int> fitness = dFitness.toHost();
        for (int r = 0; r < rows; r++) {
            for (int o = 0; o < k; o++) {
                const long long expected = published[static_cast<size_t>(r) * k + o];
                EXPECT(cost(instance, &genes[static_cast<size_t>(r) * n], o) == expected, "%s CPU cost", instance.name.c_str());
                EXPECT(fitness[static_cast<size_t>(r) * k + o] == expected, "%s GPU fitness", instance.name.c_str());
            }
        }
        count += rows;
    }
    EXPECT(count > 0, "no .PO files found in %s", data.c_str());
}

template <typename F>
void run(const char* name, F test) {
    const int before = failures;
    test();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("%-58s %s\n", name, failures == before ? "ok" : "FAILED");
}

} // namespace

int main(int argc, char** argv) {
    const std::string data = argc > 1 ? argv[1] : "mQAPData";
    const Instance kc10 = loadInstance(data + "/KC10-2fl-1rl.dat");
    const Instance kc20 = loadInstance(data + "/KC20-2fl-1rl.dat");
    const Instance kc30 = loadInstance(data + "/KC30-3fl-1rl.dat");

    run("all .dat instances load with the expected header", [&] { testLoadAllInstances(data); });
    run("published Pareto optimal solutions (.PO) fitness", [&] { testParetoOptimalFiles(data); });
    run("fitness == Trace(F*X*DT*XT)  KC10-2fl-1rl (2 runs)", [&] { testFitness<2>(kc10, 128, 2); });
    run("fitness == Trace(F*X*DT*XT)  KC20-2fl-1rl", [&] { testFitness<2>(kc20, 128, 1); });
    run("fitness == Trace(F*X*DT*XT)  KC30-3fl-1rl (3 runs)", [&] { testFitness<3>(kc30, 64, 3); });
    run("survival == CPU NSGA-II      P=16  OBJ=2", [] { testSurvival<2>(16, 3); });
    run("survival == CPU NSGA-II      P=64  OBJ=2", [] { testSurvival<2>(64, 3); });
    run("survival == CPU NSGA-II      P=64  OBJ=3", [] { testSurvival<3>(64, 3); });
    run("survival == CPU NSGA-II      P=256 OBJ=3", [] { testSurvival<3>(256, 2); });
    run("greedy 2-opt == CPU greedy   n=10 OBJ=2", [] { testGreedy<2>(10, 64, 3); });
    run("greedy 2-opt == CPU greedy   n=30 OBJ=3", [] { testGreedy<3>(30, 32, 4); });
    run("greedy 2-opt == CPU greedy   n=60 OBJ=3 (>48 KB shared)", [] { testGreedy<3>(60, 16, 1); });
    run("reproduce: survivors copied, valid children", [] { testReproduce<3>(30, 64, 3); });
    run("initial population: valid shuffled permutations", [] { testInitPopulation(30, 128, 4); });

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
