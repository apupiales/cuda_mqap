/*
 * main.cpp
 *
 *  Started on: May 19, 2019
 *      Author: Andres Pupiales Arevalo
 *      apupiales@gmail.com
 *      https://github.com/apupiales
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * The GNU General Public License is available at:
 *   http://www.gnu.org/copyleft/gpl.html
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include "instance.h"
#include "solver.h"

namespace {

struct Arguments {
    std::string instancePath;
    std::string outputPath;
    mqap::SolverOptions options;
    bool verify = false;
    bool quiet = false;
};

void printUsage(const char* program) {
    std::printf(
        "Usage: %s <instance.dat> [options]\n"
        "  --population P   population size, power of two in [16, 256] (default 64)\n"
        "  --iterations N   generations (default 70)\n"
        "  --runs R         independent runs executed concurrently (default 1)\n"
        "  --seed S         random seed (default: random, printed in the output)\n"
        "  --output FILE    result file, appended (default result_<instance>_nsga2_greedy_2opt.txt)\n"
        "  --verify         check the final populations on the CPU\n"
        "  --quiet          do not print the final solutions\n",
        program);
}

bool parseArguments(int argc, char** argv, Arguments& args) {
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--population" && hasValue) {
            args.options.population = std::atoi(argv[++i]);
        } else if (arg == "--iterations" && hasValue) {
            args.options.iterations = std::atoi(argv[++i]);
        } else if (arg == "--runs" && hasValue) {
            args.options.runs = std::atoi(argv[++i]);
        } else if (arg == "--seed" && hasValue) {
            args.options.seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--output" && hasValue) {
            args.outputPath = argv[++i];
        } else if (arg == "--verify") {
            args.verify = true;
        } else if (arg == "--quiet") {
            args.quiet = true;
        } else if (!arg.empty() && arg[0] != '-' && args.instancePath.empty()) {
            args.instancePath = arg;
        } else {
            return false;
        }
    }
    return !args.instancePath.empty();
}

bool dominates(const mqap::Solution& a, const mqap::Solution& b) {
    bool strictlyBetter = false;
    for (size_t o = 0; o < a.fitness.size(); o++) {
        if (a.fitness[o] > b.fitness[o]) {
            return false;
        }
        strictlyBetter |= a.fitness[o] < b.fitness[o];
    }
    return strictlyBetter;
}

// Independent CPU check of a run: valid permutations, exact fitness values and consistent ranks.
bool verifyRun(const mqap::Instance& instance, const mqap::RunResult& result, int run) {
    bool ok = true;
    for (const mqap::Solution& solution : result.population) {
        std::vector<int> seen(instance.n, 0);
        for (short location : solution.permutation) {
            if (location < 0 || location >= instance.n || seen[location]++) {
                std::fprintf(stderr, "verify run %d: invalid permutation\n", run);
                return false;
            }
        }
        for (int o = 0; o < instance.objectives; o++) {
            const long long expected = mqap::cost(instance, solution.permutation.data(), o);
            if (expected != static_cast<long long>(solution.fitness[o])) {
                std::fprintf(stderr, "verify run %d: fitness %u != %lld (objective %d)\n",
                             run, solution.fitness[o], expected, o);
                ok = false;
            }
        }
    }
    for (const mqap::Solution& a : result.population) {
        bool dominated = false;
        for (const mqap::Solution& b : result.population) {
            dominated |= dominates(b, a);
        }
        // Survivors are chosen by rank, so every dominated survivor has a dominator among them.
        if (dominated == (a.rank == 1)) {
            std::fprintf(stderr, "verify run %d: rank %d inconsistent with dominance\n", run, a.rank);
            ok = false;
        }
    }
    return ok;
}

// Same format as the original program: { 'permutation': [f1, f2], ... },
void writeRun(std::ostream& out, const mqap::RunResult& result) {
    out << "{\n";
    for (const mqap::Solution& solution : result.paretoFront) {
        out << "'";
        for (short location : solution.permutation) {
            out << location;
        }
        out << "': [";
        for (size_t o = 0; o < solution.fitness.size(); o++) {
            out << solution.fitness[o] << (o + 1 < solution.fitness.size() ? ", " : "");
        }
        out << "],\n";
    }
    out << "},\n";
}

} // namespace

int main(int argc, char** argv) {
    Arguments args;
    if (!parseArguments(argc, argv, args)) {
        printUsage(argv[0]);
        return 2;
    }

    try {
        const mqap::Instance instance = mqap::loadInstance(args.instancePath);
        if (args.options.seed == 0) {
            args.options.seed = (static_cast<unsigned long long>(std::random_device{}()) << 32) ^ std::random_device{}();
        }
        if (args.outputPath.empty()) {
            args.outputPath = "result_" + instance.name + "_nsga2_greedy_2opt.txt";
        }
        std::printf("Instance %s: n = %d, objectives = %d | population = %d, iterations = %d, runs = %d, seed = %llu\n",
                    instance.name.c_str(), instance.n, instance.objectives, args.options.population,
                    args.options.iterations, args.options.runs, args.options.seed);

        const auto begin = std::chrono::steady_clock::now();
        mqap::SolveStats stats;
        const std::vector<mqap::RunResult> results = mqap::solve(instance, args.options, &stats);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

        std::ofstream file(args.outputPath, std::ios::app);
        if (!file) {
            std::fprintf(stderr, "Error opening %s\n", args.outputPath.c_str());
            return 1;
        }
        for (const mqap::RunResult& result : results) {
            writeRun(file, result);
        }
        file.close();

        if (!args.quiet) {
            for (size_t run = 0; run < results.size(); run++) {
                std::printf("\nFINAL SOLUTION (run %zu, %zu non-dominated)\n", run, results[run].paretoFront.size());
                for (const mqap::Solution& solution : results[run].paretoFront) {
                    for (short location : solution.permutation) {
                        std::printf("%d ", location);
                    }
                    for (unsigned int value : solution.fitness) {
                        std::printf("%u ", value);
                    }
                    std::printf("\n");
                }
            }
        }

        int exitCode = 0;
        if (args.verify) {
            bool ok = true;
            for (size_t run = 0; run < results.size(); run++) {
                ok &= verifyRun(instance, results[run], static_cast<int>(run));
            }
            std::printf("\nVerification: %s\n", ok ? "OK" : "FAILED");
            exitCode = ok ? 0 : 1;
        }

        std::printf("\nResults appended to %s\n", args.outputPath.c_str());
        std::printf("Time Spent: %f s (GPU %.3f ms)\n", seconds, stats.gpuMilliseconds);
        return exitCode;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
}
