/*
 * instance.cpp
 *
 * Parser for the mQAP .dat format:
 *   facilities = 10 objectives = 2 max_distances = ... (header line)
 *   <n x n distance matrix>
 *   <objectives x (n x n) flow matrices>
 */
#include "instance.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "config.h"

namespace mqap {

namespace {

std::string fileStem(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const size_t dot = name.find_last_of('.');
    return (dot == std::string::npos) ? name : name.substr(0, dot);
}

} // namespace

Instance loadInstance(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open instance file: " + path);
    }

    Instance instance;
    instance.name = fileStem(path);

    // Header line, in either of the two published formats:
    //   "facilities = 10 objectives = 2 max_distances = 155 ..."
    //   "facilities: 10 objectives: 2 max_distances: 100 ..."
    std::string header;
    std::getline(in, header);
    std::replace(header.begin(), header.end(), ':', ' ');
    std::replace(header.begin(), header.end(), '=', ' ');
    std::istringstream headerTokens(header);
    std::string token;
    while (headerTokens >> token) {
        if (token == "facilities") {
            headerTokens >> instance.n;
        } else if (token == "objectives") {
            headerTokens >> instance.objectives;
        }
    }

    const int n = instance.n;
    if (n < 2 || n > kMaxFacilities) {
        throw std::runtime_error(path + ": facilities must be in [2, " + std::to_string(kMaxFacilities) + "]");
    }
    if (instance.objectives < kMinObjectives || instance.objectives > kMaxObjectives) {
        throw std::runtime_error(path + ": only 2 and 3 objective instances are supported");
    }

    std::vector<int> fileDist(n * n);
    for (int& value : fileDist) {
        in >> value;
    }
    instance.flow.resize(static_cast<size_t>(instance.objectives) * n * n);
    for (int& value : instance.flow) {
        in >> value;
    }
    if (!in) {
        throw std::runtime_error(path + ": unexpected end of file or invalid number");
    }

    instance.dist.resize(n * n);
    for (int a = 0; a < n; a++) {
        for (int b = 0; b < n; b++) {
            instance.dist[a * n + b] = fileDist[b * n + a];
        }
    }

    // Fitness values are stored as 32-bit unsigned integers on the device.
    const long long maxDist = *std::max_element(instance.dist.begin(), instance.dist.end());
    if (*std::min_element(instance.dist.begin(), instance.dist.end()) < 0 ||
        *std::min_element(instance.flow.begin(), instance.flow.end()) < 0) {
        throw std::runtime_error(path + ": negative distances or flows are not supported");
    }
    for (int k = 0; k < instance.objectives; k++) {
        long long flowSum = 0;
        for (int i = 0; i < n * n; i++) {
            flowSum += instance.flow[static_cast<size_t>(k) * n * n + i];
        }
        if (flowSum * maxDist > static_cast<long long>(UINT32_MAX)) {
            throw std::runtime_error(path + ": costs may overflow 32-bit fitness values");
        }
    }

    return instance;
}

long long cost(const Instance& instance, const short* permutation, int objective) {
    const int n = instance.n;
    const int* flow = instance.flow.data() + static_cast<size_t>(objective) * n * n;
    long long total = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            total += static_cast<long long>(flow[i * n + j]) * instance.dist[permutation[i] * n + permutation[j]];
        }
    }
    return total;
}

} // namespace mqap
