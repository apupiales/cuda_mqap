/*
 * instance.h
 *
 * mQAP instance loaded at runtime from the mQAPData/*.dat files
 * (http://www.cs.bham.ac.uk/~jdk/mQAP/).
 */
#pragma once

#include <string>
#include <vector>

namespace mqap {

struct Instance {
    std::string name;       // File name without extension, e.g. "KC10-2fl-1rl".
    int n = 0;              // Number of facilities/locations.
    int objectives = 0;     // Number of flow matrices.

    // n*n. dist[a*n + b] is the distance term between locations a and b used by cost().
    // It holds the transpose of the matrix in the .dat file, which reproduces the original
    // Trace(Fk * X * DT * XT) formulation (the published instances are symmetric anyway).
    std::vector<int> dist;

    // objectives*n*n. flow[k*n*n + i*n + j] is the flow between facilities i and j for objective k.
    std::vector<int> flow;
};

// Parses a .dat file. Throws std::runtime_error with a descriptive message on failure,
// including instances whose costs could overflow the 32-bit fitness values.
Instance loadInstance(const std::string& path);

// Reference CPU cost of permutation p (facility i is placed at location p[i]) for one objective:
//   sum_i sum_j flow[k][i][j] * dist[p[i]][p[j]]
long long cost(const Instance& instance, const short* permutation, int objective);

} // namespace mqap
