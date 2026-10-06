/*
 * gate.h
 *
 * The gate of the greedy 2-opt, shared by the CUDA kernels, the CPU solver and the host-side counting
 * of the work done. It has no CUDA dependency, so plain C++ translation units can include it.
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
#pragma once

#if defined(__CUDACC__)
#define MQAP_HOST_DEVICE __host__ __device__
#else
#define MQAP_HOST_DEVICE
#endif

namespace mqap {

// Whether one offspring gets the local search in one generation, from the rate and the period of the run.
// Stateless on purpose: it takes nothing from the random streams of the operators, so a run with a rate of
// 1.0f is identical to one made before the knobs existed, and the host side of the tests can mirror the
// decision. Every lane of a warp computes the same value from the same arguments.
MQAP_HOST_DEVICE inline bool greedyApplies(unsigned long long seed, int run, int offspring,
                                           int generation, float rate, int period) {
    if (period > 1 && generation % period != 0) {
        return false;
    }
    if (rate >= 1.0f) {
        return true;
    }
    unsigned long long x = seed + 0x9E3779B97F4A7C15ULL * (static_cast<unsigned long long>(generation) + 1);
    x ^= 0xBF58476D1CE4E5B9ULL * (static_cast<unsigned long long>(run) + 1);
    x += 0x94D049BB133111EBULL * (static_cast<unsigned long long>(offspring) + 1);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    // The top 24 bits as a float in [0, 1).
    return static_cast<float>(x >> 40) * (1.0f / 16777216.0f) < rate;
}

} // namespace mqap
