/*
 * best_configuration.h
 *
 * The configuration measured best for each instance, which the program uses as its defaults.
 *
 * Every row is the cell of the grid of scripts/run_rate_grid.ps1 that covered most of the reference front
 * of its instance: the published optimum on KC10 and reference/v0.3 on the rest, over ten runs per cell.
 * Among the cells that tie, the cheapest one, which is the smaller population and then the smaller share
 * of the population given to the local search. The generation budget is the one the grid used, because a
 * configuration is only best for the budget it was measured with. See the README, "The best configuration
 * for each problem".
 *
 * Generated from results/grid/best.json; do not edit by hand. The program takes these values only for the
 * options the command line does not give, and --untuned ignores the table altogether.
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

#include <string>

namespace mqap {

struct InstanceConfiguration {
    const char* instance;   // File name of the instance without extension, as Instance::name holds it.
    int population;         // P.
    int iterations;         // Generations the configuration was measured with.
    float greedyRate;       // Fraction of the offspring the greedy 2-opt improves.
    int greedyPeriod;       // The local search runs on the generations multiple of this.
    float coverage;         // Share of the reference front it reached, per cent, for reference only.
};

// Measured on an RTX 2060 with the pair traversal of the original version (kGreedyFullPairs = true).
constexpr InstanceConfiguration kBestConfigurations[] = {
    {"KC10-2fl-1rl",   16384,  70,   0.5f, 1, 100.00f},
    {"KC10-2fl-1uni",   1024,  70,  0.25f, 1, 100.00f},
    {"KC10-2fl-2rl",    1024,  70,   0.1f, 1, 100.00f},
    {"KC10-2fl-2uni",    256,  70,   0.1f, 1, 100.00f},
    {"KC10-2fl-3rl",   16384,  70,   0.1f, 1, 100.00f},
    {"KC10-2fl-3uni",  65536,  70,   0.1f, 1, 100.00f},
    {"KC10-2fl-4rl",   16384,  70,   0.1f, 1, 100.00f},
    {"KC10-2fl-5rl",   16384,  70,   0.1f, 1, 100.00f},
    {"KC20-2fl-1rl",   65536, 300,  0.25f, 1,  96.60f},
    {"KC20-2fl-1uni",  65536, 300,   1.0f, 2,  98.03f},
    {"KC20-2fl-2rl",   65536, 300,  0.25f, 1,  58.87f},
    {"KC20-2fl-2uni",  65536, 300,   0.1f, 1, 100.00f},
    {"KC20-2fl-3rl",   65536, 300,  0.25f, 1,  59.67f},
    {"KC20-2fl-3uni",  65536, 300,   1.0f, 1,  73.54f},
    {"KC20-2fl-4rl",   65536, 300,   0.1f, 1,  48.38f},
    {"KC20-2fl-5rl",   65536, 300,  0.25f, 1,  63.45f},
    {"KC30-2fl-1rl",   65536, 300,   0.5f, 1,  45.30f},
    {"KC30-3fl-1rl",   65536, 300,   1.0f, 1,  14.68f},
    {"KC30-3fl-1uni",  65536, 300,   1.0f, 1,  14.64f},
    {"KC30-3fl-2rl",   65536, 300,   1.0f, 1,  24.33f},
    {"KC30-3fl-2uni",  65536, 300,   1.0f, 1,  42.99f},
    {"KC30-3fl-3rl",   65536, 300,   1.0f, 1,  28.20f},
    {"KC30-3fl-3uni",  65536, 300,   1.0f, 1,  22.82f},
};

// The row of an instance, or nullptr when the table has none for that name.
inline const InstanceConfiguration* bestConfigurationOf(const std::string& instance) {
    for (const InstanceConfiguration& row : kBestConfigurations) {
        if (instance == row.instance) {
            return &row;
        }
    }
    return nullptr;
}

} // namespace mqap
