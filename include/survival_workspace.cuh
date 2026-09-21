/*
 * survival_workspace.cuh
 *
 * Device buffers of the multi-block NSGA-II survival, used when the population does not fit in a
 * single block (P > kSingleBlockMaxPopulation). Allocated once per solve; every buffer is O(N)
 * per run (N = 2P), so the VRAM needed grows linearly with the population.
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

#include <cstddef>

#include "device_buffer.cuh"

namespace mqap {

class SurvivalWorkspace {
public:
    // forceMultiblock: use the multi-block path even for small populations (tests).
    SurvivalWorkspace(int population, int runs, bool forceMultiblock = false);

    bool multiblock() const { return multiblock_; }

    // Bytes of device memory used by this workspace.
    size_t deviceBytes() const;

    int population = 0;
    int runs = 0;
    int total = 0;                      // N = 2P

    DeviceBuffer<int> dominators;       // [R][N] number of remaining dominators
    DeviceBuffer<int> rank;             // [R][N]
    DeviceBuffer<float> crowding;       // [R][N]
    DeviceBuffer<int> frontList;        // [R][N] members of the current front
    DeviceBuffer<int> frontSize;        // [2][R] double buffered front sizes
    DeviceBuffer<int> flags;            // [2]    "some individual is still unranked"
    DeviceBuffer<unsigned long long> keysIn, keysOut;   // [R][N] sort keys
    DeviceBuffer<int> valuesIn, valuesOut;              // [R][N] local indices
    DeviceBuffer<unsigned int> minFitness, maxFitness;  // [R]
    DeviceBuffer<int> offsets;          // [R + 1] segment offsets for the sorts
    DeviceBuffer<unsigned char> sortTemp;
    size_t sortTempBytes = 0;

private:
    bool multiblock_ = false;
};

} // namespace mqap
