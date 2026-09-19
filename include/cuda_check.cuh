/*
 * cuda_check.cuh
 *
 * Error checking for CUDA runtime calls and kernel launches.
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

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// Aborts with file/line information when a CUDA runtime call fails.
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        const cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "[CUDA] %s (%s) at %s:%d\n  -> %s\n",            \
                         cudaGetErrorName(err_), cudaGetErrorString(err_),        \
                         __FILE__, __LINE__, #call);                              \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

// Release: only checks the launch configuration (non-blocking).
// Debug (MQAP_SYNC_CHECK defined): also synchronizes so an execution error is reported
// right after the kernel that caused it.
#ifdef MQAP_SYNC_CHECK
#define CUDA_CHECK_KERNEL()                          \
    do {                                             \
        CUDA_CHECK(cudaGetLastError());              \
        CUDA_CHECK(cudaDeviceSynchronize());         \
    } while (0)
#else
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#endif
