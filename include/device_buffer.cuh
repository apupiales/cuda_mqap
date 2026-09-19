/*
 * device_buffer.cuh
 *
 * RAII owner of a device allocation: allocated once, released automatically.
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
#include <vector>

#include "cuda_check.cuh"

namespace mqap {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* get() const { return ptr_; }
    size_t size() const { return count_; }

    void copyFromHost(const T* host) {
        CUDA_CHECK(cudaMemcpy(ptr_, host, count_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copyFromHost(const std::vector<T>& host) {
        copyFromHost(host.data());
    }

    void copyToHost(T* host) const {
        CUDA_CHECK(cudaMemcpy(host, ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    std::vector<T> toHost() const {
        std::vector<T> host(count_);
        copyToHost(host.data());
        return host;
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

} // namespace mqap
