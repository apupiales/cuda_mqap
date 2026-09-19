/*
 * device_buffer.cuh
 *
 * RAII owner of a device allocation: allocated once, released automatically.
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
