#pragma once

#include "core/cuda_utils.cuh"

#include <cstddef>
#include <utility>

namespace cugpt::core
{
    // RAII wrapper for a raw cudaMalloc allocation
    // Copying is disabled because the pointer represents unique ownership; moving transfers that ownership
    class DeviceBuffer
    {
    public:
        DeviceBuffer() = default;
        explicit DeviceBuffer(std::size_t bytes) { allocate(bytes); }
        ~DeviceBuffer() { release(); }

        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;

        DeviceBuffer(DeviceBuffer &&other) noexcept
            : ptr_(other.ptr_), size_(other.size_)
        {
            other.ptr_ = nullptr;
            other.size_ = 0;
        }

        DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
        {
            if (this != &other)
            {
                release();
                ptr_ = other.ptr_;
                size_ = other.size_;
                other.ptr_ = nullptr;
                other.size_ = 0;
            }

            return *this;
        }

        void allocate(std::size_t bytes)
        {
            if (bytes == size_ && ptr_ != nullptr)
            {
                return;
            }

            release();

            if (bytes != 0)
            {
                CUDA_CHECK(cudaMalloc(&ptr_, bytes));
                size_ = bytes;
            }
        }

        void release() noexcept
        {
            if (ptr_ != nullptr)
            {
                cudaFree(ptr_);
                ptr_ = nullptr;
                size_ = 0;
            }
        }

        void *data() noexcept { return ptr_; }
        const void *data() const noexcept { return ptr_; }
        std::size_t bytes() const noexcept { return size_; }
        explicit operator bool() const noexcept { return ptr_ != nullptr; }

    private:
        void *ptr_ = nullptr;
        std::size_t size_ = 0;
    };
} // namespace cugpt::core