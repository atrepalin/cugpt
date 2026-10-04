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
            : ptr_(other.ptr_), size_(other.size_), owning_(other.owning_)
        {
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.owning_ = false;
        }

        DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
        {
            if (this != &other)
            {
                release();
                ptr_ = other.ptr_;
                size_ = other.size_;
                owning_ = other.owning_;
                other.ptr_ = nullptr;
                other.size_ = 0;
                other.owning_ = false;
            }

            return *this;
        }

        void allocate(std::size_t bytes)
        {
            if (owning_ && bytes == size_ && ptr_ != nullptr)
            {
                return;
            }

            release();

            if (bytes != 0)
            {
                CUDA_CHECK(cudaMalloc(&ptr_, bytes));
                size_ = bytes;
                owning_ = true;
            }
        }

        // Borrow an existing device allocation without taking ownership. The
        // caller guarantees that the allocation stays alive for the lifetime
        // of this buffer
        void view(void *ptr, std::size_t bytes)
        {
            release();
            ptr_ = ptr;
            size_ = bytes;
            owning_ = false;
        }

        void release() noexcept
        {
            if (ptr_ != nullptr && owning_)
            {
                cudaFree(ptr_);
            }

            ptr_ = nullptr;
            size_ = 0;
            owning_ = false;
        }

        void *data() noexcept { return ptr_; }
        const void *data() const noexcept { return ptr_; }
        std::size_t bytes() const noexcept { return size_; }
        bool owns() const noexcept { return owning_; }
        explicit operator bool() const noexcept { return ptr_ != nullptr; }

    private:
        void *ptr_ = nullptr;
        std::size_t size_ = 0;
        bool owning_ = false;
    };
} // namespace cugpt::core