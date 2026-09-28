#pragma once

#include "core/cuda_context.cuh"
#include "core/device_buffer.hpp"

#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cugpt::core
{
    using Shape = std::vector<size_t>;

    template <typename T>
    class AbstractTensor
    {
    public:
        AbstractTensor<T>() = default;
        explicit AbstractTensor<T>(const Shape &shape) { resize(shape); }
        AbstractTensor<T>(std::initializer_list<size_t> shape) : AbstractTensor<T>(Shape(shape)) {}

        void resize(const Shape &shape);

        const Shape &shape() const noexcept { return shape_; }
        std::size_t numel() const noexcept { return numel_; }
        std::size_t bytes() const noexcept { return numel_ * sizeof(T); }

        T *data() noexcept { return static_cast<T *>(buffer_.data()); }
        const T *data() const noexcept { return static_cast<const T *>(buffer_.data()); }

        void copyFromHost(const T *src, const CudaContext &ctx);
        void copyToHost(T *dst, const CudaContext &ctx) const;

    protected:
        Shape shape_;
        std::size_t numel_ = 0;
        DeviceBuffer buffer_;
    };

    template <typename T>
    void AbstractTensor<T>::resize(const Shape &shape)
    {
        const auto n = checkedNumel(shape);
        shape_ = shape;
        numel_ = n;
        buffer_.allocate(bytes());
    }

    template <typename T>
    void AbstractTensor<T>::copyFromHost(const T *src, const CudaContext &ctx)
    {
        CUDA_CHECK(cudaMemcpyAsync(data(), src, bytes(), cudaMemcpyHostToDevice, ctx.stream()));
    }

    // Device-to-host copies are followed by a stream synchronize because the caller
    // expects the host buffer to be complete when this function returns
    template <typename T>
    void AbstractTensor<T>::copyToHost(T *dst, const CudaContext &ctx) const
    {
        CUDA_CHECK(cudaMemcpyAsync(dst, data(), bytes(), cudaMemcpyDeviceToHost, ctx.stream()));
        ctx.synchronize();
    }

    class Tensor : public AbstractTensor<float>
    {
    public:
        Tensor() = default;
        explicit Tensor(const Shape &shape) : AbstractTensor<float>(shape) {}
        Tensor(std::initializer_list<size_t> shape) : AbstractTensor<float>(shape) {}

        void zero(const CudaContext &ctx);
        void fill(float value, const CudaContext &ctx);
    };

    class IntTensor : public AbstractTensor<int32_t>
    {
    public:
        IntTensor() = default;
        explicit IntTensor(const Shape &shape) : AbstractTensor<int32_t>(shape) {}
        IntTensor(std::initializer_list<size_t> shape) : AbstractTensor<int32_t>(shape) {}
    };

    std::size_t checkedNumel(const Shape &shape);

    template <typename T>
    void expectSameNumel(const AbstractTensor<T> &a, const AbstractTensor<T> &b, const char *message);
} // namespace cugpt::core
