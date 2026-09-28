#pragma once

#include "core/tensor.cuh"

namespace cugpt::core
{
    struct Parameter
    {
        Tensor data;
        Tensor grad;

        Parameter() = default;
        explicit Parameter(const Shape &shape) : data(shape), grad(shape) {}

        void zeroGrad(const CudaContext &ctx) { grad.zero(ctx); }
    };
} // namespace cugpt::core