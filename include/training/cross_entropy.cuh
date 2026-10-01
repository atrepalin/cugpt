#pragma once

#include "core/tensor.cuh"

namespace cugpt::training
{
    using namespace core;

    class CrossEntropyLoss
    {
    public:
        CrossEntropyLoss(CudaContext &ctx, int32_t ignore_index = 0);

        float forwardBackward(const Tensor &logits, const IntTensor &targets, Tensor &grad_logits);

    private:
        CudaContext *ctx_;
        int32_t ignore_index_;
    };
} // namespace cugpt::training
