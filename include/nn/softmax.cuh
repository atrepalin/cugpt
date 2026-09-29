#pragma once

#include "core/module.hpp"

namespace cugpt::nn
{
    using namespace core;

    class Softmax : public Module
    {
    public:
        Softmax(CudaContext &ctx, bool causal_mask = true);

        void forward(const Tensor &input, Tensor &output);
        void backward(const Tensor &grad_output, Tensor &grad_input);

    private:
        CudaContext *ctx_;
        bool causal_mask_;
        const Tensor *cached_input_ = nullptr;
    };
} // namespace cugpt::nn