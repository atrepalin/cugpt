#pragma once

#include "core/module.hpp"

namespace cugpt::nn
{
    using namespace core;

    class ReLU final : public Module
    {
    public:
        ReLU(CudaContext &ctx);

        void forward(const Tensor &input, Tensor &output);
        void backward(const Tensor &grad_output, Tensor &grad_input);

    private:
        CudaContext *ctx_;
        const Tensor *cached_input_ = nullptr;
    };
} // namespace cugpt::nn
