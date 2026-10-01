#pragma once

#include "core/module.hpp"

namespace cugpt::nn
{
    using namespace core;

    class LayerNorm final : public Module
    {
    public:
        LayerNorm(CudaContext &ctx, std::size_t in_features, float eps = 1.0e-5f);

        void forward(const Tensor &input, Tensor &output);
        void backward(const Tensor &grad_output, Tensor &grad_input);

        Parameter gamma;
        Parameter beta;

        std::size_t inputFeatures() const noexcept { return input_features_; }

    private:
        CudaContext *ctx_;
        std::size_t input_features_;
        float eps_;
        Tensor mean_;
        Tensor inv_std_;
        Tensor xhat_;
    };
} // namespace cugpt::nn
