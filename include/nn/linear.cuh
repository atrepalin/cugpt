#pragma once

#include "core/module.hpp"

#include <memory>

namespace cugpt::nn
{
    using namespace core;

    class Linear final : public Module
    {
    public:
        Linear(CudaContext &ctx, size_t in_features, size_t out_features, bool include_bias = true);

        void forward(const Tensor &input, Tensor &output);
        void backward(const Tensor &grad_output, Tensor &grad_input);

        Parameter weights;
        Parameter *bias() noexcept { return bias_.get(); }

        size_t inputFeatures() const noexcept { return input_features_; }
        size_t outputFeatures() const noexcept { return output_features_; }

    private:
        CudaContext *ctx_;
        size_t input_features_;
        size_t output_features_;
        std::unique_ptr<Parameter> bias_;
        const Tensor *cached_input_ = nullptr;
    };
} // namespace cugpt::nn