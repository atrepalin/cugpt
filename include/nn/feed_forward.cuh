#pragma once

#include "nn/linear.cuh"
#include "nn/relu.cuh"

namespace cugpt::nn
{
    using namespace core;

    class FeedForwardNetwork final : public Module
    {
    public:
        FeedForwardNetwork(CudaContext &ctx, std::size_t in_features, std::size_t hidden_features, std::size_t out_features);

        void forward(const Tensor &input, Tensor &output);
        void backward(const Tensor &grad_output, Tensor &grad_input);

        Linear &firstLinear() noexcept { return linear1_; }
        Linear &secondLinear() noexcept { return linear2_; }
        const Linear &firstLinear() const noexcept { return linear1_; }
        const Linear &secondLinear() const noexcept { return linear2_; }

        std::size_t inputFeatures() const noexcept { return input_features_; }
        std::size_t hiddenFeatures() const noexcept { return hidden_features_; }
        std::size_t outputFeatures() const noexcept { return output_features_; }

    private:
        CudaContext *ctx_;
        std::size_t input_features_;
        std::size_t hidden_features_;
        std::size_t output_features_;

        Linear linear1_;
        ReLU relu_;
        Linear linear2_;

        Tensor hidden_pre_activation_;
        Tensor hidden_;
    };
} // namespace cugpt::nn