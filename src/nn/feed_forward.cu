#include "nn/feed_forward.cuh"

#include <stdexcept>

namespace cugpt::nn
{
    using namespace core;

    FeedForwardNetwork::FeedForwardNetwork(
        CudaContext &ctx,
        std::size_t in_features,
        std::size_t hidden_features,
        std::size_t out_features)
        : ctx_(&ctx),
          input_features_(in_features),
          hidden_features_(hidden_features),
          output_features_(out_features),
          linear1_(ctx, in_features, hidden_features, true),
          relu_(ctx),
          linear2_(ctx, hidden_features, out_features, true)
    {
        if (in_features <= 0 || hidden_features <= 0 || out_features <= 0)
        {
            throw std::invalid_argument("FeedForwardNetwork dimensions must be positive");
        }

        registerModule("linear1", linear1_);
        registerModule("linear2", linear2_);
    }

    void FeedForwardNetwork::forward(const Tensor &input, Tensor &output)
    {
        if (input.shape().empty() || input.shape().back() != input_features_)
        {
            throw std::invalid_argument("FeedForwardNetwork::forward: last input dimension does not match input_features");
        }

        linear1_.forward(input, hidden_pre_activation_);
        relu_.forward(hidden_pre_activation_, hidden_);
        linear2_.forward(hidden_, output);
    }

    void FeedForwardNetwork::backward(const Tensor &grad_output, Tensor &grad_input)
    {
        if (grad_output.shape().empty() || grad_output.shape().back() != output_features_)
        {
            throw std::invalid_argument("FeedForwardNetwork::backward: grad_output last dimension mismatch");
        }

        Tensor grad_hidden;
        Tensor grad_hidden_pre_activation;

        linear2_.backward(grad_output, grad_hidden);
        relu_.backward(grad_hidden, grad_hidden_pre_activation);
        linear1_.backward(grad_hidden_pre_activation, grad_input);
    }
} // namespace cugpt::nn