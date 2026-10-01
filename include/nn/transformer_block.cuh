#pragma once

#include "nn/attention.cuh"
#include "nn/feed_forward.cuh"
#include "nn/layernorm.cuh"

namespace cugpt::nn
{
    using namespace core;

    class TransformerBlock final : public Module
    {
    public:
        TransformerBlock(CudaContext &ctx, std::size_t model_dim, std::size_t num_heads, std::size_t ffn_hidden, float residual_weight_scale = 1.0f);

        void forward(const Tensor &input, Tensor &output);
        void backward(const Tensor &grad_output, Tensor &grad_input);

        LayerNorm &firstNorm() noexcept { return norm1_; }
        LayerNorm &secondNorm() noexcept { return norm2_; }
        MultiHeadAttention &attention() noexcept { return attention_; }
        FeedForwardNetwork &feedForward() noexcept { return ffn_; }

        const LayerNorm &firstNorm() const noexcept { return norm1_; }
        const LayerNorm &secondNorm() const noexcept { return norm2_; }
        const MultiHeadAttention &attention() const noexcept { return attention_; }
        const FeedForwardNetwork &feedForward() const noexcept { return ffn_; }

        std::size_t dimModel() const noexcept { return model_dim_; }
        std::size_t numHeads() const noexcept { return num_heads_; }
        std::size_t ffnHidden() const noexcept { return ffn_hidden_; }

    private:
        CudaContext *ctx_;
        std::size_t model_dim_;
        std::size_t num_heads_;
        std::size_t ffn_hidden_;

        LayerNorm norm1_;
        MultiHeadAttention attention_;
        LayerNorm norm2_;
        FeedForwardNetwork ffn_;

        Tensor norm1_output_;
        Tensor attention_output_;
        Tensor residual1_;
        Tensor norm2_output_;
        Tensor ffn_output_;

        Tensor grad_norm1_output_;
        Tensor grad_attention_input_;
        Tensor grad_residual1_from_ffn_;
        Tensor grad_norm2_output_;
        Tensor grad_ffn_input_;
    };
} // namespace cugpt::nn
