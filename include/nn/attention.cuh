#pragma once

#include "nn/linear.cuh"
#include "nn/softmax.cuh"

namespace cugpt::nn
{
    using namespace core;

    class MultiHeadAttention final : public Module
    {
    public:
        MultiHeadAttention(CudaContext &ctx, std::size_t model_dim, std::size_t num_heads);

        void forward(const Tensor &input, Tensor &output);
        void backward(const Tensor &grad_output, Tensor &grad_input);

        Linear &queryLinear() noexcept { return query_linear_; }
        Linear &keyLinear() noexcept { return key_linear_; }
        Linear &valueLinear() noexcept { return value_linear_; }
        Linear &outputLinear() noexcept { return output_linear_; }

        const Linear &queryLinear() const noexcept { return query_linear_; }
        const Linear &keyLinear() const noexcept { return key_linear_; }
        const Linear &valueLinear() const noexcept { return value_linear_; }
        const Linear &outputLinear() const noexcept { return output_linear_; }

        std::size_t modelDim() const noexcept { return model_dim_; }
        std::size_t numHeads() const noexcept { return num_heads_; }
        std::size_t headDim() const noexcept { return head_dim_; }

    private:
        CudaContext *ctx_;
        std::size_t model_dim_;
        std::size_t num_heads_;
        std::size_t head_dim_;

        Linear query_linear_;
        Linear key_linear_;
        Linear value_linear_;
        Linear output_linear_;
        Softmax softmax_;

        Tensor q_;                // [B, H, T, Dh]
        Tensor k_;                // [B, H, T, Dh]
        Tensor v_;                // [B, H, T, Dh]
        Tensor scores_;           // [B, H, T, T]
        Tensor attention_;        // [B, H, T, T]
        Tensor attention_output_; // [B, H, T, Dh]
        Tensor merged_;           // [B, T, D]

        Tensor q_flat_; // [B, T, D]
        Tensor k_flat_; // [B, T, D]
        Tensor v_flat_; // [B, T, D]

        Tensor grad_q_;                // [B, H, T, Dh]
        Tensor grad_k_;                // [B, H, T, Dh]
        Tensor grad_v_;                // [B, H, T, Dh]
        Tensor grad_scores_;           // [B, H, T, T]
        Tensor grad_attention_;        // [B, H, T, T]
        Tensor grad_attention_output_; // [B, H, T, Dh]
        Tensor grad_q_flat_;           // [B, T, D]
        Tensor grad_k_flat_;           // [B, T, D]
        Tensor grad_v_flat_;           // [B, T, D]
    };
} // namespace cugpt::nn