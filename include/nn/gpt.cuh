#pragma once

#include "nn/embedding.cuh"
#include "nn/positional_encoding.cuh"
#include "nn/transformer_block.cuh"
#include "nn/linear.cuh"
#include "nn/layernorm.cuh"

#include <cstdint>
#include <memory>
#include <vector>

namespace cugpt::nn
{
    using namespace core;

    class GPT final : public Module
    {
    public:
        GPT(CudaContext &ctx,
            std::size_t vocab_size,
            std::size_t blocks,
            std::size_t model_dim,
            std::size_t num_heads,
            float positional_scale = 0.1f);

        void forward(const IntTensor &tokens, Tensor &logits);
        void backward(const Tensor &grad_output, Tensor &grad_input);

        Embedding &embedding() noexcept { return embedding_; }
        const Embedding &embedding() const noexcept { return embedding_; }

        TransformerBlock &block(std::size_t index) { return *blocks_.at(static_cast<std::size_t>(index)); }
        const TransformerBlock &block(std::size_t index) const { return *blocks_.at(static_cast<std::size_t>(index)); }

        LayerNorm &finalLayerNorm() noexcept { return final_norm_; }
        const LayerNorm &finalLayerNorm() const noexcept { return final_norm_; }

        Linear &finalLinear() noexcept { return final_linear_; }
        const Linear &finalLinear() const noexcept { return final_linear_; }

        std::size_t vocabSize() const noexcept { return vocab_size_; }
        std::size_t numBlocks() const noexcept { return num_blocks_; }
        std::size_t modelDim() const noexcept { return model_dim_; }
        std::size_t numHeads() const noexcept { return num_heads_; }
        float positionalScale() const noexcept { return positional_scale_; }

    private:
        CudaContext *ctx_;
        std::size_t vocab_size_;
        std::size_t num_blocks_;
        std::size_t model_dim_;
        std::size_t num_heads_;
        float positional_scale_;

        Embedding embedding_;
        PositionalEncoding positional_encoding_;

        std::vector<std::unique_ptr<TransformerBlock>> blocks_;

        LayerNorm final_norm_;
        Linear final_linear_;

        Tensor embedding_output_;
        Tensor position_output_;
        Tensor block_output_;
        Tensor final_norm_output_;

        Tensor grad_final_norm_output_;
        Tensor grad_current_;
        Tensor grad_next_;

        const IntTensor *cached_tokens_ = nullptr;
    };
} // namespace cugpt::nn
