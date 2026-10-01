#include "nn/gpt.cuh"
#include "core/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace cugpt::nn
{
    using namespace core;

    GPT::GPT(
        CudaContext &ctx,
        std::size_t vocab_size,
        std::size_t blocks,
        std::size_t model_dim,
        std::size_t num_heads,
        float positional_scale)
        : ctx_(&ctx),
          vocab_size_(vocab_size),
          num_blocks_(blocks),
          model_dim_(model_dim),
          num_heads_(num_heads),
          positional_scale_(positional_scale),
          embedding_(ctx, vocab_size, model_dim),
          positional_encoding_(ctx, model_dim, positional_scale),
          final_norm_(ctx, model_dim),
          final_linear_(ctx, model_dim, vocab_size, true)
    {
        if (vocab_size <= 0 || blocks <= 0 || model_dim <= 0 || num_heads <= 0)
        {
            throw std::invalid_argument("GPT dimensions must be positive");
        }

        if (model_dim % num_heads != 0)
        {
            throw std::invalid_argument("GPT model_dim must be divisible by num_heads");
        }

        const float residual_scale = 1.0f / std::sqrt(static_cast<float>(2 * blocks));
        blocks_.reserve(static_cast<std::size_t>(blocks));
        for (int64_t i = 0; i < blocks; ++i)
        {
            blocks_.push_back(std::make_unique<TransformerBlock>(
                ctx, model_dim, num_heads, model_dim * 4, residual_scale));
        }

        registerModule("embedding_decoder", embedding_);
        for (std::size_t i = 0; i < blocks_.size(); ++i)
        {
            registerModule("decoder_blocks." + std::to_string(i), *blocks_[i]);
        }
        registerModule("final_layernorm", final_norm_);
        registerModule("final_linear", final_linear_);
    }
    
    void GPT::forward(const IntTensor &tokens, Tensor &logits)
    {
        const auto &s = tokens.shape();

        if (s.size() != 2 || s[0] <= 0 || s[1] <= 0)
        {
            throw std::invalid_argument("GPT::forward: expects tokens shape [B, T]");
        }

        embedding_.forward(tokens, embedding_output_);
        positional_encoding_.forward(embedding_output_, position_output_);

        Tensor *current = &position_output_;
        Tensor *next = &block_output_;
        for (std::size_t i = 0; i < blocks_.size(); ++i)
        {
            blocks_[i]->forward(*current, *next);
            std::swap(current, next);
        }

        final_norm_.forward(*current, final_norm_output_);
        final_linear_.forward(final_norm_output_, logits);
        cached_tokens_ = &tokens;
    }

    void GPT::backward(const Tensor &grad_output, Tensor &grad_input)
    {
        if (cached_tokens_ == nullptr)
        {
            throw std::logic_error("GPT::backward called before forward");
        }

        if (grad_output.shape().size() != 3 ||
            grad_output.shape()[0] != cached_tokens_->shape()[0] ||
            grad_output.shape()[1] != cached_tokens_->shape()[1] ||
            grad_output.shape()[2] != vocab_size_)
        {
            throw std::invalid_argument("GPT::backward: grad_output shape mismatch");
        }

        final_linear_.backward(grad_output, grad_final_norm_output_);
        final_norm_.backward(grad_final_norm_output_, grad_current_);

        for (std::size_t i = blocks_.size(); i-- > 0;)
        {
            blocks_[i]->backward(grad_current_, grad_next_);
            std::swap(grad_current_, grad_next_);
        }

        embedding_.backward(grad_current_);

        grad_input.resize(grad_current_.shape());
        CUDA_CHECK(cudaMemcpyAsync(grad_input.data(), grad_current_.data(), grad_current_.bytes(), cudaMemcpyDeviceToDevice, ctx_->stream()));
    }
} // namespace cugpt::nn