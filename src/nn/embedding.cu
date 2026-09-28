#include "nn/embedding.cuh"
#include "core/cuda_utils.cuh"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace cugpt::nn
{
    using namespace core;
    namespace
    {
        constexpr int kThreads = 256;

        std::mt19937_64 &embeddingRng()
        {
            static std::mt19937_64 rng(42);
            return rng;
        }

        __global__ void embeddingForwardKernel(
            const int32_t *tokens,
            const float *table,
            float *output,
            std::size_t token_count,
            std::size_t dim,
            std::size_t vocab)
        {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx >= token_count)
            {
                return;
            }

            const int32_t token = tokens[idx];
            if (token < 0 || token >= vocab)
            {
                return;
            }

            const float *src = table + token * dim;
            float *dst = output + idx * dim;

            memcpy(dst, src, sizeof(float) * dim);
        }

        // Several positions may select the same vocabulary row, so the gradient
        // update must accumulate atomically rather than overwrite the row
        __global__ void embeddingBackwardKernel(
            const int32_t *tokens,
            const float *grad_output,
            float *grad_table,
            std::size_t token_count,
            std::size_t dim,
            std::size_t vocab)
        {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            const std::size_t total = token_count * dim;

            if (idx >= total)
            {
                return;
            }

            const std::size_t token_index = idx / dim;
            const std::size_t column = idx % dim;
            const int32_t token = tokens[token_index];

            if (token < 0 || token >= vocab)
            {
                return;
            }

            atomicAdd(grad_table + static_cast<std::size_t>(token) * dim + column, grad_output[idx]);
        }
    }

    Embedding::Embedding(CudaContext &ctx, std::size_t vocab_size, std::size_t embedding_dim)
        : ctx_(&ctx),
          vocab_size_(vocab_size),
          embedding_dim_(embedding_dim),
          embeddings(Shape{vocab_size, embedding_dim})
    {
        if (vocab_size <= 0 || embedding_dim <= 0)
        {
            throw std::invalid_argument("Embedding dimensions must be positive");
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(embedding_dim));
        std::normal_distribution<float> distribution(0.0f, scale);
        std::vector<float> host_embeddings(embeddings.data.numel());

        for (float &value : host_embeddings)
        {
            value = distribution(embeddingRng());
        }
        embeddings.data.copyFromHost(host_embeddings.data(), ctx);

        embeddings.grad.zero(ctx);

        registerParameter("embeddings", embeddings);
    }

    void Embedding::forward(const IntTensor &tokens, Tensor &output)
    {
        if (tokens.shape().empty())
        {
            throw std::invalid_argument("Embedding::forward: tokens cannot be scalar");
        }

        Shape out_shape = tokens.shape();
        out_shape.push_back(embedding_dim_);
        output.resize(out_shape);

        const std::size_t count = tokens.numel();
        const int blocks = (count + kThreads - 1) / kThreads;

        CUDA_KERNEL_CHECK(embeddingForwardKernel<<<blocks, kThreads, 0, ctx_->stream()>>>(
            tokens.data(), embeddings.data.data(), output.data(), count,
            embedding_dim_, vocab_size_));

        cached_tokens_ = &tokens;
    }

    void Embedding::backward(const Tensor &grad_output)
    {
        if (cached_tokens_ == nullptr)
        {
            throw std::logic_error("Embedding::backward called before forward");
        }

        if (grad_output.shape().empty() || grad_output.shape().back() != embedding_dim_)
        {
            throw std::invalid_argument("Embedding::backward: grad_output last dimension mismatch");
        }

        if (grad_output.shape().size() != cached_tokens_->shape().size() + 1)
        {
            throw std::invalid_argument("Embedding::backward: seems like grad_output doesn't have grad for new token");
        }

        const std::size_t token_count = cached_tokens_->numel();
        const std::size_t total = token_count * embedding_dim_;
        const int blocks = (total + kThreads - 1) / kThreads;

        CUDA_KERNEL_CHECK(embeddingBackwardKernel<<<blocks, kThreads, 0, ctx_->stream()>>>(
            cached_tokens_->data(), grad_output.data(), embeddings.grad.data(), token_count,
            embedding_dim_, vocab_size_));
    }
} // namespace cugpt::nn