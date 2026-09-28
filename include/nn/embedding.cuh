#pragma once

#include "core/module.hpp"

namespace cugpt::nn
{
    using namespace core;

    class Embedding final : public Module
    {
    public:
        Embedding(CudaContext &ctx, std::size_t vocab_size, std::size_t embedding_dim);

        void forward(const IntTensor &tokens, Tensor &output);
        void backward(const Tensor &grad_output);

        Parameter embeddings;
        std::size_t vocabSize() const noexcept { return vocab_size_; }
        std::size_t embeddingDim() const noexcept { return embedding_dim_; }

    private:
        CudaContext *ctx_;
        std::size_t vocab_size_;
        std::size_t embedding_dim_;
        const IntTensor *cached_tokens_ = nullptr;
    };
} // namespace cugpt::nn