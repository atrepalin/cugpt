#pragma once

#include "core/module.hpp"

namespace cugpt::nn
{
    using namespace core;

    class PositionalEncoding final : public Module
    {
    public:
        PositionalEncoding(CudaContext &ctx, std::size_t embedding_length, float scale = 1.0);

        void forward(const Tensor &input, Tensor &output);

        std::size_t inputFeatures() const noexcept { return embedding_length_; }

    private:
        CudaContext *ctx_;
        std::size_t embedding_length_;
        float scale_;
    };
} // namespace cugpt::nn
