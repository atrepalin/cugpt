#include "nn/positional_encoding.cuh"
#include "core/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace cugpt::nn
{
    using namespace core;
    namespace
    {
        constexpr int kThreads = 256;

        __global__ void positionalEncodingKernel(
            const float *inputs,
            float *outputs,
            std::size_t batch_size,
            std::size_t sequence_length,
            std::size_t embedding_length,
            float scale)
        {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            const std::size_t total = batch_size * sequence_length * embedding_length;

            if (idx >= total)
            {
                return;
            }

            const std::size_t col = idx % embedding_length;
            const std::size_t row = (idx / embedding_length) % sequence_length;

            const float exponent = static_cast<float>((col / 2) * 2) / static_cast<float>(embedding_length);
            const float denominator = 1.0f / powf(10000.0f, exponent);
            const float inner = static_cast<float>(row) * denominator;

            const float positional_encoding = (col % 2 == 0) ? sinf(inner) : cosf(inner);

            outputs[idx] = inputs[idx] + positional_encoding * scale;
        }

    } // namespace

    PositionalEncoding::PositionalEncoding(CudaContext &ctx, std::size_t embedding_length, float scale)
        : ctx_(&ctx),
          embedding_length_(embedding_length),
          scale_(scale)
    {
        if (embedding_length <= 0)
        {
            throw std::invalid_argument("PositionalEncoding embedding length must be positive");
        }
    }

    void PositionalEncoding::forward(const Tensor &input, Tensor &output)
    {
        const auto &s = input.shape();

        if (s[2] != embedding_length_)
        {
            throw std::invalid_argument("PositionalEncoding::forward: embedding dimension mismatch");
        }

        output.resize(s);

        const std::size_t total = input.numel();
        const int blocks = (total + kThreads - 1) / kThreads;

        CUDA_KERNEL_CHECK(positionalEncodingKernel<<<blocks, kThreads, 0, ctx_->stream()>>>(
            input.data(), output.data(), s[0], s[1], embedding_length_, scale_));
    }
} // namespace cugpt::nn