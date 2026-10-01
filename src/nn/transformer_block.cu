#include "nn/transformer_block.cuh"
#include "core/cuda_utils.cuh"

#include <stdexcept>

namespace cugpt::nn
{
    using namespace core;
    namespace
    {
        constexpr int kThreads = 256;

        __global__ void scaleKernel(float *data, std::size_t total, float scale)
        {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < total)
            {
                data[idx] *= scale;
            }
        }

        __global__ void addKernel(float *dst, const float *src, std::size_t total)
        {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < total)
            {
                dst[idx] += src[idx];
            }
        }

        void scaleTensor(CudaContext &ctx, Tensor &tensor, float scale)
        {
            if (scale == 1.0f || tensor.numel() == 0)
            {
                return;
            }

            const int blocks = (tensor.numel() + kThreads - 1) / kThreads;
            CUDA_KERNEL_CHECK(scaleKernel<<<blocks, kThreads, 0, ctx.stream()>>>(tensor.data(), tensor.numel(), scale));
        }

        void add(CudaContext &ctx, Tensor &dst, const Tensor &src)
        {
            if (dst.shape() != src.shape())
            {
                throw std::invalid_argument("TransformerBlock::add: residual shape mismatch");
            }

            const int blocks = static_cast<int>((dst.numel() + kThreads - 1) / kThreads);
            CUDA_KERNEL_CHECK(addKernel<<<blocks, kThreads, 0, ctx.stream()>>>(dst.data(), src.data(), dst.numel()));
        }
    } // namespace

    TransformerBlock::TransformerBlock(
        CudaContext &ctx,
        std::size_t model_dim,
        std::size_t num_heads,
        std::size_t ffn_hidden,
        float residual_weight_scale)
        : ctx_(&ctx),
          model_dim_(model_dim),
          num_heads_(num_heads),
          ffn_hidden_(ffn_hidden),
          norm1_(ctx, model_dim),
          attention_(ctx, model_dim, num_heads),
          norm2_(ctx, model_dim),
          ffn_(ctx, model_dim, ffn_hidden, model_dim)
    {
        if (model_dim <= 0 || num_heads <= 0 || ffn_hidden <= 0)
        {
            throw std::invalid_argument("TransformerBlock dimensions must be positive");
        }

        registerModule("layernorm1", norm1_);
        registerModule("multi_head_attention", attention_);
        registerModule("layernorm2", norm2_);
        registerModule("ffn", ffn_);

        if (!(residual_weight_scale > 0.0f))
        {
            throw std::invalid_argument("TransformerBlock residual_weight_scale must be positive");
        }

        // Scaling only the output projections reduces the initial residual branch magnitude
        // while preserving the internal attention/FFN parameterization
        scaleTensor(ctx, attention_.outputLinear().weights.data, residual_weight_scale);
        scaleTensor(ctx, ffn_.secondLinear().weights.data, residual_weight_scale);
    }

    void TransformerBlock::forward(const Tensor &input, Tensor &output)
    {
        norm1_.forward(input, norm1_output_);
        attention_.forward(norm1_output_, attention_output_);

        residual1_.resize(input.shape());

        CUDA_CHECK(cudaMemcpyAsync(residual1_.data(), input.data(), input.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_->stream()));
        add(*ctx_, residual1_, attention_output_);

        norm2_.forward(residual1_, norm2_output_);
        ffn_.forward(norm2_output_, ffn_output_);

        output.resize(input.shape());

        CUDA_CHECK(cudaMemcpyAsync(output.data(), residual1_.data(), residual1_.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_->stream()));
        add(*ctx_, output, ffn_output_);
    }

    void TransformerBlock::backward(const Tensor &grad_output, Tensor &grad_input)
    {
        if (residual1_.shape().empty())
        {
            throw std::logic_error("TransformerBlock::backward: called before forward");
        }

        if (grad_output.shape() != residual1_.shape())
        {
            throw std::invalid_argument("TransformerBlock::backward: grad_output shape mismatch");
        }

        ffn_.backward(grad_output, grad_norm2_output_);
        norm2_.backward(grad_norm2_output_, grad_residual1_from_ffn_);

        grad_attention_input_.resize(grad_output.shape());

        CUDA_CHECK(cudaMemcpyAsync(grad_attention_input_.data(), grad_output.data(), grad_output.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_->stream()));

        add(*ctx_, grad_attention_input_, grad_residual1_from_ffn_);

        attention_.backward(grad_attention_input_, grad_norm1_output_);
        norm1_.backward(grad_norm1_output_, grad_input);
        add(*ctx_, grad_input, grad_attention_input_);
    }
} // namespace cugpt::nn