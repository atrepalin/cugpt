#include "nn/relu.cuh"
#include "core/cuda_utils.cuh"

namespace cugpt::nn
{
    using namespace core;
    namespace
    {
        constexpr int kThreads = 256;

        __global__ void reluForwardKernel(const float *input, float *output, std::size_t total)
        {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < total)
            {
                output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
            }
        }

        __global__ void reluBackwardKernel(const float *input, const float *grad_output, float *grad_input, std::size_t total)
        {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < total)
            {
                grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
            }
        }
    }

    ReLU::ReLU(CudaContext &ctx) : ctx_(&ctx)
    {
    }

    void ReLU::forward(const Tensor &input, Tensor &output)
    {
        output.resize(input.shape());

        const int blocks = (input.numel() + kThreads - 1) / kThreads;

        CUDA_KERNEL_CHECK(reluForwardKernel<<<blocks, kThreads, 0, ctx_->stream()>>>(
            input.data(), output.data(), input.numel()));

        cached_input_ = &input;
    }

    void ReLU::backward(const Tensor &grad_output, Tensor &grad_input)
    {
        if (cached_input_ == nullptr)
        {
            throw std::logic_error("ReLU::backward called before forward");
        }

        if (grad_output.shape().empty() || grad_output.shape() != cached_input_->shape())
        {
            throw std::invalid_argument("ReLU::backward: grad_output shape mismatch");
        }

        grad_input.resize(grad_output.shape());

        const int blocks = (grad_output.numel() + kThreads - 1) / kThreads;

        CUDA_KERNEL_CHECK(reluBackwardKernel<<<blocks, kThreads, 0, ctx_->stream()>>>(
            cached_input_->data(), grad_output.data(), grad_input.data(), grad_output.numel()));
    }
} // namespace cugpt::nn