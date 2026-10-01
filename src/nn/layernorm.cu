#include "nn/layernorm.cuh"
#include "core/cuda_utils.cuh"
#include "utils/reduce.cuh"

#include <cmath>
#include <stdexcept>

namespace cugpt::nn
{
    using namespace core;
    using namespace utils;

    namespace
    {
        constexpr int kThreads = 256;

        __global__ void layerNormForwardKernel(
            const float *x,
            const float *gamma,
            const float *beta,
            float *y,
            float *mean,
            float *inv_std,
            float *xhat,
            std::size_t cols,
            std::size_t rows,
            float eps)
        {
            extern __shared__ float shared[];

            const int tid = threadIdx.x;
            const int warpCount = (blockDim.x + kWarpSize - 1) / kWarpSize;

            float *sum_shared = shared;
            float *sq_shared = shared + warpCount;

            for (std::size_t row = static_cast<std::size_t>(blockIdx.x);
                 row < rows;
                 row += static_cast<std::size_t>(gridDim.x))
            {
                const std::size_t base = row * cols;

                float local_sum = 0.0f;

                for (std::size_t j = tid; j < cols; j += blockDim.x)
                {
                    local_sum += x[base + j];
                }

                const float sum = blockReduceKernel(local_sum, sum_shared, SumOp{});

                const float mean_ = sum / static_cast<float>(cols);

                float local_sq = 0.0f;

                for (std::size_t j = tid; j < cols; j += blockDim.x)
                {
                    const float mu = x[base + j] - mean_;
                    local_sq += mu * mu;
                }

                const float sqs = blockReduceKernel(local_sq, sq_shared, SumOp{});

                const float var = sqs / static_cast<float>(cols);
                const float inv = rsqrtf(var + eps);

                mean[row] = mean_;
                inv_std[row] = inv;

                for (std::size_t j = tid; j < cols; j += blockDim.x)
                {
                    const std::size_t index = base + j;

                    const float normalized = (x[index] - mean_) * inv;
                    xhat[index] = normalized;
                    y[index] = normalized * gamma[j] + beta[j];
                }

                __syncthreads();
            }
        }

        __global__ void layerNormBackwardKernel(
            const float *gamma,
            const float *beta,
            float *mean,
            float *inv_std,
            float *xhat,
            const float *grad_output,
            float *grad_input,
            std::size_t cols,
            std::size_t rows)
        {
            extern __shared__ float shared[];

            const int tid = threadIdx.x;
            const int warpCount = (blockDim.x + kWarpSize - 1) / kWarpSize;

            float *sum_g_shared = shared;
            float *sum_g_xhat_shared = shared + warpCount;

            for (std::size_t row = static_cast<std::size_t>(blockIdx.x);
                 row < rows;
                 row += static_cast<std::size_t>(gridDim.x))
            {
                const std::size_t base = row * cols;

                float local_sum_g = 0.0f;
                float local_sum_g_xhat = 0.0f;

                for (std::size_t j = tid; j < cols; j += blockDim.x)
                {
                    const std::size_t index = base + j;

                    const float grad = grad_output[index];
                    const float g = grad * gamma[j];
                    const float xh = xhat[index];

                    local_sum_g += g;
                    local_sum_g_xhat += g * xh;
                }

                const float sum_g = blockReduceKernel(local_sum_g, sum_g_shared, SumOp{});
                const float sum_g_xhat = blockReduceKernel(local_sum_g_xhat, sum_g_xhat_shared, SumOp{});

                const float inv = inv_std[row];
                const float inv_cols = 1.0f / static_cast<float>(cols);

                for (std::size_t j = tid; j < cols; j += blockDim.x)
                {
                    const std::size_t index = base + j;

                    const float grad = grad_output[index];
                    const float g = grad * gamma[j];
                    const float xh = xhat[index];

                    const float dx = inv * (g - sum_g * inv_cols - xh * sum_g_xhat * inv_cols);

                    grad_input[index] = dx;
                }

                __syncthreads();
            }
        }

        __global__ void layerNormParamBackwardKernel(
            const float *xhat,
            const float *grad_output,
            float *grad_gamma,
            float *grad_beta,
            std::size_t cols,
            std::size_t rows)
        {
            extern __shared__ float shared[];

            const int tid = threadIdx.x;

            const std::size_t col = static_cast<std::size_t>(blockIdx.x);

            if (col >= cols)
            {
                return;
            }

            float local_gamma = 0.0f;
            float local_beta = 0.0f;

            for (std::size_t row = static_cast<std::size_t>(tid);
                 row < rows;
                 row += static_cast<std::size_t>(blockDim.x))
            {
                const std::size_t index = row * cols + col;

                const float grad = grad_output[index];
                const float xh = xhat[index];

                local_gamma += grad * xh;
                local_beta += grad;
            }

            const float gamma_sum = blockReduceKernel(local_gamma, shared, SumOp{});
            const float beta_sum = blockReduceKernel(local_beta, shared, SumOp{});

            if (tid == 0)
            {
                grad_gamma[col] = gamma_sum;
                grad_beta[col] = beta_sum;
            }
        }
    } // namespace

    LayerNorm::LayerNorm(CudaContext &ctx, std::size_t input_features, float eps)
        : ctx_(&ctx), input_features_(input_features), eps_(eps),
          gamma(Shape{1, input_features}), beta(Shape{1, input_features})
    {
        if (input_features <= 0)
        {
            throw std::invalid_argument("LayerNorm dimensions must be positive");
        }

        gamma.data.zero(ctx);
        beta.data.zero(ctx);
        gamma.data.fill(1.0f, ctx);
        beta.data.zero(ctx);
        gamma.grad.zero(ctx);
        beta.grad.zero(ctx);

        registerParameter("gain", gamma);
        registerParameter("bias", beta);
    }

    void LayerNorm::forward(const Tensor &input, Tensor &output)
    {
        const std::size_t cols = input.shape().back();

        if (cols <= 0)
        {
            throw std::invalid_argument("LayerNorm::forward: invalid number of columns");
        }

        const std::size_t rows = input.numel() / cols;

        if (input.numel() % cols != 0)
        {
            throw std::invalid_argument("LayerNorm::forward: invalid tensor shape");
        }

        output.resize(input.shape());
        mean_.resize(Shape{rows});
        inv_std_.resize(Shape{rows});
        xhat_.resize(input.shape());

        if (rows == 0)
        {
            return;
        }

        const int warp_count = (kThreads + kWarpSize - 1) / kWarpSize;

        const std::size_t shared_bytes = static_cast<std::size_t>(warp_count) * 2 * sizeof(float);

        CUDA_KERNEL_CHECK(layerNormForwardKernel<<<rows, kThreads, shared_bytes, ctx_->stream()>>>(
            input.data(), gamma.data.data(), beta.data.data(), output.data(),
            mean_.data(), inv_std_.data(), xhat_.data(), cols, rows, eps_));
    }

    void LayerNorm::backward(const Tensor &grad_output, Tensor &grad_input)
    {
        if (grad_output.shape().empty() || grad_output.shape() != xhat_.shape())
        {
            throw std::invalid_argument("LayerNorm::backward: grad_output shape mismatch");
        }

        const std::size_t cols = xhat_.shape().back();

        if (cols <= 0)
        {
            throw std::invalid_argument("LayerNorm::backward: invalid number of columns");
        }

        const std::size_t numel = xhat_.numel();
        const std::size_t rows = numel / cols;

        if (numel % cols != 0)
        {
            throw std::invalid_argument("LayerNorm::backward: invalid tensor shape");
        }

        grad_input.resize(grad_output.shape());

        if (rows == 0)
        {
            return;
        }

        const int warp_count = (kThreads + kWarpSize - 1) / kWarpSize;

        const std::size_t shared_bytes = static_cast<std::size_t>(warp_count) * sizeof(float);

        CUDA_KERNEL_CHECK(layerNormBackwardKernel<<<rows, kThreads, shared_bytes * 2, ctx_->stream()>>>(
            gamma.data.data(), beta.data.data(), mean_.data(), inv_std_.data(), xhat_.data(),
            grad_output.data(), grad_input.data(), cols, rows));

        CUDA_KERNEL_CHECK(layerNormParamBackwardKernel<<<cols, kThreads, shared_bytes, ctx_->stream()>>>(
            xhat_.data(), grad_output.data(), gamma.grad.data(), beta.grad.data(),
            cols, rows));
    }
} // namespace cugpt::nn