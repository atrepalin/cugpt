#include "nn/softmax.cuh"
#include "utils/reduce.cuh"

namespace cugpt::nn
{
    using namespace core;
    using namespace utils;

    namespace
    {
        constexpr int kThreads = 256;

        __global__ void softmaxForwardKernel(
            const float *x,
            float *y,
            std::size_t cols,
            std::size_t rows,
            bool causal)
        {
            extern __shared__ float shared[];

            const int tid = threadIdx.x;
            const int warpCount = (blockDim.x + kWarpSize - 1) / kWarpSize;

            float *max_shared = shared;
            float *sum_shared = shared + warpCount;

            // A block can process multiple rows using a grid-stride loop
            for (std::size_t row = static_cast<std::size_t>(blockIdx.x);
                 row < rows;
                 row += static_cast<std::size_t>(gridDim.x))
            {
                const std::size_t base = row * cols;
                const std::size_t query = row % cols;

                float local_max = -INFINITY;

                // Calculate partial maximum for reduce
                // Used when cols > kThreads
                for (std::size_t j = tid; j < cols; j += blockDim.x)
                {
                    const float value = (causal && j > query) ? -INFINITY : x[base + j];

                    local_max = fmaxf(local_max, value);
                }

                const float max_value = blockReduceKernel(local_max, max_shared, MaxOp{});

                float local_sum = 0.0f;

                for (std::size_t j = tid; j < cols; j += blockDim.x)
                {
                    const float value = (causal && j > query) ? -INFINITY : x[base + j];

                    const float e = expf(value - max_value);

                    y[base + j] = e;
                    local_sum += e;
                }

                const float sum = blockReduceKernel(local_sum, sum_shared, SumOp{});

                const float inv_sum = 1.0f / sum;

                for (std::size_t j = tid; j < cols; j += blockDim.x)
                {
                    y[base + j] *= inv_sum;
                }

                // Ensure all threads finish the current row before reusing shared memory
                __syncthreads();
            }
        }

        __global__ void softmaxBackwardKernel(
            const float *output,
            const float *grad_output,
            float *grad_input,
            std::size_t cols,
            std::size_t rows)
        {
            extern __shared__ float shared[];

            const int tid = threadIdx.x;

            for (std::size_t row = static_cast<std::size_t>(blockIdx.x);
                 row < rows;
                 row += static_cast<std::size_t>(gridDim.x))
            {
                const std::size_t base = row * cols;

                float local_dot = 0.0f;

                for (std::size_t j = tid; j < cols; j += blockDim.x)
                {
                    const std::size_t index = base + j;

                    local_dot += grad_output[index] * output[index];
                }

                // The softmax Jacobian-vector product simplifies to o * (dL/dO - sum(dL/dO * o))
                const float dot = blockReduceKernel(local_dot, shared, SumOp{});

                for (std::size_t j = tid; j < cols; j += blockDim.x)
                {
                    const std::size_t index = base + j;

                    grad_input[index] = output[index] * (grad_output[index] - dot);
                }

                __syncthreads();
            }
        }
    } // namespace

    Softmax::Softmax(CudaContext &ctx, bool causal_mask)
        : ctx_(&ctx),
          causal_mask_(causal_mask)
    {
    }

    void Softmax::forward(const Tensor &input, Tensor &output)
    {
        if (input.shape().empty())
        {
            throw std::invalid_argument("Softmax::forward: input cannot be scalar");
        }

        const std::size_t cols = input.shape().back();

        if (cols <= 0)
        {
            throw std::invalid_argument("Softmax::forward: invalid number of columns");
        }

        const std::size_t rows = input.numel() / cols;

        if (input.numel() % cols != 0)
        {
            throw std::invalid_argument("Softmax::forward: invalid tensor shape");
        }

        output.resize(input.shape());

        if (rows == 0)
        {
            return;
        }

        const int warp_count = (kThreads + kWarpSize - 1) / kWarpSize;

        const std::size_t shared_bytes = static_cast<std::size_t>(warp_count) * 2 * sizeof(float);

        CUDA_KERNEL_CHECK(softmaxForwardKernel<<<rows, kThreads, shared_bytes, ctx_->stream()>>>(
            input.data(), output.data(), cols, rows, causal_mask_));

        cached_output_ = &output;
    }

    void Softmax::backward(const Tensor &grad_output, Tensor &grad_input)
    {
        if (cached_output_ == nullptr)
        {
            throw std::logic_error("Softmax::backward called before forward");
        }

        if (grad_output.shape().empty() || grad_output.shape() != cached_output_->shape())
        {
            throw std::invalid_argument("Softmax::backward: grad_output shape mismatch");
        }

        const std::size_t cols = cached_output_->shape().back();

        if (cols <= 0)
        {
            throw std::invalid_argument("Softmax::backward: invalid number of columns");
        }

        const std::size_t numel = cached_output_->numel();
        const std::size_t rows = numel / cols;

        if (numel % cols != 0)
        {
            throw std::invalid_argument("Softmax::backward: invalid tensor shape");
        }

        grad_input.resize(cached_output_->shape());

        if (rows == 0)
        {
            return;
        }

        const int warp_count = (kThreads + kWarpSize - 1) / kWarpSize;

        const std::size_t shared_bytes = static_cast<std::size_t>(warp_count) * sizeof(float);

        CUDA_KERNEL_CHECK(softmaxBackwardKernel<<<rows, kThreads, shared_bytes, ctx_->stream()>>>(
            cached_output_->data(), grad_output.data(), grad_input.data(), cols, rows));
    }
} // namespace cugpt::nn