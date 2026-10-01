#include "training/cross_entropy.cuh"
#include "core/cuda_utils.cuh"
#include "utils/reduce.cuh"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace cugpt::training
{
    using namespace core;
    using namespace utils;

    namespace
    {
        constexpr int kThreads = 256;

        __global__ void crossEntropyKernel(
            const float *logits,
            const int32_t *targets,
            float *grad,
            float *row_loss,
            int32_t *row_valid,
            std::size_t rows,
            std::size_t vocab,
            int32_t ignore_index)
        {
            extern __shared__ float shared[];

            const int tid = threadIdx.x;
            const int warp_count = (blockDim.x + kWarpSize - 1) / kWarpSize;

            float *max_shared = shared;
            float *sum_shared = shared + warp_count;

            for (std::size_t row = static_cast<std::size_t>(blockIdx.x);
                 row < rows;
                 row += static_cast<std::size_t>(gridDim.x))
            {
                const std::size_t base = row * vocab;

                float local_max = -INFINITY;

                // Find max(logits) for numerically stable softmax

                for (std::size_t j = static_cast<std::size_t>(tid);
                     j < vocab;
                     j += static_cast<std::size_t>(blockDim.x))
                {
                    local_max = fmaxf(local_max, logits[base + j]);
                }

                const float max_logit = blockReduceKernel(local_max, max_shared, MaxOp{});

                // Compute sum(exp(logits - max_logit))
                float local_sum = 0.0f;

                for (std::size_t j = static_cast<std::size_t>(tid);
                     j < vocab;
                     j += static_cast<std::size_t>(blockDim.x))
                {
                    local_sum += expf(logits[base + j] - max_logit);
                }

                const float sum_exp = blockReduceKernel(local_sum, sum_shared, SumOp{});

                const float logsumexp = max_logit + logf(sum_exp);

                const int32_t target = targets[row];
                const bool valid = target != ignore_index;

                if (tid == 0)
                {
                    row_valid[row] = valid ? 1 : 0;

                    row_loss[row] = valid ? (logsumexp - logits[base + target]) : 0.0f;
                }

                // dL/dlogits = softmax - one_hot
                const float inv_sum = 1.0f / sum_exp;

                for (std::size_t j = static_cast<std::size_t>(tid);
                     j < vocab;
                     j += static_cast<std::size_t>(blockDim.x))
                {
                    float probability = expf(logits[base + j] - max_logit) * inv_sum;

                    if (static_cast<int32_t>(j) == target)
                    {
                        probability -= 1.0f;
                    }

                    grad[base + j] = valid ? probability : 0.0f;
                }

                __syncthreads();
            }
        }

        __global__ void reduceLossKernel(
            const float *row_loss,
            const int32_t *row_valid,
            float *total_loss,
            int32_t *valid_count,
            std::size_t rows)
        {
            extern __shared__ float shared[];

            const int tid = threadIdx.x;

            const int warp_count = (blockDim.x + kWarpSize - 1) / kWarpSize;

            float *loss_shared = shared;
            float *valid_shared = shared + warp_count;

            float local_loss = 0.0f;
            float local_valid = 0.0f;

            for (std::size_t row = static_cast<std::size_t>(tid);
                 row < rows;
                 row += static_cast<std::size_t>(blockDim.x))
            {
                local_loss += row_loss[row];
                local_valid += static_cast<float>(row_valid[row]);
            }

            const float loss = blockReduceKernel(local_loss, loss_shared, SumOp{});
            const float valid = blockReduceKernel(local_valid, valid_shared, SumOp{});

            if (threadIdx.x == 0)
            {
                *total_loss = loss;
                *valid_count = static_cast<int32_t>(valid);
            }
        }

        __global__ void normalizeLossGradKernel(
            float *grad,
            float *total_loss,
            const int32_t *valid_count,
            std::size_t total)
        {
            const int32_t count = *valid_count;

            const float inv = count > 0 ? 1.0f / static_cast<float>(count) : 0.0f;

            if (blockIdx.x == 0 && threadIdx.x == 0)
            {
                *total_loss *= inv;
            }

            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < total)
            {
                grad[idx] *= inv;
            }
        }
    } // namespace

    CrossEntropyLoss::CrossEntropyLoss(CudaContext &ctx, int32_t ignore_index)
        : ctx_(&ctx),
          ignore_index_(ignore_index)
    {
    }

    float CrossEntropyLoss::forwardBackward(const Tensor &logits, const IntTensor &targets, Tensor &grad_logits)
    {
        if (logits.shape().empty())
        {
            throw std::invalid_argument("CrossEntropyLoss::forwardBackward: logits cannot be scalar");
        }

        if (logits.shape().size() != targets.shape().size() + 1)
        {
            throw std::invalid_argument("CrossEntropyLoss::forwardBackward: target rank must be logits rank - 1");
        }

        for (std::size_t i = 0; i < targets.shape().size(); ++i)
        {
            if (logits.shape()[i] != targets.shape()[i])
            {
                throw std::invalid_argument("CrossEntropyLoss::forwardBackward: shape mismatch");
            }
        }

        const std::size_t vocab = logits.shape().back();

        if (vocab <= 0)
        {
            throw std::invalid_argument("CrossEntropyLoss::forwardBackward: vocabulary size must be positive");
        }

        const std::size_t rows = targets.numel();

        grad_logits.resize(logits.shape());

        if (rows == 0)
        {
            return 0.0f;
        }

        Tensor row_loss(Shape{rows});
        IntTensor row_valid(Shape{rows});

        Tensor total_loss(Shape{1});
        IntTensor valid_count(Shape{1});

        const int warp_count = (kThreads + kWarpSize - 1) / kWarpSize;

        const std::size_t shared_bytes = static_cast<std::size_t>(warp_count) * 2 * sizeof(float);

        // Calculate grad, loss and valid count for each row
        CUDA_KERNEL_CHECK(crossEntropyKernel<<<rows, kThreads, shared_bytes, ctx_->stream()>>>(
            logits.data(), targets.data(), grad_logits.data(), row_loss.data(), row_valid.data(),
            rows, vocab, ignore_index_));

        // Reduce total loss and number of valid rows
        CUDA_KERNEL_CHECK(reduceLossKernel<<<1, kThreads, shared_bytes, ctx_->stream()>>>(
            row_loss.data(), row_valid.data(), total_loss.data(), valid_count.data(),
            rows));

        // Normalize loss and gradient by valid row count
        const std::size_t grad_blocks = (grad_logits.numel() + kThreads - 1) / kThreads;

        CUDA_KERNEL_CHECK(normalizeLossGradKernel<<<grad_blocks, kThreads, 0, ctx_->stream()>>>(
            grad_logits.data(), total_loss.data(), valid_count.data(), grad_logits.numel()));

        float host_loss = 0.0f;
        total_loss.copyToHost(&host_loss, *ctx_);

        return host_loss;
    }
} // namespace cugpt::training