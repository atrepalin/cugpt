#include "nn/linear.cuh"
#include "core/cuda_utils.cuh"

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace cugpt::nn
{
    using namespace core;

    namespace
    {
        constexpr int kThreads = 256;

        std::mt19937_64 &linearRng()
        {
            static std::mt19937_64 rng(42);
            return rng;
        }

        __global__ void addBiasKernel(float *output, const float *bias, size_t rows, size_t cols)
        {
            const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t total = rows * cols;

            if (idx < total)
            {
                const int col = idx % cols;
                output[idx] += bias[col];
            }
        }

        __global__ void reduceBiasKernel(const float *grad, float *db, size_t rows, size_t cols)
        {
            const int j = blockIdx.x * blockDim.x + threadIdx.x;

            if (j >= cols)
            {
                return;
            }

            float sum = 0.0f;
            for (size_t i = 0; i < rows; ++i)
            {
                sum += grad[i * cols + j];
            }

            db[j] += sum;
        }

    } // namespace

    Linear::Linear(CudaContext &ctx, size_t in_features, size_t out_features, bool include_bias)
        : ctx_(&ctx), input_features_(in_features), output_features_(out_features),
          weights(Shape{in_features, out_features})
    {
        if (in_features <= 0 || out_features <= 0)
        {
            throw std::invalid_argument("Linear dimensions must be positive");
        }

        const float limit = std::sqrt(6.0f / static_cast<float>(in_features + out_features));
        std::uniform_real_distribution<float> distribution(-limit, limit);
        std::vector<float> host_weights(weights.data.numel());

        for (float &value : host_weights)
        {
            value = distribution(linearRng());
        }
        weights.data.copyFromHost(host_weights.data(), ctx);

        weights.grad.zero(ctx);

        if (include_bias)
        {
            bias_ = std::make_unique<Parameter>(Shape{1, out_features});
            bias_->data.zero(ctx);
            bias_->grad.zero(ctx);

            registerParameter("bias", *bias_);
        }

        registerParameter("weights", weights);
    }

    void Linear::forward(const Tensor &input, Tensor &output)
    {
        if (input.shape().empty() || input.shape().back() != input_features_)
        {
            throw std::invalid_argument("Linear::forward: last input dimension does not match input_features");
        }

        Shape out_shape = input.shape();
        out_shape.back() = output_features_;
        output.resize(out_shape);
        cached_input_ = &input;

        const size_t rows = input.numel() / input_features_;
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // [B, M] @ [M, N] -> [B, N]
        // O = X @ W (rm)
        // O^T = W^T @ X^T (cm)
        CUBLAS_CHECK(cublasSgemm(
            ctx_->blas(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            output_features_, rows, input_features_,
            &alpha,
            weights.data.data(), output_features_,
            input.data(), input_features_,
            &beta,
            output.data(), output_features_));

        if (bias_)
        {
            const int blocks = (rows * output_features_ + kThreads - 1) / kThreads;

            // [B, N] + bias for each row
            CUDA_KERNEL_CHECK(addBiasKernel<<<blocks, kThreads, 0, ctx_->stream()>>>(
                output.data(), bias_->data.data(), rows, output_features_));
        }
    }

    void Linear::backward(const Tensor &grad_output, Tensor &grad_input)
    {
        if (cached_input_ == nullptr)
        {
            throw std::logic_error("Linear::backward called before forward");
        }

        if (grad_output.shape().empty() || grad_output.shape().back() != output_features_)
        {
            throw std::invalid_argument("Linear::backward: grad_output last dimension mismatch");
        }

        Shape grad_shape = cached_input_->shape();
        grad_input.resize(grad_shape);

        const size_t rows = cached_input_->numel() / input_features_;
        const float alpha = 1.0f;
        const float beta_zero = 0.0f;
        const float beta_one = 1.0f; // accumulate into parameter gradients

        // [B, N] @ [N, M] -> [B, M]
        // dL/dX = dL/dO @ W^T (rm)
        // dL/dX^T = W @ dL/dO^T (cm)
        CUBLAS_CHECK(cublasSgemm(
            ctx_->blas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            input_features_, rows, output_features_,
            &alpha,
            weights.data.data(), output_features_,
            grad_output.data(), output_features_,
            &beta_zero,
            grad_input.data(), input_features_));

        // [M, B] @ [B, N] -> [M, N]
        // dL/dW = X^T @ dL/dO (rm)
        // dL/dW^T = dL/dO^T @ X (cm)
        CUBLAS_CHECK(cublasSgemm(
            ctx_->blas(),
            CUBLAS_OP_N, CUBLAS_OP_T,
            output_features_, input_features_, rows,
            &alpha,
            grad_output.data(), output_features_,
            cached_input_->data(), input_features_,
            &beta_one,
            weights.grad.data(), output_features_));

        if (bias_)
        {
            const int blocks = (output_features_ + kThreads - 1) / kThreads;

            // dL/dB = sum_rows(dL/dO)
            CUDA_KERNEL_CHECK(reduceBiasKernel<<<blocks, kThreads, 0, ctx_->stream()>>>(
                grad_output.data(), bias_->grad.data(), rows, output_features_));
        }
    }
} // namespace cugpt::nn