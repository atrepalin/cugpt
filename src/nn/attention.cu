#include "nn/attention.cuh"
#include "core/cuda_utils.cuh"

#include <cmath>
#include <stdexcept>

namespace cugpt::nn
{
    using namespace core;
    namespace
    {
        constexpr int kThreads = 256;

        // Convert linear index from [B, T, D] layout to [B, H, T, Dh] layout
        __device__ __forceinline__ const std::size_t getDstIndex(
            std::size_t idx,
            std::size_t time,
            std::size_t model_dim,
            std::size_t heads,
            std::size_t head_dim)
        {
            const std::size_t d = idx % model_dim;
            const std::size_t tmp = idx / model_dim;
            const std::size_t t = tmp % time;
            const std::size_t b = tmp / time;

            const std::size_t h = d / head_dim;
            const std::size_t hd = d % head_dim;
            const std::size_t dst_idx = (((b * heads + h) * time + t) * head_dim) + hd;

            return dst_idx;
        }

        __global__ void splitHeadsKernel(
            const float *src,
            float *dst,
            std::size_t batch,
            std::size_t time,
            std::size_t model_dim,
            std::size_t heads,
            std::size_t head_dim)
        {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            const std::size_t total = batch * time * model_dim;

            if (idx >= total)
            {
                return;
            }

            const std::size_t dst_idx = getDstIndex(idx, time, model_dim, heads, head_dim);
            dst[dst_idx] = src[idx];
        }

        __global__ void mergeHeadsKernel(
            const float *src,
            float *dst,
            std::size_t batch,
            std::size_t time,
            std::size_t model_dim,
            std::size_t heads,
            std::size_t head_dim)
        {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            const std::size_t total = batch * time * model_dim;

            if (idx >= total)
            {
                return;
            }

            const std::size_t src_idx = getDstIndex(idx, time, model_dim, heads, head_dim);
            dst[idx] = src[src_idx];
        }

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

        void launchSplit(CudaContext &ctx, const Tensor &src, Tensor &dst, std::size_t heads, std::size_t head_dim)
        {
            const auto &s = src.shape();

            if (s.size() != 3)
            {
                throw std::invalid_argument("MultiHeadAttention::launchSplit: expected [B, T, D]");
            }

            if (s[2] != heads * head_dim)
            {
                throw std::invalid_argument("MultiHeadAttention::launchSplit: dimension mismatch");
            }

            const std::size_t B = s[0], T = s[1], D = s[2];

            // [B, H, T, Dh]
            dst.resize(Shape{B, heads, T, head_dim});
            const std::size_t total = src.numel();
            const int blocks = (total + kThreads - 1) / kThreads;

            // [B, T, D] -> [B, H, T, Dh]
            CUDA_KERNEL_CHECK(splitHeadsKernel<<<blocks, kThreads, 0, ctx.stream()>>>(
                src.data(), dst.data(), B, T,
                D, heads, head_dim));
        }

        void launchMerge(CudaContext &ctx, const Tensor &src, Tensor &dst)
        {
            const auto &s = src.shape();

            if (s.size() != 4)
            {
                throw std::invalid_argument("MultiHeadAttention::launchMerge: expected [B, H, T, Dh]");
            }

            const std::size_t B = s[0], H = s[1], T = s[2], Dh = s[3];

            // [B, T, D]
            dst.resize(Shape{B, T, H * Dh});
            const std::size_t total = dst.numel();

            const int blocks = (total + kThreads - 1) / kThreads;

            // [B, H, T, Dh] -> [B, T, D]
            CUDA_KERNEL_CHECK(mergeHeadsKernel<<<blocks, kThreads, 0, ctx.stream()>>>(
                src.data(), dst.data(), B, T,
                H * Dh, H, Dh));
        }

        void batchedQK(CudaContext &ctx, const Tensor &q, const Tensor &k, Tensor &scores)
        {
            const auto &s = q.shape();

            const std::size_t B = s[0], H = s[1], T = s[2], Dh = s[3];

            // [B, H, T, T]
            scores.resize(Shape{B, H, T, T});

            const long long batch_count = B * H;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            const long long q_stride = T * Dh;
            const long long k_stride = T * Dh;
            const long long s_stride = T * T;

            // [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]
            // S = Q @ K^T (rm)
            // S^T = K @ Q^T (cm)
            CUBLAS_CHECK(cublasSgemmStridedBatched(
                ctx.blas(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(T), static_cast<int>(T), static_cast<int>(Dh),
                &alpha,
                k.data(), static_cast<int>(Dh), k_stride,
                q.data(), static_cast<int>(Dh), q_stride,
                &beta,
                scores.data(), static_cast<int>(T), s_stride,
                batch_count));
        }

        void batchedAV(CudaContext &ctx, const Tensor &attention, const Tensor &v, Tensor &out)
        {
            const auto &a = attention.shape();
            const std::size_t B = a[0], H = a[1], T = a[2], Dh = v.shape()[3];

            //[B, H, T, Dh]
            out.resize(Shape{B, H, T, Dh});

            const long long batch_count = B * H;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            const long long a_stride = T * T;
            const long long v_stride = T * Dh;
            const long long o_stride = T * Dh;

            // [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
            // O = A @ V (rm)
            // O^T = V^T @ A^T (cm)
            CUBLAS_CHECK(cublasSgemmStridedBatched(
                ctx.blas(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(Dh), static_cast<int>(T), static_cast<int>(T),
                &alpha,
                v.data(), static_cast<int>(Dh), v_stride,
                attention.data(), static_cast<int>(T), a_stride,
                &beta,
                out.data(), static_cast<int>(Dh), o_stride,
                batch_count));
        }

        void backward_dV(CudaContext &ctx, const Tensor &attention, const Tensor &grad_out, Tensor &grad_v)
        {
            const auto &a = attention.shape();
            const std::size_t B = a[0], H = a[1], T = a[2], Dh = grad_out.shape()[3];

            //[B, H, T, Dh]
            grad_v.resize(Shape{B, H, T, Dh});

            const long long batch_count = B * H;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            const long long a_stride = T * T;
            const long long v_stride = T * Dh;
            const long long o_stride = T * Dh;

            // [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
            // dL/dV = A^T @ dL/dO (rm)
            // dL/dV^T = dL/dO^T @ A (cm)
            CUBLAS_CHECK(cublasSgemmStridedBatched(
                ctx.blas(),
                CUBLAS_OP_N, CUBLAS_OP_T,
                static_cast<int>(Dh), static_cast<int>(T), static_cast<int>(T),
                &alpha,
                grad_out.data(), static_cast<int>(Dh), o_stride,
                attention.data(), static_cast<int>(T), a_stride,
                &beta,
                grad_v.data(), static_cast<int>(Dh), v_stride,
                batch_count));
        }

        void backward_dA(CudaContext &ctx, const Tensor &grad_out, const Tensor &v, Tensor &grad_a)
        {
            const auto &o = grad_out.shape();
            const std::size_t B = o[0], H = o[1], T = o[2], Dh = o[3];

            // [B, H, T, T]
            grad_a.resize(Shape{B, H, T, T});

            const long long batch_count = B * H;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            const long long a_stride = T * T;
            const long long v_stride = T * Dh;
            const long long o_stride = T * Dh;

            // [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]
            // dL/dA = dL/dO @ V^T (rm)
            // dL/dA^T = V @ dL/dO^T (cm)
            CUBLAS_CHECK(cublasSgemmStridedBatched(
                ctx.blas(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(T), static_cast<int>(T), static_cast<int>(Dh),
                &alpha,
                v.data(), static_cast<int>(Dh), v_stride,
                grad_out.data(), static_cast<int>(Dh), o_stride,
                &beta,
                grad_a.data(), static_cast<int>(T), a_stride,
                batch_count));
        }

        void backward_dQ(CudaContext &ctx, const Tensor &grad_scores, const Tensor &k, Tensor &grad_q)
        {
            const auto &s = grad_scores.shape();
            const std::size_t B = s[0], H = s[1], T = s[2], Dh = k.shape()[3];

            // [B, H, T, Dh]
            grad_q.resize(Shape{B, H, T, Dh});

            const long long batch_count = B * H;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            const long long q_stride = T * Dh;
            const long long k_stride = T * Dh;
            const long long s_stride = T * T;

            // [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
            // dL/dQ = dL/dS @ K (rm)
            // dL/dQ^T = K^T @ dL/dS^T
            CUBLAS_CHECK(cublasSgemmStridedBatched(
                ctx.blas(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(Dh), static_cast<int>(T), static_cast<int>(T),
                &alpha,
                k.data(), static_cast<int>(Dh), k_stride,
                grad_scores.data(), static_cast<int>(T), s_stride,
                &beta,
                grad_q.data(), static_cast<int>(Dh), q_stride,
                batch_count));
        }

        void backward_dK(CudaContext &ctx, const Tensor &grad_scores, const Tensor &q, Tensor &grad_k)
        {
            const auto &s = grad_scores.shape();
            const std::size_t B = s[0], H = s[1], T = s[2], Dh = q.shape()[3];

            // [B, H, T, Dh]
            grad_k.resize(Shape{B, H, T, Dh});

            const long long batch_count = B * H;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            const long long q_stride = T * Dh;
            const long long k_stride = T * Dh;
            const long long s_stride = T * T;

            // [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
            // dL/dK = dL/dS^T @ Q (rm)
            // dL^dK^T = Q^T @ dL/dS
            CUBLAS_CHECK(cublasSgemmStridedBatched(
                ctx.blas(),
                CUBLAS_OP_N, CUBLAS_OP_T,
                static_cast<int>(Dh), static_cast<int>(T), static_cast<int>(T),
                &alpha,
                q.data(), static_cast<int>(Dh), q_stride,
                grad_scores.data(), static_cast<int>(T), s_stride,
                &beta,
                grad_k.data(), static_cast<int>(Dh), k_stride,
                batch_count));
        }

    } // namespace

    MultiHeadAttention::MultiHeadAttention(CudaContext &ctx, std::size_t model_dim, std::size_t num_heads)
        : ctx_(&ctx), model_dim_(model_dim), num_heads_(num_heads), head_dim_(model_dim / num_heads),
          query_linear_(ctx, model_dim, model_dim, false),
          key_linear_(ctx, model_dim, model_dim, false),
          value_linear_(ctx, model_dim, model_dim, false),
          output_linear_(ctx, model_dim, model_dim, false),
          softmax_(ctx, true)
    {
        if (model_dim <= 0 || num_heads <= 0 || model_dim % num_heads != 0)
        {
            throw std::invalid_argument("MultiHeadAttention model_dim must be positive and divisible by num_heads");
        }

        registerModule("query_linear", query_linear_);
        registerModule("key_linear", key_linear_);
        registerModule("value_linear", value_linear_);
        registerModule("final_projection", output_linear_);
    }

    void MultiHeadAttention::forward(const Tensor &input, Tensor &output)
    {
        const auto &s = input.shape();
        const std::size_t B = s[0], T = s[1];

        q_flat_.resize(Shape{B, T, model_dim_});
        k_flat_.resize(Shape{B, T, model_dim_});
        v_flat_.resize(Shape{B, T, model_dim_});

        query_linear_.forward(input, q_flat_);
        key_linear_.forward(input, k_flat_);
        value_linear_.forward(input, v_flat_);

        launchSplit(*ctx_, q_flat_, q_, num_heads_, head_dim_);
        launchSplit(*ctx_, k_flat_, k_, num_heads_, head_dim_);
        launchSplit(*ctx_, v_flat_, v_, num_heads_, head_dim_);

        batchedQK(*ctx_, q_, k_, scores_);

        int blocks = (scores_.numel() + kThreads - 1) / kThreads;

        CUDA_KERNEL_CHECK(scaleKernel<<<blocks, kThreads, 0, ctx_->stream()>>>(
            scores_.data(), scores_.numel(), 1.0f / std::sqrt(static_cast<float>(head_dim_))));

        softmax_.forward(scores_, attention_);
        batchedAV(*ctx_, attention_, v_, attention_output_);
        launchMerge(*ctx_, attention_output_, merged_);
        output_linear_.forward(merged_, output);
    }

    void MultiHeadAttention::backward(const Tensor &grad_output, Tensor &grad_input)
    {
        if (q_.shape().empty())
        {
            throw std::logic_error("MultiHeadAttention::backward called before forward");
        }

        if (grad_output.shape() != merged_.shape())
        {
            throw std::invalid_argument("MultiHeadAttention: grad_output shape mismatch");
        }

        Tensor grad_merged;
        output_linear_.backward(grad_output, grad_merged);

        launchSplit(*ctx_, grad_merged, grad_attention_output_, num_heads_, head_dim_);

        backward_dV(*ctx_, attention_, grad_attention_output_, grad_v_);
        backward_dA(*ctx_, grad_attention_output_, v_, grad_attention_);

        softmax_.backward(grad_attention_, grad_scores_);

        int blocks = (grad_scores_.numel() + kThreads - 1) / kThreads;

        CUDA_KERNEL_CHECK(scaleKernel<<<blocks, kThreads, 0, ctx_->stream()>>>(
            grad_scores_.data(), grad_scores_.numel(), 1.0f / std::sqrt(static_cast<float>(head_dim_))));

        backward_dQ(*ctx_, grad_scores_, k_, grad_q_);
        backward_dK(*ctx_, grad_scores_, q_, grad_k_);

        launchMerge(*ctx_, grad_q_, grad_q_flat_);
        launchMerge(*ctx_, grad_k_, grad_k_flat_);
        launchMerge(*ctx_, grad_v_, grad_v_flat_);

        Tensor grad_x_q, grad_x_k, grad_x_v;
        query_linear_.backward(grad_q_flat_, grad_x_q);
        key_linear_.backward(grad_k_flat_, grad_x_k);
        value_linear_.backward(grad_v_flat_, grad_x_v);

        grad_input.resize(grad_x_q.shape());

        const std::size_t n = grad_input.numel();
        blocks = (n + kThreads - 1) / kThreads;

        CUDA_CHECK(cudaMemcpyAsync(grad_input.data(), grad_x_q.data(), grad_input.bytes(), cudaMemcpyDeviceToDevice, ctx_->stream()));

        CUDA_KERNEL_CHECK(addKernel<<<blocks, kThreads, 0, ctx_->stream()>>>(grad_input.data(), grad_x_k.data(), n));

        CUDA_KERNEL_CHECK(addKernel<<<blocks, kThreads, 0, ctx_->stream()>>>(grad_input.data(), grad_x_v.data(), n));
    }
} // namespace cugpt::nn
