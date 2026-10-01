#include "training/adamw.cuh"
#include "core/cuda_utils.cuh"

#include <cmath>
#include <stdexcept>

namespace cugpt::training
{
    using namespace core;
    namespace
    {
        constexpr int kThreads = 256;

        __global__ void adamwKernel(
            float *parameter,
            const float *gradient,
            float *m,
            float *v,
            std::size_t total,
            float learning_rate,
            float beta1,
            float beta2,
            float inv_beta1_correction,
            float inv_beta2_correction,
            float eps,
            float weight_decay)
        {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx >= total)
            {
                return;
            }

            const float g = gradient[idx];

            // Adam moments are updated first; bias correction then recovers an unbiased
            // estimate for early steps when m/v started at zero
            const float new_m = beta1 * m[idx] + (1.0f - beta1) * g;
            const float new_v = beta2 * v[idx] + (1.0f - beta2) * g * g;

            m[idx] = new_m;
            v[idx] = new_v;

            const float m_hat = new_m * inv_beta1_correction;
            const float v_hat = new_v * inv_beta2_correction;

            // First the adaptive update, then decoupled weight decay on the updated value
            float value = parameter[idx];
            value -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
            if (weight_decay != 0.0f)
            {
                value -= learning_rate * weight_decay * value;
            }
            parameter[idx] = value;
        }
    } // namespace

    AdamW::AdamW(
        CudaContext &ctx,
        const std::vector<Parameter *> &parameters,
        float learning_rate,
        float beta1,
        float beta2,
        float eps,
        float weight_decay)
        : ctx_(&ctx),
          parameters_(parameters),
          m_(parameters.size()),
          v_(parameters.size()),
          learning_rate_(learning_rate),
          beta1_(beta1),
          beta2_(beta2),
          eps_(eps),
          weight_decay_(weight_decay)
    {
        if (!(learning_rate > 0.0f))
        {
            throw std::invalid_argument("AdamW: learning_rate must be positive");
        }

        if (!(beta1 >= 0.0f && beta1 < 1.0f))
        {
            throw std::invalid_argument("AdamW: beta1 must be in [0, 1)");
        }

        if (!(beta2 >= 0.0f && beta2 < 1.0f))
        {
            throw std::invalid_argument("AdamW: beta2 must be in [0,1)");
        }

        if (!(eps > 0.0f))
        {
            throw std::invalid_argument("AdamW: eps must be positive");
        }

        if (weight_decay < 0.0f)
        {
            throw std::invalid_argument("AdamW: weight_decay must be non-negative");
        }

        for (std::size_t i = 0; i < parameters_.size(); ++i)
        {
            if (parameters_[i] == nullptr)
            {
                throw std::invalid_argument("AdamW: null parameter");
            }
            m_[i].resize(parameters_[i]->data.shape());
            v_[i].resize(parameters_[i]->data.shape());
            m_[i].zero(ctx);
            v_[i].zero(ctx);
        }
    }

    void AdamW::step()
    {
        // Bias correction uses the current 1-based optimizer step
        ++step_count_;

        const double beta1_corr = 1.0 - std::pow(static_cast<double>(beta1_), static_cast<double>(step_count_));
        const double beta2_corr = 1.0 - std::pow(static_cast<double>(beta2_), static_cast<double>(step_count_));

        if (!(beta1_corr > 0.0) || !(beta2_corr > 0.0))
        {
            throw std::overflow_error("AdamW: invalid bias correction");
        }

        const float inv_beta1_correction = static_cast<float>(1.0 / beta1_corr);
        const float inv_beta2_correction = static_cast<float>(1.0 / beta2_corr);

        for (std::size_t i = 0; i < parameters_.size(); ++i)
        {
            Parameter *p = parameters_[i];
            const std::size_t total = p->data.numel();

            if (total == 0)
            {
                continue;
            }

            const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);

            CUDA_KERNEL_CHECK(adamwKernel<<<blocks, kThreads, 0, ctx_->stream()>>>(
                p->data.data(), p->grad.data(), m_[i].data(), v_[i].data(), total,
                learning_rate_, beta1_, beta2_, inv_beta1_correction, inv_beta2_correction, eps_, weight_decay_));
        }

        ctx_->synchronize();
    }

    void AdamW::zeroGrad()
    {
        for (Parameter *p : parameters_)
        {
            p->zeroGrad(*ctx_);
        }
    }
} // namespace cugpt::training