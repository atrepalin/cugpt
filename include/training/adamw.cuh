#pragma once

#include "core/parameter.hpp"

#include <cstdint>
#include <vector>

namespace cugpt::training
{
    using namespace core;

    class AdamW final
    {
    public:
        AdamW(CudaContext &ctx,
              const std::vector<Parameter *> &parameters,
              float learning_rate = 3.0e-4f,
              float beta1 = 0.9f,
              float beta2 = 0.999f,
              float eps = 1.0e-8f,
              float weight_decay = 0.0f);

        AdamW(const AdamW &) = delete;
        AdamW &operator=(const AdamW &) = delete;

        void step();
        void zeroGrad();

        std::uint64_t stepCount() const noexcept { return step_count_; }
        float learningRate() const noexcept { return learning_rate_; }
        float beta1() const noexcept { return beta1_; }
        float beta2() const noexcept { return beta2_; }
        float eps() const noexcept { return eps_; }
        float weightDecay() const noexcept { return weight_decay_; }

    private:
        CudaContext *ctx_;
        std::vector<Parameter *> parameters_;
        std::vector<Tensor> m_;
        std::vector<Tensor> v_;

        float learning_rate_;
        float beta1_;
        float beta2_;
        float eps_;
        float weight_decay_;
        std::uint64_t step_count_ = 0;
    };
} // namespace cugpt::training
