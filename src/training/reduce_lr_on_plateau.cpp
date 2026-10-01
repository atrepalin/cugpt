#include "training/reduce_lr_on_plateau.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace cugpt::training
{
    namespace
    {
        constexpr float kPositiveInfinity = std::numeric_limits<float>::infinity();
        constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
    }

    ReduceLROnPlateau::ReduceLROnPlateau(AdamW &optimizer, const Config &config)
        : optimizer_(&optimizer),
          config_(config)
    {
        if (!(config_.factor >= 0.0f && config_.factor < 1.0f) || !std::isfinite(config_.factor))
        {
            throw std::invalid_argument("ReduceLROnPlateau: factor must be finite and in [0, 1)");
        }

        if (!(config_.threshold >= 0.0f) || !std::isfinite(config_.threshold))
        {
            throw std::invalid_argument("ReduceLROnPlateau: threshold must be finite and non-negative");
        }

        if (!(config_.min_lr >= 0.0f) || !std::isfinite(config_.min_lr))
        {
            throw std::invalid_argument("ReduceLROnPlateau: min_lr must be finite and non-negative");
        }

        if (!(config_.eps >= 0.0f) || !std::isfinite(config_.eps))
        {
            throw std::invalid_argument("ReduceLROnPlateau: eps must be finite and non-negative");
        }

        best_ = config_.mode == ReduceLROnPlateauMode::Min ? kPositiveInfinity : kNegativeInfinity;
    }

    bool ReduceLROnPlateau::isBetter(float current) const noexcept
    {
        if (config_.mode == ReduceLROnPlateauMode::Min)
        {
            if (config_.threshold_mode == ReduceLROnPlateauThresholdMode::Rel)
            {
                return current < best_ * (1.0f - config_.threshold);
            }
            return current < best_ - config_.threshold;
        }

        if (config_.threshold_mode == ReduceLROnPlateauThresholdMode::Rel)
        {
            return current > best_ * (1.0f + config_.threshold);
        }
        return current > best_ + config_.threshold;
    }

    void ReduceLROnPlateau::reduceLearningRate()
    {
        const float old_lr = optimizer_->learningRate();
        const float proposed_lr = old_lr * config_.factor;
        const float new_lr = proposed_lr > config_.min_lr ? proposed_lr : config_.min_lr;

        if ((old_lr - new_lr) > config_.eps)
        {
            optimizer_->setLearningRate(new_lr);
        }
    }

    bool ReduceLROnPlateau::step(float metric)
    {
        if (!std::isfinite(metric))
        {
            throw std::invalid_argument("ReduceLROnPlateau: metric must be finite");
        }

        if (isBetter(metric))
        {
            best_ = metric;
            bad_epochs_ = 0;
            return false;
        }

        if (cooldown_counter_ > 0)
        {
            --cooldown_counter_;
            bad_epochs_ = 0;
            return false;
        }

        ++bad_epochs_;
        if (bad_epochs_ <= config_.patience)
        {
            return false;
        }

        const float old_lr = optimizer_->learningRate();
        reduceLearningRate();
        const bool reduced = optimizer_->learningRate() < old_lr;

        bad_epochs_ = 0;
        if (reduced)
        {
            cooldown_counter_ = config_.cooldown;
        }
        return reduced;
    }
} // namespace cugpt::training
