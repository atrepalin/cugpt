#pragma once

#include "training/adamw.cuh"

#include <cstddef>

namespace cugpt::training
{
    enum class ReduceLROnPlateauMode
    {
        Min,
        Max
    };

    enum class ReduceLROnPlateauThresholdMode
    {
        Rel,
        Abs
    };

    class ReduceLROnPlateau final
    {
    public:
        struct Config
        {
            ReduceLROnPlateauMode mode = ReduceLROnPlateauMode::Min;
            float factor = 0.1f;
            std::size_t patience = 10;
            float threshold = 1.0e-4f;
            ReduceLROnPlateauThresholdMode threshold_mode = ReduceLROnPlateauThresholdMode::Rel;
            std::size_t cooldown = 0;
            float min_lr = 0.0f;
            float eps = 1.0e-8f;
        };

        explicit ReduceLROnPlateau(AdamW &optimizer, const Config &config);

        ReduceLROnPlateau(const ReduceLROnPlateau &) = delete;
        ReduceLROnPlateau &operator=(const ReduceLROnPlateau &) = delete;

        // Supply one scalar metric once per scheduler interval, normally once per epoch.
        // Returns true when the optimizer learning rate was actually reduced
        bool step(float metric);

        float best() const noexcept { return best_; }
        std::size_t badEpochs() const noexcept { return bad_epochs_; }
        std::size_t cooldownCounter() const noexcept { return cooldown_counter_; }
        float currentLearningRate() const noexcept { return optimizer_->learningRate(); }
        const Config &config() const noexcept { return config_; }

    private:
        bool isBetter(float current) const noexcept;
        void reduceLearningRate();

        AdamW *optimizer_;
        Config config_;
        float best_ = 0.0f;
        std::size_t bad_epochs_ = 0;
        std::size_t cooldown_counter_ = 0;
    };
} // namespace cugpt::training
