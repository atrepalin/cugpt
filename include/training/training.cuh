#pragma once

#include "data/dataset.hpp"
#include "nn/gpt.cuh"
#include "training/cross_entropy.cuh"
#include "training/adamw.cuh"
#include "training/reduce_lr_on_plateau.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

namespace cugpt::training
{
    using namespace core;
    using namespace nn;
    using namespace data;

    float trainStep(
        CudaContext &ctx,
        GPT &model,
        CrossEntropyLoss &loss,
        AdamW &optimizer,
        const IntTensor &tokens,
        const IntTensor &targets);

    struct EvaluationMetrics
    {
        float loss = 0.0f;
        float sequence_accuracy = 0.0f;
    };

    struct EpochMetrics
    {
        std::size_t epoch = 0;
        std::size_t steps = 0;
        float train_loss = 0.0f;
        float val_loss = 0.0f;
        float val_sequence_accuracy = 0.0f;
        float learning_rate = 0.0f;
        bool has_validation = false;
    };

    enum class PlateauMonitor
    {
        TrainLoss,
        ValidationLoss,
        ValidationAccuracy
    };

    struct TrainingConfig
    {
        std::size_t epochs = 1;
        std::size_t batch_size = 64;
        std::uint64_t seed = 42;
        int32_t ignore_index = 0;
        std::size_t print_every = 1;
        std::string best_checkpoint_path = "best_model.safetensors";
        PlateauMonitor plateau_monitor = PlateauMonitor::ValidationLoss;
    };

    EvaluationMetrics evaluate(
        CudaContext &ctx,
        GPT &model,
        CrossEntropyLoss &loss,
        const Dataset &dataset,
        std::size_t batch_size,
        bool validation,
        int32_t ignore_index);

    std::vector<EpochMetrics> trainModel(
        CudaContext &ctx,
        GPT &model,
        CrossEntropyLoss &loss,
        AdamW &optimizer,
        ReduceLROnPlateau *lr_scheduler,
        const Dataset &dataset,
        const TrainingConfig &config);
} // namespace cugpt::training
