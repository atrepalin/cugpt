#include "training/training.cuh"
#include "io/serialization.hpp"
#include "core/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <fstream>

namespace cugpt::training
{
    using namespace core;
    using namespace nn;
    using namespace data;
    namespace
    {
        constexpr int kArgmaxThreads = 256;

        __global__ void argmaxRowsKernel(
            const float *logits,
            int32_t *predictions,
            std::size_t rows,
            int vocab)
        {
            const std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= rows)
            {
                return;
            }

            const float *base = logits + row * static_cast<std::size_t>(vocab);
            int32_t best = 0;
            float best_value = base[0];
            for (int j = 1; j < vocab; ++j)
            {
                const float value = base[j];
                if (value > best_value)
                {
                    best_value = value;
                    best = static_cast<int32_t>(j);
                }
            }
            predictions[row] = best;
        }

        void copyIndexedBatch(
            const Dataset &dataset,
            bool validation,
            const std::vector<std::size_t> &order,
            std::size_t start,
            std::size_t count,
            std::vector<int32_t> &host_x,
            std::vector<int32_t> &host_y)
        {
            const std::size_t total = validation ? dataset.valSize() : dataset.trainSize();

            if (start > total || count > total - start)
            {
                throw std::out_of_range("copyIndexedBatch: range out of bounds");
            }

            if (order.size() != total)
            {
                throw std::invalid_argument("copyIndexedBatch: order size does not match dataset size");
            }

            const auto &x_source = validation ? dataset.valX() : dataset.trainX();
            const auto &y_source = validation ? dataset.valY() : dataset.trainY();
            const std::size_t stride = static_cast<std::size_t>(dataset.sequenceLength());

            host_x.resize(count * stride);
            host_y.resize(count * stride);

            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t source_index = order[start + i];
                if (source_index >= total)
                {
                    throw std::out_of_range("copyIndexedBatch: sample index out of bounds");
                }

                const std::size_t source_offset = source_index * stride;
                const std::size_t destination_offset = i * stride;

                std::copy_n(x_source.data() + source_offset, stride, host_x.data() + destination_offset);
                std::copy_n(y_source.data() + source_offset, stride, host_y.data() + destination_offset);
            }
        }
    } // namespace

    EvaluationMetrics evaluate(
        CudaContext &ctx,
        GPT &model,
        CrossEntropyLoss &loss,
        const Dataset &dataset,
        std::size_t batch_size,
        bool validation,
        int32_t ignore_index)
    {
        const std::size_t total = validation ? dataset.valSize() : dataset.trainSize();
        if (total == 0)
        {
            throw std::invalid_argument("evaluate: dataset split is empty");
        }

        if (batch_size == 0)
        {
            throw std::invalid_argument("evaluate: batch_size must be positive");
        }

        std::vector<std::size_t> order(total);
        std::iota(order.begin(), order.end(), 0);

        std::vector<int32_t> host_x;
        std::vector<int32_t> host_y;
        IntTensor predictions;
        DeviceBatch batch;
        Tensor grad_logits;

        double loss_sum = 0.0;
        std::size_t steps = 0;
        std::size_t correct_sequences = 0;

        for (std::size_t start = 0; start < total; start += batch_size)
        {
            const std::size_t count = std::min(batch_size, total - start);
            copyIndexedBatch(dataset, validation, order, start, count, host_x, host_y);

            batch.count = count;
            batch.x.resize({count, dataset.sequenceLength()});
            batch.y.resize({count, dataset.sequenceLength()});
            batch.x.copyFromHost(host_x.data(), ctx);
            batch.y.copyFromHost(host_y.data(), ctx);

            Tensor logits;
            model.forward(batch.x, logits);
            const float batch_loss = loss.forwardBackward(logits, batch.y, grad_logits);
            loss_sum += static_cast<double>(batch_loss);

            const std::size_t rows = logits.shape()[0] * logits.shape()[1];
            predictions.resize(batch.y.shape());
            const int vocab = static_cast<int>(logits.shape()[2]);
            const int blocks = static_cast<int>((rows + kArgmaxThreads - 1) / kArgmaxThreads);

            CUDA_KERNEL_CHECK(argmaxRowsKernel<<<blocks, kArgmaxThreads, 0, ctx.stream()>>>(
                logits.data(), predictions.data(), rows, vocab));

            std::vector<int32_t> host_predictions(predictions.numel());
            predictions.copyToHost(host_predictions.data(), ctx);

            const std::size_t sequence_length = static_cast<std::size_t>(dataset.sequenceLength());
            for (std::size_t i = 0; i < count; ++i)
            {
                bool correct = true;
                for (std::size_t t = 0; t < sequence_length; ++t)
                {
                    const int32_t target = host_y[i * sequence_length + t];
                    if (target != ignore_index && host_predictions[i * sequence_length + t] != target)
                    {
                        correct = false;
                        break;
                    }
                }
                if (correct)
                {
                    ++correct_sequences;
                }
            }
            ++steps;
        }

        if (steps == 0)
        {
            throw std::logic_error("evaluate: no batches processed");
        }

        // Loss is averaged per processed batch, while sequence accuracy is averaged
        // over samples, so the two metrics use intentionally different denominators
        EvaluationMetrics result;
        result.loss = static_cast<float>(loss_sum / static_cast<double>(steps));
        result.sequence_accuracy = static_cast<float>(
            static_cast<double>(correct_sequences) / static_cast<double>(total));
        return result;
    }

    float trainStep(
        CudaContext &ctx,
        GPT &model,
        CrossEntropyLoss &loss,
        AdamW &optimizer,
        const IntTensor &tokens,
        const IntTensor &targets)
    {
        model.zeroGrad(ctx);

        Tensor logits;
        Tensor grad_logits;
        Tensor grad_input;

        model.forward(tokens, logits);
        const float value = loss.forwardBackward(logits, targets, grad_logits);
        model.backward(grad_logits, grad_input);
        optimizer.step();

        return value;
    }

    std::vector<EpochMetrics> trainModel(
        CudaContext &ctx,
        GPT &model,
        CrossEntropyLoss &loss,
        AdamW &optimizer,
        const Dataset &dataset,
        const TrainingConfig &config)
    {
        if (config.epochs == 0)
        {
            throw std::invalid_argument("trainModel: epochs must be positive");
        }

        if (config.batch_size == 0)
        {
            throw std::invalid_argument("trainModel: batch_size must be positive");
        }

        if (dataset.trainSize() == 0)
        {
            throw std::invalid_argument("trainModel: training split is empty");
        }

        std::mt19937_64 rng(config.seed);
        std::vector<std::size_t> order(dataset.trainSize());
        std::iota(order.begin(), order.end(), 0);

        DeviceBatch batch;
        std::vector<int32_t> host_x;
        std::vector<int32_t> host_y;
        std::vector<EpochMetrics> history;
        history.reserve(config.epochs);

        const std::size_t steps_per_epoch =
            (dataset.trainSize() + config.batch_size - 1) / config.batch_size;
        float best_acc = -1.0f;

        for (std::size_t epoch = 1; epoch <= config.epochs; ++epoch)
        {
            std::shuffle(order.begin(), order.end(), rng);

            double train_loss_sum = 0.0;
            for (std::size_t start = 0; start < dataset.trainSize(); start += config.batch_size)
            {
                const std::size_t count = std::min(config.batch_size, dataset.trainSize() - start);
                copyIndexedBatch(dataset, false, order, start, count, host_x, host_y);

                batch.count = count;
                batch.x.resize({count, dataset.sequenceLength()});
                batch.y.resize({count, dataset.sequenceLength()});
                batch.x.copyFromHost(host_x.data(), ctx);
                batch.y.copyFromHost(host_y.data(), ctx);

                const float step_loss = trainStep(ctx, model, loss, optimizer, batch.x, batch.y);
                train_loss_sum += static_cast<double>(step_loss);
            }

            EpochMetrics metrics;
            metrics.epoch = epoch;
            metrics.steps = steps_per_epoch;
            metrics.train_loss = static_cast<float>(train_loss_sum / static_cast<double>(steps_per_epoch));

            if (dataset.valSize() > 0)
            {
                const EvaluationMetrics validation = evaluate(ctx, model, loss, dataset, config.batch_size, true, config.ignore_index);
                metrics.val_loss = validation.loss;
                metrics.val_sequence_accuracy = validation.sequence_accuracy;
                metrics.has_validation = true;

                if (validation.sequence_accuracy > best_acc)
                {
                    best_acc = validation.sequence_accuracy;
                    if (!config.best_checkpoint_path.empty())
                    {
                        ::cugpt::io::saveModel(model, ctx, config.best_checkpoint_path);
                        std::cout << "Saved best checkpoint: " << config.best_checkpoint_path << " (val_acc="
                                  << validation.sequence_accuracy << ")\n";
                    }
                }
            }

            history.push_back(metrics);

            if (config.print_every != 0 &&
                (epoch == 1 || epoch % config.print_every == 0))
            {
                std::cout << "Epoch " << epoch << '/' << config.epochs
                          << " | steps=" << metrics.steps
                          << " | train_loss=" << metrics.train_loss;
                if (metrics.has_validation)
                {
                    std::cout << " | val_loss=" << metrics.val_loss
                              << " | val_acc=" << metrics.val_sequence_accuracy;
                }
                std::cout << '\n';
            }
        }

        ctx.synchronize();

        std::ofstream file("history.csv");

        if (file.is_open())
        {
            file << "step,train_loss,val_loss,val_acc\n";

            for (auto epoch : history)
            {
                file << epoch.epoch << ','
                     << epoch.train_loss << ','
                     << epoch.val_loss << ','
                     << epoch.val_sequence_accuracy << '\n';
            }

            file.close();
        }

        return history;
    }
} // namespace cugpt::training
