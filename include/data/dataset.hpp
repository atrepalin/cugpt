#pragma once

#include "core/tensor.cuh"

#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace cugpt::data
{
    using namespace core;

    class Tokenizer
    {
    public:
        virtual int32_t tokenId(std::string_view token) const = 0;
        virtual const std::string &token(int32_t id) const = 0;
        virtual std::vector<int32_t> encode(std::string_view text) const = 0;
        virtual std::string decode(const std::vector<int32_t> &ids) const = 0;

        const std::vector<std::string> &vocabulary() const noexcept { return vocabulary_; }
        const std::size_t vocabularySize() const noexcept { return vocabulary_.size(); }
        virtual int32_t padId() const noexcept = 0;

    protected:
        std::vector<std::string> vocabulary_;
    };

    class Dataset
    {
    public:
        virtual const Tokenizer &tokenizer() const noexcept = 0;
        virtual std::size_t sequenceLength() const noexcept = 0;
        virtual std::size_t trainSize() const noexcept = 0;
        virtual std::size_t valSize() const noexcept = 0;

        const std::vector<int32_t> &trainX() const noexcept { return train_x_; }
        const std::vector<int32_t> &trainY() const noexcept { return train_y_; }
        const std::vector<int32_t> &valX() const noexcept { return val_x_; }
        const std::vector<int32_t> &valY() const noexcept { return val_y_; }

        virtual std::vector<int32_t> makeBatchX(bool validation, std::size_t start, std::size_t count) const = 0;
        virtual std::vector<int32_t> makeBatchY(bool validation, std::size_t start, std::size_t count) const = 0;

    protected:
        std::vector<int32_t> train_x_;
        std::vector<int32_t> train_y_;
        std::vector<int32_t> val_x_;
        std::vector<int32_t> val_y_;
    };

    class MathTokenizer : public Tokenizer
    {
    public:
        MathTokenizer(const std::vector<std::string> &operations);

        int32_t tokenId(std::string_view token) const override;
        const std::string &token(int32_t id) const override;
        std::vector<int32_t> encode(std::string_view text) const override;
        std::string decode(const std::vector<int32_t> &ids) const override;

        int32_t padId() const noexcept override { return tokenId("<pad>"); }
        int32_t startId() const noexcept { return tokenId("<start>"); }
        int32_t endId() const noexcept { return tokenId("<end>"); }
        int32_t equalsId() const noexcept { return tokenId("равно"); }

    private:
        std::vector<std::string> token_patterns_;
    };

    class MathDataset : public Dataset
    {
    public:
        struct Config
        {
            int max_number = 999;
            std::vector<std::string> operations{
                "plus",
                "minus",
                "multiply",
                "divide"};
            std::uint64_t seed = 42;
            double val_fraction = 0.2;
            std::size_t examples_per_operation = 20'000;
        };

        MathDataset(std::vector<std::string> operations);
        static MathDataset generate(const Config &config = {});

        const Tokenizer &tokenizer() const noexcept override { return tokenizer_; }
        std::size_t sequenceLength() const noexcept override { return sequence_length_; }
        std::size_t trainSize() const noexcept override { return train_x_.size() / sequence_length_; }
        std::size_t valSize() const noexcept override { return val_x_.size() / sequence_length_; }

        std::vector<int32_t> makeBatchX(bool validation, std::size_t start, std::size_t count) const override;
        std::vector<int32_t> makeBatchY(bool validation, std::size_t start, std::size_t count) const override;

    private:
        MathTokenizer tokenizer_;
        std::size_t sequence_length_;
    };

    struct DeviceBatch
    {
        IntTensor x;
        IntTensor y;
        std::size_t count = 0;
    };

    void copyBatchToDevice(
        CudaContext &ctx,
        const MathDataset &dataset,
        bool validation,
        std::size_t start,
        std::size_t batch_size,
        DeviceBatch &batch);

    std::vector<std::string> numberToTokens(int value);
    std::string makeExpression(int a, int b, std::string_view operation);
} // namespace cugpt::data
