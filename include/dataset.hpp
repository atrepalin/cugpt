#pragma once

#include "data/dataset.hpp"

using namespace cugpt::data;

class MathTokenizer : public Tokenizer
{
public:
    MathTokenizer(const std::vector<std::string> &operations);

    int32_t tokenId(std::string_view token) const override;
    const std::string &token(int32_t id) const override;
    std::vector<int32_t> encode(std::string_view text) const override;
    std::string decode(const std::vector<int32_t> &ids) const override;

    int32_t padId() const noexcept override { return tokenId("<pad>"); }
    int32_t startId() const noexcept override { return tokenId("<start>"); }
    int32_t endId() const noexcept override { return tokenId("<end>"); }
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

std::vector<std::string> numberToTokens(int value);
std::string makeExpression(int a, int b, std::string_view operation);