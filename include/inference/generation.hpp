#pragma once

#include "data/dataset.hpp"
#include "nn/gpt.cuh"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cugpt::inference
{
    using namespace core;
    using namespace nn;
    using namespace data;

    std::vector<int32_t> generateTokenIds(
        CudaContext &ctx,
        GPT &model,
        const MathTokenizer &tokenizer,
        std::string_view text,
        std::size_t max_new_tokens = 50);

    std::string generateText(
        CudaContext &ctx,
        GPT &model,
        const MathTokenizer &tokenizer,
        std::string_view text,
        std::size_t max_new_tokens = 50);
} // namespace cugpt::inference
