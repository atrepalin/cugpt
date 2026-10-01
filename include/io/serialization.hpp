#pragma once

#include "nn/gpt.cuh"

#include <string>

namespace cugpt::io
{
    using namespace core;
    using namespace nn;

    void saveModel(const GPT &model, const CudaContext &ctx, const std::string &path);

    void loadModel(GPT &model, const CudaContext &ctx, const std::string &path);
} // namespace cugpt::io
