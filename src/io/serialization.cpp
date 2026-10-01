#include "io/serialization.hpp"
#include "io/safetensors.hpp"

#include <charconv>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <string>
#include <unordered_set>
#include <vector>

namespace cugpt::io
{
    using namespace core;
    using namespace nn;

    namespace
    {
        constexpr std::string_view kFormatVersion = "1";
        constexpr std::string_view kFormatName = "cugpt";

        constexpr std::string_view kMetaFormat = "cugpt.format";
        constexpr std::string_view kMetaVocabSize = "cugpt.vocab_size";
        constexpr std::string_view kMetaBlocks = "cugpt.blocks";
        constexpr std::string_view kMetaModelDim = "cugpt.model_dim";
        constexpr std::string_view kMetaNumHeads = "cugpt.n_heads";
        constexpr std::string_view kMetaPositionalScale = "cugpt.positional_scale";

        std::string int64String(const std::int64_t value)
        {
            return std::to_string(value);
        }

        std::string floatString(const float value)
        {
            std::ostringstream stream;
            stream << value;
            return stream.str();
        }

        std::int64_t parseInt64(const std::string &value, const char *name)
        {
            std::int64_t result = 0;
            const char *first = value.data();
            const char *last = value.data() + value.size();
            const auto parsed = std::from_chars(first, last, result);

            if (parsed.ec != std::errc{} || parsed.ptr != last)
            {
                throw std::runtime_error("Invalid safetensors metadata '" + std::string(name) + "': " + value);
            }

            return result;
        }

        float parseFloat(const std::string &value, const char *name)
        {
            try
            {
                std::size_t consumed = 0;
                const float result = std::stof(value, &consumed);

                if (consumed != value.size())
                {
                    throw std::invalid_argument("trailing characters");
                }

                return result;
            }
            catch (const std::exception &)
            {
                throw std::runtime_error("Invalid safetensors metadata '" + std::string(name) + "': " + value);
            }
        }

        std::vector<std::size_t> toSafeTensorShape(const Shape &shape)
        {
            std::vector<std::size_t> result;
            result.reserve(shape.size());
            for (const std::int64_t dim : shape)
            {
                if (dim <= 0)
                {
                    throw std::runtime_error("Cannot serialize tensor with non-positive dimension");
                }

                result.push_back(static_cast<std::size_t>(dim));
            }
            return result;
        }

        void validateArchitecture(const SafeTensorsReader &reader, const GPT &model)
        {
            if (reader.metadata(kMetaFormat) != std::string(kFormatName) + ":" + std::string(kFormatVersion))
            {
                throw std::runtime_error("Unsupported cugpt safetensors format");
            }

            if (parseInt64(reader.metadata(kMetaVocabSize), "cugpt.vocab_size") != model.vocabSize() ||
                parseInt64(reader.metadata(kMetaBlocks), "cugpt.blocks") != model.numBlocks() ||
                parseInt64(reader.metadata(kMetaModelDim), "cugpt.dim_model") != model.modelDim() ||
                parseInt64(reader.metadata(kMetaNumHeads), "cugpt.num_heads") != model.numHeads() ||
                parseFloat(reader.metadata(kMetaPositionalScale), "cugpt.positional_scale") != model.positionalScale())
            {
                throw std::runtime_error("Model checkpoint architecture does not match target model");
            }
        }

        std::unordered_set<std::string> expectedNames(const GPT &model)
        {
            std::unordered_set<std::string> result;
            for (const auto &[name, parameter] : model.namedParameters())
            {
                if (parameter == nullptr)
                {
                    throw std::runtime_error("Model contains a null parameter: " + name);
                }

                if (!result.insert(name).second)
                {
                    throw std::runtime_error("Duplicate model parameter name: " + name);
                }
            }
            return result;
        }

        void validateTensorNames(const SafeTensorsReader &reader, const GPT &model)
        {
            const auto expected = expectedNames(model);
            const auto actual = reader.tensorNames();

            std::vector<std::string> missing;
            std::vector<std::string> unexpected;

            for (const std::string &name : expected)
            {
                if (!reader.contains(name))
                {
                    missing.push_back(name);
                }
            }

            for (const std::string &name : actual)
            {
                if (expected.find(name) == expected.end())
                {
                    unexpected.push_back(name);
                }
            }

            if (!missing.empty() || !unexpected.empty())
            {
                std::ostringstream message;
                message << "Model checkpoint tensor names do not match target model";

                if (!missing.empty())
                {
                    message << "\nMissing keys:";
                    for (const auto &name : missing)
                    {
                        message << "\n  " << name;
                    }
                }

                if (!unexpected.empty())
                {
                    message << "\nUnexpected keys:";
                    for (const auto &name : unexpected)
                    {
                        message << "\n  " << name;
                    }
                }
                throw std::runtime_error(message.str());
            }
        }
    } // namespace

    void saveModel(const GPT &model, const CudaContext &ctx, const std::string &path)
    {
        SafeTensorsWriter writer;
        std::vector<float> host;

        writer.addMetadata(kMetaFormat, std::string(kFormatName) + ":" + std::string(kFormatVersion));
        writer.addMetadata(kMetaVocabSize, int64String(model.vocabSize()));
        writer.addMetadata(kMetaBlocks, int64String(model.numBlocks()));
        writer.addMetadata(kMetaModelDim, int64String(model.modelDim()));
        writer.addMetadata(kMetaNumHeads, int64String(model.numHeads()));
        writer.addMetadata(kMetaPositionalScale, floatString(model.positionalScale()));

        for (const auto &[name, parameter] : model.namedParameters())
        {
            if (parameter == nullptr)
            {
                throw std::runtime_error("Cannot save null model parameter: " + name);
            }

            host.resize(parameter->data.numel());
            parameter->data.copyToHost(host.data(), ctx);

            writer.add(name, host.data(), host.size(),
                       toSafeTensorShape(parameter->data.shape()));
        }

        writer.save(path);
    }

    void loadModel(GPT &model, const CudaContext &ctx, const std::string &path)
    {
        SafeTensorsReader reader(path);
        validateArchitecture(reader, model);
        validateTensorNames(reader, model);

        for (const auto &[name, parameter] : model.namedParameters())
        {
            if (parameter == nullptr)
            {
                throw std::runtime_error("Cannot load into null model parameter: " + name);
            }

            const Shape &shape = parameter->data.shape();
            const std::vector<std::size_t> safeShape = toSafeTensorShape(shape);
            std::vector<float> host(parameter->data.numel());

            reader.read(name, host.data(), host.size(), safeShape);
            parameter->data.copyFromHost(host.data(), ctx);
        }

        ctx.synchronize();
    }
} // namespace cugpt::io
