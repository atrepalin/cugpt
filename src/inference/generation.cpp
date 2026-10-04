#include "inference/generation.hpp"

#include "core/cuda_utils.cuh"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace cugpt::inference
{
    using namespace core;
    using namespace nn;
    using namespace data;
    namespace
    {
        int32_t greedyNextToken(CudaContext &ctx, GPT &model, const std::vector<int32_t> &tokens)
        {
            if (tokens.empty())
            {
                throw std::invalid_argument("Generation requires at least one token");
            }

            const std::size_t length = tokens.size();
            const std::size_t vocab = model.vocabSize();

            IntTensor input({1, length});
            input.copyFromHost(tokens.data(), ctx);

            Tensor logits({1, length, vocab});
            model.forward(input, logits);

            std::vector<float> last_logits(vocab);
            const float *last_row = logits.data() + (length - 1) * vocab;

            CUDA_CHECK(cudaMemcpyAsync(last_logits.data(), last_row, last_logits.size() * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
            ctx.synchronize();

            const auto it = std::max_element(last_logits.begin(), last_logits.end());
            return std::distance(last_logits.begin(), it);
        }

    } // namespace

    std::vector<int32_t> generateTokenIds(
        CudaContext &ctx,
        GPT &model,
        const Tokenizer &tokenizer,
        std::string_view text,
        std::size_t max_new_tokens)
    {
        std::string prompt = "<start> ";
        prompt.append(text.data(), text.size());

        std::vector<int32_t> tokens = tokenizer.encode(prompt);
        if (tokens.empty())
        {
            throw std::runtime_error("Encoded generation prompt is empty");
        }

        const int32_t end_id = tokenizer.endId();

        for (std::size_t step = 0; step < max_new_tokens; ++step)
        {
            const int32_t next_token = greedyNextToken(ctx, model, tokens);
            tokens.push_back(next_token);

            if (next_token == end_id)
            {
                break;
            }
        }

        return tokens;
    }

    std::string generateText(
        CudaContext &ctx,
        GPT &model,
        const Tokenizer &tokenizer,
        std::string_view text,
        std::size_t max_new_tokens)
    {
        return tokenizer.decode(generateTokenIds(ctx, model, tokenizer, text, max_new_tokens));
    }
} // namespace cugpt::inference
