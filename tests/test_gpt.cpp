#include "core/cuda_context.cuh"
#include "core/cuda_utils.cuh"
#include "nn/gpt.cuh"
#include "test_utils.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

using namespace cugpt::core;
using namespace cugpt::nn;

namespace
{
    void loadTensor(Tensor &tensor, const std::vector<float> &host, CudaContext &ctx)
    {
        tensor.copyFromHost(host.data(), ctx);
    }

    void loadParameter(Parameter *param, const std::vector<float> &host, CudaContext &ctx)
    {
        param->data.copyFromHost(host.data(), ctx);
    }
} // namespace

int main()
{

    CudaContext ctx;

    // Shapes: B=1, T=2, D=3, H=1, V=2, with one transformer block
    // The block and positional scale are disabled, so only token embeddings
    // reach the final LayerNorm
    // Embeddings: token 0 -> [1,0,-1], token 1 -> [-1,0,1]
    // Final LayerNorm gives +/- c*[1,0,-1], where
    // c = 1/sqrt(2/3 + 1e-5) ~= 1.2247357
    constexpr int B = 1, T = 2, D = 3, H = 1, V = 2, BLOCKS = 1;
    constexpr float c = 1.2247357f;

    GPT model(ctx, V, BLOCKS, D, H, 0.0f);
    auto &params = model.parameters();

    // Parameter order: embedding; 12 parameters for the transformer block;
    // final LayerNorm gamma/beta; final linear weight/bias
    for (Parameter *p : params)
    {
        std::vector<float> zeros(p->data.numel(), 0.0f);
        loadParameter(p, zeros, ctx);
    }

    // embedding.weights: rows 0 and 1 are the two hand-written tokens
    loadParameter(params[0], {1, 0, -1, -1, 0, 1, 0, 0, 0}, ctx);

    // Disable the transformer block while keeping both LayerNorm scales at 1
    loadParameter(params[1], {1, 1, 1}, ctx); // norm1.gamma
    loadParameter(params[7], {1, 1, 1}, ctx); // norm2.gamma

    // Final LayerNorm: gamma=1, beta=0
    loadParameter(params[13], {1, 1, 1}, ctx);
    loadParameter(params[14], {0, 0, 0}, ctx);

    // Final linear weight (D x V):
    // [[1,2], [0,0], [0,-1]]
    loadParameter(params[15], {0, 0}, ctx);
    loadParameter(params[16], {1, 2, 0, 0, 0, -1}, ctx);
    ctx.synchronize();

    const std::vector<int32_t> tokenHost = {0, 1};
    IntTensor tokens({B, T});
    tokens.copyFromHost(tokenHost.data(), ctx);

    Tensor logits;
    model.forward(tokens, logits);
    ctx.synchronize();

    // The final linear projection maps the normalized rows to
    // [c, 3c, -c, -3c]
    expectNear(copyTensor(logits, ctx),
               {1.2247357f, 3.6742071f,
                -1.2247357f, -3.6742071f},
               1e-5f, "forward");

    // Seed one logit per token position: token 0 reads logit 0,
    // token 1 reads logit 1
    Tensor gradLogits({B, T, V});
    const std::vector<float> gy = {1, 0, 0, 1};
    gradLogits.copyFromHost(gy.data(), ctx);

    model.zeroGrad(ctx);
    model.forward(tokens, logits);
    Tensor gradEmbedding;
    model.backward(gradLogits, gradEmbedding);
    ctx.synchronize();

    expectNear(copyTensor(params[16]->grad, ctx),
               {c, -c,
                0, 0,
                -c, c},
               1e-5f, "dW");

    expectNear(copyTensor(params[15]->grad, ctx), {1, 1}, 1e-5f, "dB");

    expectNear(copyTensor(params[13]->grad, ctx), {-c, 0, -c}, 1e-5f, "dGamma");
    expectNear(copyTensor(params[14]->grad, ctx), {3, 0, -1}, 1e-5f, "dBeta");

    expectNear(copyTensor(gradEmbedding, ctx),
               {0.2041318f, -0.4082452f, 0.2041134f,
                0.2041502f, -0.4082452f, 0.2040951f},
               1e-5f, "dEmbeddingX");

    // The two token IDs are unique, so each embedding row receives exactly
    // one gradient contribution from the corresponding sequence position
    expectNear(copyTensor(params[0]->grad, ctx),
               {0.2041318f, -0.4082452f, 0.2041134f,
                0.2041502f, -0.4082452f, 0.2040951f},
               1e-5f, "dEmbedding");

    std::cout << "gpt: OK\n";
    return 0;
}