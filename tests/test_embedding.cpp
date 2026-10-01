#include "core/cuda_context.cuh"
#include "core/cuda_utils.cuh"
#include "nn/embedding.cuh"
#include "test_utils.hpp"

#include <iostream>

using namespace cugpt::core;
using namespace cugpt::nn;

int main()
{
    CudaContext ctx;

    // Token 2 appears three times, which also tests gradient accumulation into
    // a single embedding row during the backward pass
    const std::vector<int32_t> ht = {0, 2, 1, 2, 2, 3};

    // A unit upstream gradient means every selected embedding row receives
    // one contribution per occurrence; row 2 therefore accumulates three times
    const std::vector<float> hgrad(18, 1.0f);

    const std::vector<float> table = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12};

    const std::vector<float> output_ref = {
        1, 2, 3,
        7, 8, 9,
        4, 5, 6,
        7, 8, 9,
        7, 8, 9,
        10, 11, 12};

    const std::vector<float> grad_ref = {
        1, 1, 1,
        1, 1, 1,
        3, 3, 3,
        1, 1, 1};

    Embedding embedding(ctx, 4, 3);

    embedding.embeddings.data.copyFromHost(table.data(), ctx);

    // 2 batches of 3 tokens each
    IntTensor tokens({2, 3});
    Tensor grad{{2, 3, 3}};

    tokens.copyFromHost(ht.data(), ctx);
    grad.copyFromHost(hgrad.data(), ctx);

    Tensor output;
    embedding.forward(tokens, output);
    ctx.synchronize();

    expectNear(copyTensor(output, ctx), output_ref, 1e-5f, "forward");

    embedding.zeroGrad(ctx);
    embedding.backward(grad);
    ctx.synchronize();

    expectNear(copyTensor(embedding.embeddings.grad, ctx), grad_ref, 1e-5f, "backward");

    std::cout << "embedding: OK\n";
    return 0;
}
