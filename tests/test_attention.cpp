#include "core/cuda_context.cuh"
#include "core/cuda_utils.cuh"
#include "nn/attention.cuh"
#include "test_utils.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

using namespace cugpt::core;
using namespace cugpt::nn;

int main()
{
    CudaContext ctx;

    constexpr int B = 1, T = 2, D = 2, H = 1;

    const std::vector<float> x = {1, 2, 3, 4};
    const std::vector<float> zero2x2 = {0, 0, 0, 0};
    const std::vector<float> identity2 = {1, 0, 0, 1};
    const std::vector<float> gradOut = {1, 1, 1, 1};

    MultiHeadAttention attn(ctx, D, H);

    attn.queryLinear().weights.data.copyFromHost(zero2x2.data(), ctx);
    attn.keyLinear().weights.data.copyFromHost(zero2x2.data(), ctx);
    attn.valueLinear().weights.data.copyFromHost(identity2.data(), ctx);
    attn.outputLinear().weights.data.copyFromHost(identity2.data(), ctx);
    Tensor input({B, T, D}), output, gy({B, T, D}), gx;

    input.copyFromHost(x.data(), ctx);
    gy.copyFromHost(gradOut.data(), ctx);

    attn.forward(input, output);
    ctx.synchronize();

    expectNear(copyTensor(output, ctx), {1.0f, 2.0f, 2.0f, 3.0f}, 1e-5f, "forward");

    attn.zeroGrad(ctx);
    attn.backward(gy, gx);
    ctx.synchronize();

    expectNear(copyTensor(gx, ctx), {1.5f, 1.5f, 0.5f, 0.5f}, 1e-5f, "dX");

    expectNear(copyTensor(attn.valueLinear().weights.grad, ctx),
               {3.0f, 3.0f,
                5.0f, 5.0f},
               1e-5f, "dWv");

    expectNear(copyTensor(attn.outputLinear().weights.grad, ctx),
               {3.0f, 3.0f,
                5.0f, 5.0f},
               1e-5f, "dWp");

    expectNear(copyTensor(attn.queryLinear().weights.grad, ctx),
               {0.0f, 0.0f, 0.0f, 0.0f},
               1e-5f, "dWq");

    expectNear(copyTensor(attn.keyLinear().weights.grad, ctx),
               {0.0f, 0.0f, 0.0f, 0.0f},
               1e-5f, "fWk");

    std::cout << "attention: OK\n";
    return 0;
}
