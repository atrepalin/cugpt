#include "core/cuda_context.cuh"
#include "core/cuda_utils.cuh"
#include "nn/transformer_block.cuh"
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
} // namespace

int main()
{
    CudaContext ctx;

    // Shapes: B=1, T=1, D=2, H=1, F=2
    // Input x=[1,-1]. With eps=1e-5, LayerNorm gives
    //   mean=0, var=1, a=1/sqrt(1+eps) ~= 0.999995
    // Zero attention projections make the attention branch contribute nothing
    // The FFN uses W1=I, b1=0, ReLU([a,-a])=[a,0], W2=I, b2=0
    // Therefore the block output is y = x + [a,0] = [1+a,-1]
    constexpr int B = 1, T = 1, D = 2, H = 1, F = 2;
    const float a = 0.999995f;
    const float ln_delta = 4.999925e-6f;

    TransformerBlock block(ctx, D, H, F);

    const std::vector<float> zero22 = {0, 0, 0, 0};
    const std::vector<float> eye22 = {1, 0, 0, 1};
    const std::vector<float> zero2 = {0, 0};
    const std::vector<float> one2 = {1, 1};
    const std::vector<float> gradOut = {2, -3};
    const std::vector<float> x = {1, -1};

    loadTensor(block.firstNorm().gamma.data, one2, ctx);
    loadTensor(block.firstNorm().beta.data, zero2, ctx);
    loadTensor(block.secondNorm().gamma.data, one2, ctx);
    loadTensor(block.secondNorm().beta.data, zero2, ctx);

    loadTensor(block.attention().queryLinear().weights.data, zero22, ctx);
    loadTensor(block.attention().keyLinear().weights.data, zero22, ctx);
    loadTensor(block.attention().valueLinear().weights.data, zero22, ctx);
    loadTensor(block.attention().outputLinear().weights.data, zero22, ctx);

    loadTensor(block.feedForward().firstLinear().weights.data, eye22, ctx);
    loadTensor(block.feedForward().firstLinear().bias()->data, zero2, ctx);
    loadTensor(block.feedForward().secondLinear().weights.data, eye22, ctx);
    loadTensor(block.feedForward().secondLinear().bias()->data, zero2, ctx);

    Tensor input({B, T, D}), output, gy({B, T, D}), gx;
    loadTensor(input, x, ctx);
    loadTensor(gy, gradOut, ctx);

    block.forward(input, output);
    ctx.synchronize();

    expectNear(copyTensor(output, ctx), {1.999995f, -1.0f}, 1e-5f, "forward");

    block.zeroGrad(ctx);
    block.backward(gy, gx);
    ctx.synchronize();

    // Backward decomposition:
    // the FFN contributes a tiny LayerNorm derivative
    //   dx_LN ~= [+4.999925e-6, -4.999925e-6],
    // while the outer residual contributes [2, -3]
    expectNear(copyTensor(gx, ctx), {2.000005f, -3.000005f}, 1e-5f, "dX");

    // For the second FFN linear layer, dW2 = ReLU(n2)^T * dY
    // = [[2a, -3a],[0, 0]]
    expectNear(copyTensor(block.feedForward().secondLinear().weights.grad, ctx),
               {1.99999f, -2.999985f,
                0.0f, 0.0f},
               1e-5f, "dW2");

    // The second-layer bias receives the output gradient directly: db2 = dY
    expectNear(copyTensor(block.feedForward().secondLinear().bias()->grad, ctx), {2.0f, -3.0f}, 1e-5f, "dB2");

    // ReLU keeps only the first hidden unit, so dW1 = n2^T * [2,0]
    // = [[2a, -2a],[0, 0]]
    expectNear(copyTensor(block.feedForward().firstLinear().weights.grad, ctx),
               {1.99999f, 0.0f,
                -1.99999f, 0.0f},
               1e-5f, "dW1");

    // After ReLU backward, the hidden-layer bias gradient is db1 = [2, 0]
    expectNear(copyTensor(block.feedForward().firstLinear().bias()->grad, ctx),
               {2.0f, 0.0f},
               1e-5f, "dB1");

    // The second LayerNorm sees the FFN input after the first residual addition
    expectNear(copyTensor(block.secondNorm().gamma.grad, ctx),
               {1.99999f, 0.0f},
               1e-5f, "dGamma1");

    expectNear(copyTensor(block.secondNorm().beta.grad, ctx),
               {2.0f, 0.0f},
               1e-5f, "dBeta1");

    // With zero attention-output weights, the attention branch receives zero gradient
    // from this backward pass, so all four attention weight gradients stay zero.
    expectNear(copyTensor(block.attention().queryLinear().weights.grad, ctx),
               {0, 0, 0, 0},
               1e-5f, "dWq");

    expectNear(copyTensor(block.attention().keyLinear().weights.grad, ctx),
               {0, 0, 0, 0},
               1e-5f, "dWk");

    expectNear(copyTensor(block.attention().valueLinear().weights.grad, ctx),
               {0, 0, 0, 0},
               1e-5f, "dWv");

    expectNear(copyTensor(block.attention().outputLinear().weights.grad, ctx),
               {0, 0, 0, 0},
               1e-5f, "dWo");

    std::cout << "transformer_block: OK\n";
    return 0;
}
