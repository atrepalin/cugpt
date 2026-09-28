#include "core/cuda_context.cuh"
#include "core/cuda_utils.cuh"
#include "nn/feed_forward.cuh"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace cugpt::core;
using namespace cugpt::nn;

int main()
{
    CudaContext ctx;

    constexpr int rows = 2;
    constexpr int din = 3;
    constexpr int dh = 4;
    constexpr int dout = 2;

    const std::vector<float> x = {
        1.0f, -2.0f, 3.0f,
        -1.0f, 2.0f, -3.0f};

    const std::vector<float> w1 = {
        1.0f, 0.0f, -1.0f, 2.0f,
        2.0f, 1.0f, 0.0f, -1.0f,
        -1.0f, 2.0f, 1.0f, 0.5f};
    const std::vector<float> b1 = {0.5f, -1.0f, 1.0f, -0.5f};

    const std::vector<float> w2 = {
        1.0f, -1.0f,
        2.0f, 0.5f,
        -1.0f, 1.0f,
        0.5f, 2.0f};
    const std::vector<float> b2 = {0.25f, -0.5f};

    const std::vector<float> gy = {
        1.0f, -2.0f,
        0.5f, 1.0f};

    const std::vector<float> output_ref = {
        5.75f, 14.0f,
        6.75f, -7.0f};

    const std::vector<float> grad_input_ref = {
        -4.0f, 4.5f, -2.75f,
        -0.5f, -1.0f, 0.5f};

    const std::vector<float> grad_w1_ref = {
        0.5f, 1.0f, -3.0f, -3.5f,
        -1.0f, -2.0f, 6.0f, 7.0f,
        1.5f, 3.0f, -9.0f, -10.5f};
    const std::vector<float> grad_b1_ref = {-0.5f, 1.0f, -3.0f, -3.5f};

    const std::vector<float> grad_w2_ref = {
        3.25f, 6.5f,
        3.0f, -6.0f,
        3.0f, -6.0f,
        5.0f, -10.0f};
    const std::vector<float> grad_b2_ref = {1.5f, -1.0f};

    FeedForwardNetwork ffn(ctx, din, dh, dout);

    ffn.firstLinear().weights.data.copyFromHost(w1.data(), ctx);
    ffn.firstLinear().bias()->data.copyFromHost(b1.data(), ctx);
    ffn.secondLinear().weights.data.copyFromHost(w2.data(), ctx);
    ffn.secondLinear().bias()->data.copyFromHost(b2.data(), ctx);

    Tensor input({rows, din});
    Tensor output;
    Tensor gradOutput({rows, dout});
    Tensor gradInput;

    input.copyFromHost(x.data(), ctx);
    gradOutput.copyFromHost(gy.data(), ctx);

    ffn.zeroGrad(ctx);
    ffn.forward(input, output);
    ctx.synchronize();

    expectNear(copyTensor(output, ctx), output_ref, 1e-5f, "forward");

    ffn.backward(gradOutput, gradInput);
    ctx.synchronize();

    expectNear(copyTensor(gradInput, ctx), grad_input_ref, 1e-5f, "dX");
    expectNear(copyTensor(ffn.firstLinear().weights.grad, ctx), grad_w1_ref, 1e-5f, "dW1");
    expectNear(copyTensor(ffn.firstLinear().bias()->grad, ctx), grad_b1_ref, 1e-5f, "db1");
    expectNear(copyTensor(ffn.secondLinear().weights.grad, ctx), grad_w2_ref, 1e-5f, "dW2");
    expectNear(copyTensor(ffn.secondLinear().bias()->grad, ctx), grad_b2_ref, 1e-5f, "db2");

    std::cout << "feed_forward: OK\n";
    return 0;
}
