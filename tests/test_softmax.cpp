#include "core/cuda_context.cuh"
#include "core/cuda_utils.cuh"
#include "nn/softmax.cuh"
#include "test_utils.hpp"

#include <cmath>
#include <iostream>

using namespace cugpt::core;
using namespace cugpt::nn;

int main()
{
    CudaContext ctx;

    const std::vector<float> hx = {
        1, 2, 3, 4,
        0, 0, 0, 0};
    const std::vector<float> hy = {
        0.0320586f, 0.0871443f, 0.236883f, 0.643914f,
        0.25f, 0.25f, 0.25f, 0.25f};

    const std::vector<float> hdy = {
        1, -1, 2, 3,
        0.5f, 1, -2, 4};
    const std::vector<float> hdx = {
        -0.0432927f, -0.29197f, -0.0830091f, 0.418272f,
        -0.09375f, 0.03125f, -0.71875f, 0.78125f};

    Softmax softmax(ctx, false);

    Tensor x({2, 4});
    Tensor y;
    Tensor dy({2, 4});
    Tensor dx;

    x.copyFromHost(hx.data(), ctx);
    dy.copyFromHost(hdy.data(), ctx);

    softmax.forward(x, y);
    ctx.synchronize();

    expectNear(copyTensor(y, ctx), hy, 1e-5f, "forward");

    softmax.zeroGrad(ctx);
    softmax.backward(dy, dx);
    ctx.synchronize();

    expectNear(copyTensor(dx, ctx), hdx, 1e-5f, "backward");

    std::cout << "softmax: OK\n";
}
