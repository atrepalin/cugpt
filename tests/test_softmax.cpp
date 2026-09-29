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

    Softmax softmax(ctx, false);

    Tensor x({2, 4});
    Tensor y;

    x.copyFromHost(hx.data(), ctx);

    softmax.forward(x, y);
    ctx.synchronize();

    expectNear(copyTensor(y, ctx), hy, 1e-5f, "forward");

    std::cout << "softmax: OK\n";
}
