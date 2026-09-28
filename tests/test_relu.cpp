#include "core/cuda_context.cuh"
#include "core/cuda_utils.cuh"
#include "nn/relu.cuh"
#include "test_utils.hpp"

#include <iostream>

using namespace cugpt::core;
using namespace cugpt::nn;

int main()
{
    CudaContext ctx;

    const std::vector<float> hx = {-2, -1, 0, 1, 2, 3, 4, -5, 0, -0.5f};
    const std::vector<float> hdy = {1, 1, 1, 2, 3, -1, 2, 3, 4, 5};
    const std::vector<float> hy = {0, 0, 0, 1, 2, 3, 4, 0, 0, 0};
    const std::vector<float> hdx = {0, 0, 0, 2, 3, -1, 2, 0, 0, 0};

    ReLU relu(ctx);

    Tensor x({2, 5});
    Tensor y;
    Tensor dy({2, 5});
    Tensor dx;

    x.copyFromHost(hx.data(), ctx);
    dy.copyFromHost(hdy.data(), ctx);

    relu.forward(x, y);
    ctx.synchronize();

    expectNear(copyTensor(y, ctx), hy, 1e-5f, "forward");

    relu.zeroGrad(ctx);
    relu.backward(dy, dx);
    ctx.synchronize();

    expectNear(copyTensor(dx, ctx), hdx, 1e-5f, "backward");

    std::cout << "relu: OK\n";
    return 0;
}
