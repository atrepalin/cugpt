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

    Tensor x({2, 5});
    Tensor y;
    Tensor dy({2, 5});
    Tensor dx;

    const std::vector<float> hx = {-2, -1, 0, 1, 2, 3, 4, -5, 0, -0.5f};
    const std::vector<float> hdy = {1, 1, 1, 2, 3, -1, 2, 3, 4, 5};
    const std::vector<float> hy = {0, 0, 0, 1, 2, 3, 4, 0, 0, 0};
    const std::vector<float> hdx = {0, 0, 0, 2, 3, -1, 2, 0, 0, 0};

    x.copyFromHost(hx.data(), ctx);
    dy.copyFromHost(hdy.data(), ctx);

    ReLU relu(ctx);
    relu.forward(x, y);
    relu.backward(dy, dx);
    ctx.synchronize();

    std::vector<float> got_y(hy.size()), got_dx(hdx.size());
    y.copyToHost(got_y.data(), ctx);
    dx.copyToHost(got_dx.data(), ctx);
    ctx.synchronize();

    expectNear(got_y, hy, 1e-5f, "forward");
    expectNear(got_dx, hdx, 1e-5f, "backward");

    std::cout << "relu: OK\n";
    return 0;
}
