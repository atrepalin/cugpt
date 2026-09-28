#include "core/cuda_context.cuh"
#include "core/cuda_utils.cuh"
#include "nn/linear.cuh"
#include "test_utils.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace cugpt::core;
using namespace cugpt::nn;

// Forward values and backward gradients are checked against python-computed matrix expressions
int main()
{
    CudaContext ctx;

    Linear layer(ctx, 3, 2, true);
    const std::vector<float> w = {1, 2, 3, 4, 5, 6};
    const std::vector<float> b = {0.5f, -1.0f};
    const std::vector<float> x = {1, 2, 3, 4, 5, 6};
    const std::vector<float> gy = {0.5f, 1.0f, -2.0f, 3.0f};

    layer.weights.data.copyFromHost(w.data(), ctx);
    layer.bias()->data.copyFromHost(b.data(), ctx);

    Tensor tx({2, 3});
    Tensor ty({2, 2});
    Tensor tgy({2, 2});
    Tensor tgx({2, 3});

    tx.copyFromHost(x.data(), ctx);
    tgy.copyFromHost(gy.data(), ctx);

    layer.forward(tx, ty);
    ctx.synchronize();

    std::vector<float> y(4);
    ty.copyToHost(y.data(), ctx);
    expectNear(y, {22.5f, 27.0f, 49.5f, 63.0f}, 1e-5f, "forward");

    layer.zeroGrad(ctx);
    layer.backward(tgy, tgx);
    ctx.synchronize();

    std::vector<float> gx(6), gw(6), gb(2);
    tgx.copyToHost(gx.data(), ctx);
    layer.weights.grad.copyToHost(gw.data(), ctx);
    layer.bias()->grad.copyToHost(gb.data(), ctx);

    expectNear(gx, {2.5f, 5.5f, 8.5f, 4.0f, 6.0f, 8.0f}, 1e-5f, "dX");
    expectNear(gw, {-7.5f, 13.0f, -9.0f, 17.0f, -10.5f, 21.0f}, 1e-5f, "dW");
    expectNear(gb, {-1.5f, 4.0f}, 1e-5f, "db");

    std::cout << "linear: OK\n";
    return 0;
}
