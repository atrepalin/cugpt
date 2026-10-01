#include "core/cuda_context.cuh"
#include "core/cuda_utils.cuh"
#include "nn/layernorm.cuh"
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
        2, 4, 6, 8};

    const std::vector<float> hgamma = {1, 2, 3, 4};
    const std::vector<float> hbeta = {0.5f, -1, 2, 3};

    const std::vector<float> output_ref = {
        -0.841635f, -1.89442f, 3.34164f, 8.36654f,
        -0.841639f, -1.89443f, 3.34164f, 8.36656f};

    const std::vector<float> hdy = {
        1, 2, 3, 4,
        2, 3, 4, 5};

    const std::vector<float> grad_input_ref = {
        0.894371f, -0.894441f, -0.894406f, 0.894477f,
        0.447205f, -0.447216f, -0.447211f, 0.447221f};

    const std::vector<float> grad_gamma_ref = {-4.02491f, -2.23606f, 3.13049f, 12.0747f};

    const std::vector<float> grad_beta_ref = {3, 5, 7, 9};

    LayerNorm ln(ctx, 4, 1e-5f);

    ln.gamma.data.copyFromHost(hgamma.data(), ctx);
    ln.beta.data.copyFromHost(hbeta.data(), ctx);

    Tensor x({2, 4});
    Tensor y;
    Tensor dy({2, 4});
    Tensor dx;

    x.copyFromHost(hx.data(), ctx);
    dy.copyFromHost(hdy.data(), ctx);

    ln.forward(x, y);
    ctx.synchronize();

    expectNear(copyTensor(y, ctx), output_ref, 1e-5f, "forward");

    ln.zeroGrad(ctx);
    ln.backward(dy, dx);
    ctx.synchronize();

    expectNear(copyTensor(dx, ctx), grad_input_ref, 1e-5f, "dX");
    expectNear(copyTensor(ln.gamma.grad, ctx), grad_gamma_ref, 4e-5f, "dGamma");
    expectNear(copyTensor(ln.beta.grad, ctx), grad_beta_ref, 1e-5f, "dBeta");

    std::cout << "layernorm: OK\n";
    return 0;
}