#include "core/cuda_context.cuh"
#include "core/cuda_utils.cuh"
#include "training/cross_entropy.cuh"
#include "test_utils.hpp"

#include <cmath>
#include <iostream>

using namespace cugpt::core;
using namespace cugpt::training;

int main()
{
    CudaContext ctx;

    CrossEntropyLoss loss(ctx, 0);

    Tensor logits({2, 4});
    IntTensor targets({2});
    Tensor grad;

    const std::vector<float> hlogits = {
        1, 2, 3, 4,
        4, 3, 2, 1};
    const std::vector<int32_t> htargets = {3, 1};
    const std::vector<float> grad_ref = {
        0.0160293f, 0.0435722f, 0.118441f, -0.178043f,
        0.321957f, -0.381559f, 0.0435722f, 0.0160293f};
    const float loss_red = 0.94019f;

    logits.copyFromHost(hlogits.data(), ctx);
    targets.copyFromHost(htargets.data(), ctx);

    const float got_loss = loss.forwardBackward(logits, targets, grad);

    expectNear(copyTensor(grad, ctx), grad_ref, 1e-5f, "grad");
    expectNear(got_loss, loss_red, 1e-5, "loss");

    std::cout << "loss: OK\n";
    return 0;
}
