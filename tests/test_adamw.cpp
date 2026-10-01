#include "core/cuda_context.cuh"
#include "core/cuda_utils.cuh"
#include "training/adamw.cuh"
#include "test_utils.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace cugpt::core;
using namespace cugpt::training;

int main()
{
    CudaContext ctx;

    // Use a single five-element parameter so the optimizer state is easy to
    // compare against an independent host-side AdamW reference update
    Parameter parameter(Shape{5});
    const std::vector<float> initial = {0.25f, -0.5f, 1.25f, 0.0f, -1.0f};
    parameter.data.copyFromHost(initial.data(), ctx);

    AdamW optimizer(
        ctx,
        std::vector<Parameter *>{&parameter},
        3.0e-4f,
        0.9f,
        0.999f,
        1.0e-8f,
        0.01f);

    // Three deliberately different gradients exercise both moving averages
    // and the decoupled weight-decay term across multiple optimizer steps
    const std::vector<std::vector<float>> grads = {
        {0.1f, -0.2f, 0.3f, 0.4f, -0.5f},
        {0.2f, 0.1f, -0.4f, 0.0f, 0.25f},
        {-0.3f, 0.15f, 0.2f, -0.1f, 0.05f}};

    // Reproduce AdamW on the host: first moments, second moments,
    // bias correction, gradient update, then decoupled weight decay
    std::vector<float> expected = initial;
    std::vector<float> m(5, 0.0f);
    std::vector<float> v(5, 0.0f);

    constexpr float lr = 3.0e-4f;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float eps = 1.0e-8f;
    constexpr float weight_decay = 0.01f;

    for (std::size_t step = 0; step < grads.size(); ++step)
    {
        // The CUDA optimizer and the reference update see exactly the same
        // gradient at every step; their parameter vectors must stay close
        parameter.grad.copyFromHost(grads[step].data(), ctx);
        optimizer.step();
        ctx.synchronize();

        const float b1corr = 1.0f - std::pow(beta1, static_cast<float>(step + 1));
        const float b2corr = 1.0f - std::pow(beta2, static_cast<float>(step + 1));

        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            m[i] = beta1 * m[i] + (1.0f - beta1) * grads[step][i];
            v[i] = beta2 * v[i] + (1.0f - beta2) * grads[step][i] * grads[step][i];

            const float m_hat = m[i] / b1corr;
            const float v_hat = v[i] / b2corr;
            expected[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            expected[i] -= lr * weight_decay * expected[i];
        }

        std::vector<float> actual(expected.size());
        parameter.data.copyToHost(actual.data(), ctx);
        ctx.synchronize();
        expectNear(actual, expected, 2.0e-6f, "adamw.step");
    }

    std::cout << "adamw: OK\n";
    return 0;
}
