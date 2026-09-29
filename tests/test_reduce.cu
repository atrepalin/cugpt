#include "core/cuda_context.cuh"
#include "core/cuda_utils.cuh"
#include "utils/reduce.cuh"
#include "test_utils.hpp"

#include <iostream>

using namespace cugpt::core;
using namespace cugpt::utils;

namespace
{
    __global__ void testKernel(const float *input, float *output)
    {
        extern __shared__ float shared[];

        const int warpCount = (blockDim.x + kWarpSize - 1) / kWarpSize;

        float localSum = input[threadIdx.x];

        output[blockIdx.x] = blockReduceKernel(localSum, shared, SumOp{});
    }
} // namespace

int main()
{
    CudaContext ctx;

    Tensor input({1024});
    Tensor output({4});

    input.fill(1.0f, ctx);

    constexpr int kThreads = 256;

    const int warp_count = (kThreads + kWarpSize - 1) / kWarpSize;

    const std::size_t shared_bytes = static_cast<std::size_t>(warp_count) * sizeof(float);

    CUDA_KERNEL_CHECK(testKernel<<<4, kThreads, shared_bytes, ctx.stream()>>>(
        input.data(), output.data()));

    ctx.synchronize();

    expectNear(copyTensor(output, ctx), {256.0f, 256.0f, 256.0f, 256.0f}, 1e-5, "sum");

    std::cout << "reduce: OK\n";
    return 0;
}