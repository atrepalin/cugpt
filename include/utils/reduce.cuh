#pragma once

namespace cugpt::utils
{
    constexpr int kWarpSize = 32;

    struct MaxOp
    {
        __device__ __forceinline__ float operator()(float a, float b) const
        {
            return fmax(a, b);
        }

        __device__ __forceinline__ float identity() const
        {
            return -INFINITY;
        }
    };

    struct SumOp
    {
        __device__ __forceinline__ float operator()(float a, float b) const
        {
            return a + b;
        }

        __device__ __forceinline__ float identity()
        {
            return 0.0f;
        }
    };

    template <typename Op>
    __device__ __forceinline__ float warpReduceKernel(float value, Op op)
    {
        for (int stride = kWarpSize >> 1; stride > 0; stride >>= 1)
        {
            value = op(value, __shfl_down_sync(0xffffffff, value, stride));
        }

        return value;
    }

    // Reduce up to 1024 values using two warp reduces
    template <typename Op>
    __device__ __forceinline__ float blockReduceKernel(float value, float *shared, Op op)
    {
        const int lane = threadIdx.x & (kWarpSize - 1);
        const int warp = threadIdx.x / kWarpSize;
        const int warpCount = (blockDim.x + kWarpSize - 1) / kWarpSize;

        // Reduce within each wrap
        value = warpReduceKernel(value, op);

        if (lane == 0)
        {
            shared[warp] = value;
        }

        __syncthreads();

        float result = op.identity();

        // Load per-warp results into first warp
        if (threadIdx.x < warpCount)
        {
            result = shared[threadIdx.x];
        }

        // Reduce results again
        if (warp == 0)
        {
            result = warpReduceKernel(result, op);
        }

        // Only first thread publish result to shared memory
        if (threadIdx.x == 0)
        {
            shared[0] = result;
        }

        __syncthreads();

        return shared[0];
    }
} // namespace cugpt::utils