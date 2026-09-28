#include "core/tensor.cuh"
#include "core/cuda_utils.cuh"

namespace cugpt::core
{
    std::size_t checkedNumel(const Shape &shape)
    {
        if (shape.empty())
        {
            return 0;
        }

        std::size_t result = 1;
        for (const size_t dim : shape)
        {
            if (dim <= 0)
            {
                throw std::invalid_argument("Tensor dimensions must be positive");
            }

            result *= dim;
        }
        return result;
    }

    template <typename T>
    void expectSameNumel(const AbstractTensor<T> &a, const AbstractTensor<T> &b, const char *message)
    {
        if (a.numel() != b.numel())
        {
            throw std::invalid_argument(message);
        }
    }

    void Tensor::zero(const CudaContext &ctx)
    {
        CUDA_CHECK(cudaMemsetAsync(data(), 0, bytes(), ctx.stream()));
    }

    void Tensor::fill(float value, const CudaContext &ctx)
    {
        if (value == 0.0f)
        {
            zero(ctx);
            return;
        }

        std::vector<float> host(numel_, value);
        copyFromHost(host.data(), ctx);
    }
} // namespace cugpt::core