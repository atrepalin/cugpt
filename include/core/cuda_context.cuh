#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cugpt::core
{
    // RAII wrapper for CUDA and cuBLAS
    class CudaContext
    {
    public:
        CudaContext();
        ~CudaContext();

        CudaContext(const CudaContext &) = delete;
        CudaContext &operator=(const CudaContext &) = delete;

        cudaStream_t stream() const noexcept { return stream_; }
        cublasHandle_t blas() const noexcept { return blas_; }

        void synchronize() const;

    private:
        cudaStream_t stream_ = nullptr;
        cublasHandle_t blas_ = nullptr;
    };
} // namespace cugpt::core
