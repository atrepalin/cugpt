#include "core/cuda_context.cuh"
#include "core/cuda_utils.cuh"

namespace cugpt::core
{
    CudaContext::CudaContext()
    {
        // Attach stream to cublas for implicit synchronization

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUBLAS_CHECK(cublasCreate(&blas_));
        CUBLAS_CHECK(cublasSetStream(blas_, stream_));
    }

    CudaContext::~CudaContext()
    {
        if (blas_ != nullptr)
        {
            cublasDestroy(blas_);
            blas_ = nullptr;
        }
        if (stream_ != nullptr)
        {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    void CudaContext::synchronize() const
    {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
} // namespace cugpt::core
