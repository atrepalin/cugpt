#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdlib>
#include <iostream>

namespace cugpt::core
{
    inline void checkCuda(cudaError_t status, const char *expr, const char *file, int line)
    {
        if (status != cudaSuccess)
        {
            std::cerr << "CUDA error: " << expr << " -> "
                      << cudaGetErrorString(status) << " at " << file << ':' << line << '\n';
            std::abort();
        }
    }

    inline void checkCublas(cublasStatus_t status, const char *expr, const char *file, int line)
    {
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "cuBLAS error: " << expr << " -> "
                      << cublasGetStatusString(status) << " at " << file << ':' << line << '\n';
            std::abort();
        }
    }
} // namespace cugpt::core

#define CUDA_CHECK(expr) ::cugpt::core::checkCuda((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) ::cugpt::core::checkCublas((expr), #expr, __FILE__, __LINE__)
#define CUDA_KERNEL_CHECK(...)                                     \
    do                                                             \
    {                                                              \
        __VA_ARGS__;                                               \
        ::cugpt::core::checkCuda(                                  \
            cudaGetLastError(), #__VA_ARGS__, __FILE__, __LINE__); \
    } while (false)
