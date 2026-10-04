#include "core/cuda_utils.cuh"
#include "core/tensor.cuh"
#include "io/serialization.hpp"
#include "nn/gpt.cuh"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
namespace cugpt::bindings
{
    using namespace cugpt::core;
    using namespace cugpt::nn;
    using namespace cugpt::io;

    // The C++ GPT implementation uses int32 token IDs. Keeping this exact in
    // both overloads avoids an implicit host/device dtype conversion inside
    // nanobind and keeps the CuPy path genuinely zero-copy
    using CpuTokens = nb::ndarray<int32_t,
                                  nb::ndim<2>,
                                  nb::c_contig,
                                  nb::device::cpu>;

    using CudaTokens = nb::ndarray<int32_t,
                                   nb::ndim<2>,
                                   nb::c_contig,
                                   nb::device::cuda>;

    using NumpyLogits = nb::ndarray<nb::numpy,
                                    float,
                                    nb::ndim<3>,
                                    nb::c_contig,
                                    nb::device::cpu>;

    using CupyLogits = nb::ndarray<nb::cupy,
                                   float,
                                   nb::ndim<3>,
                                   nb::c_contig,
                                   nb::device::cuda>;

    class GPTBinding final
    {
    public:
        GPTBinding(std::size_t vocab_size,
                   std::size_t blocks,
                   std::size_t model_dim,
                   std::size_t num_heads,
                   float positional_scale = 0.1f)
            : ctx_(),
              model_(ctx_, vocab_size, blocks, model_dim, num_heads, positional_scale)
        {
        }

        void load(const std::string &path)
        {
            nb::gil_scoped_release release;
            loadModel(model_, ctx_, path);
        }

        // Both overloads intentionally share the Python name ``forward``.
        // Their array constraints are disjoint (CPU vs CUDA), so there is no
        // runtime Python type check or isinstance dispatch in the binding
        NumpyLogits forward(const CpuTokens &tokens)
        {
            return forward_numpy_impl(tokens);
        }

        CupyLogits forward(const CudaTokens &tokens)
        {
            return forward_cupy_impl(tokens);
        }

    private:
        NumpyLogits forward_numpy_impl(const CpuTokens &tokens)
        {
            const std::size_t batch = tokens.shape(0);
            const std::size_t time = tokens.shape(1);

            CUDA_CHECK(cudaSetDevice(0));
            IntTensor device_tokens({batch, time});
            {
                nb::gil_scoped_release release;
                device_tokens.copyFromHost(tokens.data(), ctx_);
            }

            Tensor logits;
            {
                nb::gil_scoped_release release;
                model_.forward(device_tokens, logits);
            }

            // Tensor::copyToHost synchronizes the model stream, so the NumPy
            // array is ready for immediate use when this function returns
            auto host = std::make_unique<std::vector<float>>(logits.numel());
            {
                nb::gil_scoped_release release;
                logits.copyToHost(host->data(), ctx_);
            }

            if (logits.shape().size() != 3)
            {
                throw std::runtime_error("GPT forward produced a tensor with unexpected rank");
            }

            const std::size_t rows = logits.shape()[0];
            const std::size_t cols = logits.shape()[1];
            const std::size_t vocab = logits.shape()[2];
            float *data = host->data();

            nb::capsule owner(host.release(), [](void *p) noexcept
                              { delete static_cast<std::vector<float> *>(p); });

            return NumpyLogits(data, {rows, cols, vocab}, owner);
        }

        CupyLogits forward_cupy_impl(const CudaTokens &tokens)
        {
            if (tokens.device_id() != 0)
            {
                throw std::invalid_argument("GPT CUDA binding currently supports device 0 only");
            }

            const std::size_t batch = tokens.shape(0);
            const std::size_t time = tokens.shape(1);

            CUDA_CHECK(cudaSetDevice(0));

            // Borrow the CuPy allocation directly. No H2D copy is performed.
            // The input ndarray wrapper keeps the originating Python object
            // alive for the duration of this call
            IntTensor device_tokens;
            device_tokens.view({batch, time}, tokens.data());

            // The core currently owns its own CUDA stream and has no stream
            // interop API. Synchronize device work before consuming foreign
            // CuPy memory, then synchronize the model stream before exposing
            // its result to Python. This preserves correctness while keeping
            // both input and output zero-copy on the CUDA side
            {
                nb::gil_scoped_release release;
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            auto logits = std::make_unique<Tensor>();
            {
                nb::gil_scoped_release release;
                model_.forward(device_tokens, *logits);
                ctx_.synchronize();
            }

            if (logits->shape().size() != 3)
            {
                throw std::runtime_error("GPT forward produced a tensor with unexpected rank");
            }

            const std::size_t rows = logits->shape()[0];
            const std::size_t cols = logits->shape()[1];
            const std::size_t vocab = logits->shape()[2];
            float *data = logits->data();

            nb::capsule owner(logits.release(), [](void *p) noexcept
                              { delete static_cast<Tensor *>(p); });

            return CupyLogits(data, {rows, cols, vocab}, owner);
        }

        CudaContext ctx_;
        GPT model_;
    };
} // namespace cugpt::bindings

NB_MODULE(cugpt, m)
{
    using cugpt::bindings::GPTBinding;
    using namespace nanobind::literals;

    nb::class_<GPTBinding>(m, "GPT")
        .def(nb::init<std::size_t, std::size_t, std::size_t, std::size_t, float>(),
             "vocab_size"_a,
             "blocks"_a,
             "model_dim"_a,
             "n_heads"_a,
             "positional_scale"_a = 0.1f)
        .def("load", &GPTBinding::load, "path"_a)
        .def("forward", nb::overload_cast<const cugpt::bindings::CpuTokens &>(&GPTBinding::forward), "tokens"_a.noconvert())
        .def("forward", nb::overload_cast<const cugpt::bindings::CudaTokens &>(&GPTBinding::forward), "tokens"_a.noconvert());
}
