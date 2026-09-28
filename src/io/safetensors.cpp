#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

#include "io/safetensors.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cugpt::io
{
    namespace
    {
        std::size_t shapeNumel(const std::vector<std::size_t> &shape)
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

        std::string operationError(const char *operation, const std::string &path, const std::string &err)
        {
            std::string message = std::string(operation) + " '" + path + "' failed";

            if (!err.empty())
            {
                message += ": " + err;
            }

            return message;
        }

        safetensors::tensor_t findTensor(const safetensors::safetensors_t &st, std::string_view name)
        {
            const std::string key(name);
            safetensors::tensor_t tensor{};

            if (!st.tensors.at(key, &tensor))
            {
                throw std::out_of_range("Tensor not found: " + key);
            }

            return tensor;
        }
    } // namespace

    struct SafeTensorsWriter::Impl
    {
        safetensors::safetensors_t st;
    };

    SafeTensorsWriter::SafeTensorsWriter()
        : impl_(std::make_unique<Impl>())
    {
    }

    SafeTensorsWriter::~SafeTensorsWriter() = default;
    SafeTensorsWriter::SafeTensorsWriter(SafeTensorsWriter &&) noexcept = default;
    SafeTensorsWriter &SafeTensorsWriter::operator=(SafeTensorsWriter &&) noexcept = default;

    void SafeTensorsWriter::add(
        std::string_view name,
        const float *data,
        std::size_t count,
        std::vector<std::size_t> shape)
    {
        if (name.empty())
        {
            throw std::invalid_argument("Tensor name cannot be empty");
        }

        if (count != 0 && data == nullptr)
        {
            throw std::invalid_argument("Data is null for non-empty tensor");
        }

        if (shapeNumel(shape) != count)
        {
            throw std::invalid_argument("Shape element count does not match data count for '" + std::string(name) + "'");
        }

        const std::string key(name);
        if (impl_->st.tensors.count(key) != 0)
        {
            throw std::invalid_argument("Duplicate tensor name: " + key);
        }

        if (count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        {
            throw std::overflow_error("Tensor byte size overflow");
        }

        const std::size_t offset = impl_->st.storage.size();
        const std::size_t bytes = count * sizeof(float);
        impl_->st.storage.resize(offset + bytes);

        if (bytes != 0)
        {
            std::memcpy(impl_->st.storage.data() + offset, data, bytes);
        }

        safetensors::tensor_t tensor;
        tensor.dtype = safetensors::dtype::kFLOAT32;
        tensor.shape = std::move(shape);
        tensor.data_offsets[0] = offset;
        tensor.data_offsets[1] = offset + bytes;

        impl_->st.tensors.insert(key, std::move(tensor));
    }

    void SafeTensorsWriter::addMetadata(std::string_view name, std::string_view value)
    {
        if (name.empty())
        {
            throw std::invalid_argument("Metadata name cannot be empty");
        }

        impl_->st.metadata.insert(std::string(name), std::string(value));
    }

    void SafeTensorsWriter::save(const std::string &path) const
    {
        std::string warn;
        std::string err;
        if (!safetensors::validate_data_offsets(impl_->st, err))
        {
            throw std::runtime_error("Safetensors has invalid data offsets: " + err);
        }

        if (!safetensors::save_to_file(impl_->st, path, &warn, &err))
        {
            throw std::runtime_error(operationError("Safetensors save", path, err));
        }
    }

    struct SafeTensorsReader::Impl
    {
        safetensors::safetensors_t st;
    };

    SafeTensorsReader::SafeTensorsReader(const std::string &path)
        : impl_(std::make_unique<Impl>())
    {
        std::string warn;
        std::string err;

        if (!safetensors::load_from_file(path, &impl_->st, &warn, &err))
        {
            throw std::runtime_error(operationError("Safetensors load", path, err));
        }

        if (!safetensors::validate_data_offsets(impl_->st, err))
        {
            throw std::runtime_error("Safetensors load '" + path + "' has invalid data offsets: " + err);
        }
    }

    SafeTensorsReader::~SafeTensorsReader() = default;
    SafeTensorsReader::SafeTensorsReader(SafeTensorsReader &&) noexcept = default;
    SafeTensorsReader &SafeTensorsReader::operator=(SafeTensorsReader &&) noexcept = default;

    bool SafeTensorsReader::contains(std::string_view name) const
    {
        return impl_->st.tensors.count(std::string(name)) != 0;
    }

    std::vector<std::string> SafeTensorsReader::tensorNames() const
    {
        return impl_->st.tensors.keys();
    }

    std::string SafeTensorsReader::metadata(std::string_view name) const
    {
        std::string value;
        if (!impl_->st.metadata.at(std::string(name), &value))
        {
            throw std::out_of_range("Metadata not found: " + std::string(name));
        }

        return value;
    }

    void SafeTensorsReader::read(
        std::string_view name,
        float *data,
        std::size_t count,
        const std::vector<std::size_t> &expected_shape) const
    {
        if (count != 0 && data == nullptr)
        {
            throw std::invalid_argument("Data is null for non-empty tensor");
        }

        const std::string key(name);
        const safetensors::tensor_t tensor = findTensor(impl_->st, name);

        if (tensor.dtype != safetensors::dtype::kFLOAT32)
        {
            throw std::runtime_error("Tensor '" + key + "' is not F32");
        }

        if (tensor.shape != expected_shape)
        {
            throw std::runtime_error("Shape mismatch for tensor '" + key + "'");
        }

        if (shapeNumel(tensor.shape) != count)
        {
            throw std::runtime_error("Element count mismatch for tensor '" + key + "'");
        }

        if (count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        {
            throw std::overflow_error("Tensor byte size overflow");
        }

        const std::size_t begin = tensor.data_offsets[0];
        const std::size_t end = tensor.data_offsets[1];
        const std::size_t bytes = count * sizeof(float);

        if (end < begin || end - begin != bytes)
        {
            throw std::runtime_error("Byte range mismatch for tensor '" + key + "'");
        }

        if (end > impl_->st.storage.size())
        {
            throw std::runtime_error("Tensor '" + key + "' points outside the safetensors storage");
        }

        if (bytes != 0)
        {
            std::memcpy(data, impl_->st.storage.data() + begin, bytes);
        }
    }
} // namespace cugpt::io
