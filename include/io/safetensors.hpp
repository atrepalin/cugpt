#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cugpt::io
{
    class SafeTensorsWriter
    {
    public:
        SafeTensorsWriter();
        ~SafeTensorsWriter();

        SafeTensorsWriter(const SafeTensorsWriter &) = delete;
        SafeTensorsWriter &operator=(const SafeTensorsWriter &) = delete;
        SafeTensorsWriter(SafeTensorsWriter &&) noexcept;
        SafeTensorsWriter &operator=(SafeTensorsWriter &&) noexcept;

        void add(
            std::string_view name,
            const float *data,
            std::size_t count,
            std::vector<std::size_t> shape);

        void addMetadata(std::string_view name, std::string_view value);
        void save(const std::string &path) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    class SafeTensorsReader
    {
    public:
        SafeTensorsReader(const std::string &path);
        ~SafeTensorsReader();

        SafeTensorsReader(const SafeTensorsReader &) = delete;
        SafeTensorsReader &operator=(const SafeTensorsReader &) = delete;
        SafeTensorsReader(SafeTensorsReader &&) noexcept;
        SafeTensorsReader &operator=(SafeTensorsReader &&) noexcept;

        bool contains(std::string_view name) const;
        std::vector<std::string> tensorNames() const;
        std::string metadata(std::string_view name) const;

        void read(
            std::string_view name,
            float *data,
            std::size_t count,
            const std::vector<std::size_t> &shape) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace cugpt::io
