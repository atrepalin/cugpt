#include "data/dataset.hpp"

namespace cugpt::data
{
    using namespace core;

    void copyBatchToDevice(
        CudaContext &ctx,
        const Dataset &dataset,
        bool validation,
        std::size_t start,
        std::size_t batch_size,
        DeviceBatch &batch)
    {
        const std::size_t total = validation ? dataset.valSize() : dataset.trainSize();

        if (start > total)
        {
            throw std::out_of_range("Batch start out of bounds");
        }

        const std::size_t count = std::min(batch_size, total - start);

        if (count == 0)
        {
            throw std::invalid_argument("Empty batch");
        }

        const auto &x_source = validation ? dataset.valX() : dataset.trainX();

        const auto &y_source = validation ? dataset.valY() : dataset.trainY();

        const std::size_t stride = dataset.sequenceLength();

        const std::size_t offset = start * stride;

        batch.count = count;

        batch.x.resize({count, stride});
        batch.y.resize({count, stride});

        batch.x.copyFromHost(x_source.data() + offset, ctx);

        batch.y.copyFromHost(y_source.data() + offset, ctx);
    }
} // namespace cugpt::data