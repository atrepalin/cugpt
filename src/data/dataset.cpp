#include "data/dataset.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cugpt::data
{
    using namespace core;

    namespace
    {
        const std::vector<std::string> kDigits{
            "ноль",
            "один",
            "два",
            "три",
            "четыре",
            "пять",
            "шесть",
            "семь",
            "восемь",
            "девять"};

        const std::array<std::pair<int, const char *>, 3> kPlaces{{
            {100, "<сотни>"},
            {10, "<десятки>"},
            {1, "<единицы>"},
        }};

        struct Operation
        {
            std::string token;
            int (*apply)(int, int);
        };

        int plusOp(int a, int b)
        {
            return a + b;
        }

        int minusOp(int a, int b)
        {
            return a - b;
        }

        int multiplyOp(int a, int b)
        {
            return a * b;
        }

        int divideOp(int a, int b)
        {
            return a / b;
        }

        Operation operationOf(std::string_view name)
        {
            if (name == "plus")
            {
                return {"плюс", plusOp};
            }
            if (name == "minus")
            {
                return {"минус", minusOp};
            }
            if (name == "multiply")
            {
                return {"умножить на", multiplyOp};
            }
            if (name == "divide")
            {
                return {"делить на", divideOp};
            }

            throw std::invalid_argument("Unknown operation: " + std::string(name));
        }

        std::vector<std::pair<int, int>> samplePairs(
            std::string_view operation,
            int max_number,
            std::size_t count,
            std::mt19937_64 &rng)
        {
            if (operation == "multiply" || operation == "divide")
            {
                std::vector<std::pair<int, int>> all_pairs;

                if (operation == "multiply")
                {
                    for (int a = 1; a <= max_number; ++a)
                    {
                        for (int b = 0; b <= max_number / a; ++b)
                        {
                            all_pairs.emplace_back(a, b);
                        }
                    }

                    for (int b = 0; b <= max_number; ++b)
                    {
                        all_pairs.emplace_back(0, b);
                    }
                }
                else
                {
                    for (int a = 0; a <= max_number; ++a)
                    {
                        for (int b = 1; b <= max_number; ++b)
                        {
                            if (a % b == 0)
                            {
                                all_pairs.emplace_back(a, b);
                            }
                        }
                    }
                }

                std::sort(all_pairs.begin(), all_pairs.end());
                all_pairs.erase(std::unique(all_pairs.begin(), all_pairs.end()), all_pairs.end());

                std::shuffle(all_pairs.begin(), all_pairs.end(), rng);

                if (all_pairs.size() > count)
                {
                    all_pairs.resize(count);
                }

                return all_pairs;
            }

            std::unordered_set<std::uint64_t> seen;
            std::vector<std::pair<int, int>> pairs;

            pairs.reserve(count);

            const std::size_t max_unique =
                operation == "plus"
                    ? static_cast<std::size_t>(max_number + 1) *
                          static_cast<std::size_t>(max_number + 2) / 2
                    : static_cast<std::size_t>(max_number + 1) *
                          static_cast<std::size_t>(max_number + 2) / 2;

            if (count > max_unique)
            {
                throw std::invalid_argument("Requested more unique pairs than possible for operation " + std::string(operation));
            }

            while (pairs.size() < count)
            {
                std::uniform_int_distribution<int> a_dist(0, max_number);

                const int a = a_dist(rng);
                int b = 0;

                if (operation == "plus")
                {
                    std::uniform_int_distribution<int> b_dist(0, max_number - a);

                    b = b_dist(rng);
                }
                else if (operation == "minus")
                {
                    std::uniform_int_distribution<int> b_dist(0, a);

                    b = b_dist(rng);
                }
                else
                {
                    throw std::invalid_argument("Unknown operation: " + std::string(operation));
                }

                const auto key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) | static_cast<std::uint32_t>(b);

                if (seen.insert(key).second)
                {
                    pairs.emplace_back(a, b);
                }
            }

            return pairs;
        }

        std::vector<int32_t> flattenSequences(const std::vector<std::vector<int32_t>> &sequences, std::size_t sequence_length)
        {
            std::vector<int32_t> result;

            result.resize(sequences.size() * sequence_length);

            for (std::size_t i = 0; i < sequences.size(); ++i)
            {
                std::copy(sequences[i].begin(), sequences[i].end(), result.begin() + i * sequence_length);
            }

            return result;
        }
    } // namespace

    std::vector<std::string> numberToTokens(int value)
    {
        if (value < 0 || value > 999)
        {
            throw std::invalid_argument("number must be in [0, 999]");
        }

        if (value == 0)
        {
            return {"<единицы>", "ноль"};
        }

        const int hundreds = (value / 100) % 10;
        const int tens = (value / 10) % 10;
        const int ones = value % 10;

        const std::array<std::pair<int, int>, 3> digits{{
            {100, hundreds},
            {10, tens},
            {1, ones},
        }};

        int first = -1;
        int last = -1;

        for (int i = 0; i < static_cast<int>(digits.size()); ++i)
        {
            if (digits[static_cast<std::size_t>(i)].second != 0)
            {
                if (first < 0)
                {
                    first = i;
                }

                last = i;
            }
        }

        std::vector<std::string> tokens;

        for (int i = first; i <= last; ++i)
        {
            const int place = digits[static_cast<std::size_t>(i)].first;

            const int digit = digits[static_cast<std::size_t>(i)].second;

            const char *place_token = nullptr;

            for (const auto &[p, token] : kPlaces)
            {
                if (p == place)
                {
                    place_token = token;
                    break;
                }
            }

            tokens.emplace_back(place_token);
            tokens.emplace_back(kDigits[static_cast<std::size_t>(digit)]);
        }

        return tokens;
    }

    std::string makeExpression(int a, int b, std::string_view operation)
    {
        const Operation op = operationOf(operation);

        if (operation == "divide" && b == 0)
        {
            throw std::invalid_argument("division by zero");
        }

        const int result = op.apply(a, b);

        if (result < 0 || result > 999)
        {
            throw std::invalid_argument("result must be in [0, 999]");
        }

        std::vector<std::string> tokens;
        tokens.emplace_back("<start>");

        const auto appendNumber = [&tokens](int value)
        {
            const auto parts = numberToTokens(value);

            tokens.insert(tokens.end(), parts.begin(), parts.end());
        };

        appendNumber(a);
        tokens.push_back(op.token);
        appendNumber(b);
        tokens.emplace_back("равно");
        appendNumber(result);
        tokens.emplace_back("<end>");

        std::string text;

        for (std::size_t i = 0; i < tokens.size(); ++i)
        {
            if (i != 0)
            {
                text.push_back(' ');
            }

            text += tokens[i];
        }

        return text;
    }

    MathTokenizer::MathTokenizer(const std::vector<std::string> &operations)
    {
        vocabulary_ = {
            "<pad>",
            "<start>",
            "<end>",
            "<сотни>",
            "<десятки>",
            "<единицы>",
        };

        for (const std::string &operation : operations)
        {
            vocabulary_.push_back(operationOf(operation).token);
        }

        vocabulary_.push_back("равно");

        for (const std::string &digit : kDigits)
        {
            vocabulary_.push_back(digit);
        }

        std::vector<std::string> unique_vocabulary;
        unique_vocabulary.reserve(vocabulary_.size());

        for (const std::string &token : vocabulary_)
        {
            if (std::find(unique_vocabulary.begin(), unique_vocabulary.end(), token) == unique_vocabulary.end())
            {
                unique_vocabulary.push_back(token);
            }
        }

        vocabulary_ = std::move(unique_vocabulary);

        token_patterns_ = vocabulary_;

        token_patterns_.erase(std::remove(token_patterns_.begin(), token_patterns_.end(), "<pad>"), token_patterns_.end());

        std::sort(token_patterns_.begin(), token_patterns_.end(), [](const std::string &a, const std::string &b)
                  { return a.size() > b.size(); });
    }

    int32_t MathTokenizer::tokenId(std::string_view token) const
    {
        for (std::size_t i = 0; i < vocabulary_.size(); ++i)
        {
            if (vocabulary_[i] == token)
            {
                return static_cast<int32_t>(i);
            }
        }

        throw std::invalid_argument("Unknown token: " + std::string(token));
    }

    const std::string &MathTokenizer::token(int32_t id) const
    {
        if (id < 0 || id >= static_cast<int32_t>(vocabulary_.size()))
        {
            throw std::out_of_range("Token id out of range");
        }

        return vocabulary_[static_cast<std::size_t>(id)];
    }

    std::vector<int32_t> MathTokenizer::encode(std::string_view text) const
    {
        std::vector<int32_t> result;

        std::size_t pos = 0;

        while (pos < text.size())
        {
            if (std::isspace(static_cast<unsigned char>(text[pos])))
            {
                ++pos;
                continue;
            }

            bool matched = false;

            for (const std::string &pattern : token_patterns_)
            {
                if (text.substr(pos, pattern.size()) == pattern)
                {
                    result.push_back(tokenId(pattern));

                    pos += pattern.size();
                    matched = true;
                    break;
                }
            }

            if (!matched)
            {
                throw std::invalid_argument("Unknown token at position " + std::to_string(pos) + ": " + std::string(text.substr(pos)));
            }
        }

        return result;
    }

    std::string MathTokenizer::decode(const std::vector<int32_t> &ids) const
    {
        std::string result;
        bool first = true;

        for (const int32_t id : ids)
        {
            if (id == padId())
            {
                continue;
            }

            if (!first)
            {
                result.push_back(' ');
            }

            result += token(id);
            first = false;
        }

        return result;
    }

    MathDataset::MathDataset(std::vector<std::string> operations)
        : tokenizer_(operations)
    {
    }

    MathDataset MathDataset::generate(const Config &config)
    {
        if (config.max_number != 999)
        {
            throw std::invalid_argument("This dataset format supports numbers from 0 to 999.");
        }

        if (config.operations.empty())
        {
            throw std::invalid_argument("At least one operation is required");
        }

        if (config.val_fraction < 0.0 || config.val_fraction >= 1.0)
        {
            throw std::invalid_argument("val_fraction must be in [0, 1)");
        }

        for (const std::string &operation : config.operations)
        {
            operationOf(operation);
        }

        MathDataset dataset(config.operations);

        std::mt19937_64 rng(config.seed);

        std::vector<std::string> examples;

        examples.reserve(config.operations.size() * config.examples_per_operation);

        for (const std::string &operation : config.operations)
        {
            const auto pairs = samplePairs(operation, config.max_number, config.examples_per_operation, rng);

            for (const auto &[a, b] : pairs)
            {
                examples.push_back(makeExpression(a, b, operation));
            }
        }

        std::shuffle(examples.begin(), examples.end(), rng);

        const std::size_t split = static_cast<std::size_t>(static_cast<double>(examples.size()) * (1.0 - config.val_fraction));

        std::vector<std::string> train_texts(examples.begin(), examples.begin() + split);

        std::vector<std::string> val_texts(examples.begin() + split, examples.end());

        std::vector<std::vector<int32_t>> encoded_train(train_texts.size());

        std::vector<std::vector<int32_t>> encoded_val(val_texts.size());

        std::size_t max_len = 0;

        for (std::size_t i = 0; i < train_texts.size(); ++i)
        {
            encoded_train[i] = dataset.tokenizer_.encode(train_texts[i]);

            max_len = std::max(max_len, encoded_train[i].size());
        }

        for (std::size_t i = 0; i < val_texts.size(); ++i)
        {
            encoded_val[i] = dataset.tokenizer_.encode(val_texts[i]);

            max_len = std::max(max_len, encoded_val[i].size());
        }

        if (max_len < 2)
        {
            throw std::runtime_error("Dataset contains sequences shorter than two tokens");
        }

        const int32_t pad = dataset.tokenizer_.padId();

        auto padSequences = [max_len, pad](const std::vector<std::vector<int32_t>> &sequences)
        {
            std::vector<std::vector<int32_t>> result(sequences.size(), std::vector<int32_t>(max_len, pad));

            for (std::size_t i = 0; i < sequences.size(); ++i)
            {
                std::copy(sequences[i].begin(), sequences[i].end(), result[i].begin());
            }

            return result;
        };

        const auto train_seq = padSequences(encoded_train);

        const auto val_seq = padSequences(encoded_val);

        dataset.sequence_length_ = max_len - 1;

        auto makeShifted = [max_len, pad, eq = dataset.tokenizer_.equalsId()](
                               const std::vector<std::vector<int32_t>> &sequences,
                               std::vector<int32_t> &x_out,
                               std::vector<int32_t> &y_out)
        {
            const std::size_t t = max_len - 1;

            x_out.resize(sequences.size() * t);

            y_out.resize(sequences.size() * t);

            for (std::size_t i = 0; i < sequences.size(); ++i)
            {
                const auto base = i * t;

                std::copy(sequences[i].begin(), sequences[i].begin() + t, x_out.begin() + base);

                std::copy(sequences[i].begin() + 1, sequences[i].end(), y_out.begin() + base);

                std::size_t eq_count = 0;
                std::size_t eq_pos = 0;

                for (std::size_t j = 0; j < max_len; ++j)
                {
                    if (sequences[i][j] == eq)
                    {
                        ++eq_count;
                        eq_pos = j;
                    }
                }

                if (eq_count != 1)
                {
                    throw std::runtime_error("Each sequence must contain exactly one 'равно' token.");
                }

                for (std::size_t j = 0; j < eq_pos; ++j)
                {
                    y_out[base + j] = pad;
                }
            }
        };

        makeShifted(train_seq, dataset.train_x_, dataset.train_y_);

        makeShifted(val_seq, dataset.val_x_, dataset.val_y_);

        return dataset;
    }

    std::vector<int32_t> MathDataset::makeBatchX(bool validation, std::size_t start, std::size_t count) const
    {
        const auto &source = validation ? val_x_ : train_x_;

        if (sequence_length_ == 0)
        {
            throw std::runtime_error("Invalid sequence length");
        }

        const std::size_t total = source.size() / sequence_length_;

        if (start > total ||
            count > total - start)
        {
            throw std::out_of_range("Batch range out of bounds");
        }

        const std::size_t offset = start * sequence_length_;

        const std::size_t size = count * sequence_length_;

        return std::vector<int32_t>(source.begin() + offset, source.begin() + (offset + size));
    }

    std::vector<int32_t> MathDataset::makeBatchY(bool validation, std::size_t start, std::size_t count) const
    {
        const auto &source = validation ? val_y_ : train_y_;

        if (sequence_length_ == 0)
        {
            throw std::runtime_error("Invalid sequence length");
        }

        const std::size_t total = source.size() / sequence_length_;

        if (start > total || count > total - start)
        {
            throw std::out_of_range("Batch range out of bounds");
        }

        const std::size_t offset = start * sequence_length_;

        const std::size_t size = count * sequence_length_;

        return std::vector<int32_t>(source.begin() + offset, source.begin() + (offset + size));
    }

    void copyBatchToDevice(
        CudaContext &ctx,
        const MathDataset &dataset,
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
