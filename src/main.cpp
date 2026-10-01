#include "data/dataset.hpp"
#include "inference/generation.hpp"
#include "nn/gpt.cuh"
#include "training/cross_entropy.cuh"
#include "training/adamw.cuh"
#include "io/serialization.hpp"
#include "training/training.cuh"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace cugpt::core;
using namespace cugpt::nn;
using namespace cugpt::data;
using namespace cugpt::training;
using namespace cugpt::io;
using namespace cugpt::inference;

namespace
{
    struct TrainArgs
    {
        std::size_t epochs = 15;
        std::size_t batch_size = 64;
        std::size_t examples_per_operation = 20'000;
        std::uint64_t seed = 42;
        std::string checkpoint_path = "best_model.safetensors";
    };

    struct InferenceArgs
    {
        std::string checkpoint_path;
        std::string text;
        std::size_t max_new_tokens = 50;
        std::uint64_t seed = 42;
    };

    std::size_t parseSize(const char *value, const char *name)
    {
        try
        {
            const unsigned long long parsed = std::stoull(value);
            if (parsed == 0)
            {
                throw std::invalid_argument("must be positive");
            }
            return static_cast<std::size_t>(parsed);
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument(std::string("Invalid ") + name + ": " + value);
        }
    }

    std::uint64_t parseSeed(const char *value)
    {
        try
        {
            return static_cast<std::uint64_t>(std::stoull(value));
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument(std::string("Invalid seed: ") + value);
        }
    }

    [[noreturn]] void printUsage()
    {
        std::cout
            << "Usage:\n"
            << "  cugpt train [options]\n"
            << "  cugpt inference --checkpoint PATH --text TEXT [options]\n\n"
            << "Train options:\n"
            << "  --epochs N\n"
            << "  --batch-size N\n"
            << "  --examples-per-operation N\n"
            << "  --seed N\n"
            << "  --checkpoint PATH   Best model checkpoint (default: best_model.safetensors)\n\n"
            << "Inference options:\n"
            << "  --checkpoint PATH\n"
            << "  --text TEXT\n"
            << "  --max-new-tokens N\n";
        std::exit(0);
    }

    TrainArgs parseTrainArgs(int argc, char **argv)
    {
        TrainArgs args;
        for (int i = 2; i < argc; ++i)
        {
            const std::string option = argv[i];
            if (option == "--epochs" && i + 1 < argc)
            {
                args.epochs = parseSize(argv[++i], "epochs");
            }
            else if (option == "--batch-size" && i + 1 < argc)
            {
                args.batch_size = parseSize(argv[++i], "batch-size");
            }
            else if (option == "--examples-per-operation" && i + 1 < argc)
            {
                args.examples_per_operation = parseSize(argv[++i], "examples-per-operation");
            }
            else if (option == "--seed" && i + 1 < argc)
            {
                args.seed = parseSeed(argv[++i]);
            }
            else if (option == "--checkpoint" && i + 1 < argc)
            {
                args.checkpoint_path = argv[++i];
            }
            else if (option == "--help")
            {
                printUsage();
            }
            else
            {
                throw std::invalid_argument("Unknown or incomplete train option: " + option);
            }
        }
        return args;
    }

    InferenceArgs parseInferenceArgs(int argc, char **argv)
    {
        InferenceArgs args;
        for (int i = 2; i < argc; ++i)
        {
            const std::string option = argv[i];
            if (option == "--checkpoint" && i + 1 < argc)
            {
                args.checkpoint_path = argv[++i];
            }
            else if (option == "--text" && i + 1 < argc)
            {
                args.text = argv[++i];
            }
            else if (option == "--max-new-tokens" && i + 1 < argc)
            {
                args.max_new_tokens = parseSize(argv[++i], "max-new-tokens");
            }
            else if (option == "--help")
            {
                printUsage();
            }
            else
            {
                throw std::invalid_argument("Unknown or incomplete inference option: " + option);
            }
        }

        if (args.checkpoint_path.empty())
        {
            throw std::invalid_argument("inference requires --checkpoint PATH");
        }
        if (args.text.empty())
        {
            throw std::invalid_argument("inference requires --text TEXT");
        }
        return args;
    }

    GPT makeModel(CudaContext &ctx, std::size_t vocab_size)
    {
        constexpr std::size_t blocks = 2;
        constexpr std::size_t model_dim = 128;
        constexpr std::size_t num_heads = 4;
        constexpr float positional_scale = 0.1f;
        return GPT(ctx, vocab_size, blocks, model_dim, num_heads, positional_scale);
    }

    int runTrain(int argc, char **argv)
    {
        const TrainArgs args = parseTrainArgs(argc, argv);
        CudaContext ctx;

        MathDataset::Config dataset_config;
        dataset_config.examples_per_operation = args.examples_per_operation;
        dataset_config.seed = args.seed;
        const MathDataset dataset = MathDataset::generate(dataset_config);

        GPT model = makeModel(ctx, dataset.sequenceLength());
        CrossEntropyLoss loss(ctx, dataset.tokenizer().padId());
        AdamW optimizer(
            ctx,
            model.parameters(),
            3.0e-4f,
            0.9f,
            0.999f,
            1.0e-8f,
            0.01f);

        TrainingConfig config;
        config.epochs = args.epochs;
        config.batch_size = args.batch_size;
        config.seed = args.seed;
        config.ignore_index = dataset.tokenizer().padId();
        config.print_every = 1;
        config.best_checkpoint_path = args.checkpoint_path;

        std::cout << "train_size=" << dataset.trainSize()
                  << " val_size=" << dataset.valSize()
                  << " sequence_length=" << dataset.sequenceLength() << '\n';

        const auto history = trainModel(ctx, model, loss, optimizer, dataset, config);

        float best_acc = -1.0f;
        for (const auto &epoch : history)
        {
            if (epoch.has_validation && epoch.val_sequence_accuracy > best_acc)
            {
                best_acc = epoch.val_sequence_accuracy;
            }
        }
        if (best_acc >= 0.0f)
        {
            std::cout << "best_val_acc=" << best_acc << '\n';
            std::cout << "best_checkpoint=" << args.checkpoint_path << '\n';
        }
        return 0;
    }

    int runInference(int argc, char **argv)
    {
        const InferenceArgs args = parseInferenceArgs(argc, argv);
        CudaContext ctx;

        MathTokenizer tokenizer({"plus", "minus", "multiply", "divide"});
        const std::size_t vocab_size = tokenizer.vocabularySize();
        GPT model = makeModel(ctx, vocab_size);
        loadModel(model, ctx, args.checkpoint_path);

        const std::string prediction = generateText(ctx, model, tokenizer, args.text, args.max_new_tokens);

        std::cout << "input: " << args.text << '\n';
        std::cout << "prediction: " << prediction << '\n';
        return 0;
    }

} // namespace

int runMain(int argc, char **argv)
{
    try
    {
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")
        {
            printUsage();
        }

        const std::string command = argv[1];
        if (command == "train")
        {
            return runTrain(argc, argv);
        }
        if (command == "inference")
        {
            return runInference(argc, argv);
        }
        throw std::invalid_argument("Unknown command: " + command);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

#ifdef _WIN32
std::string wideToUtf8(const wchar_t *text)
{
    if (text == nullptr || *text == L'\0')
    {
        return {};
    }

    const int required = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (required <= 0)
    {
        throw std::runtime_error("WideCharToMultiByte failed");
    }

    std::string result(static_cast<std::size_t>(required), '\0');

    if (WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), required, nullptr, nullptr) <= 0)
    {
        throw std::runtime_error("WideCharToMultiByte failed");
    }

    result.pop_back();
    return result;
}

int wmain(int argc, wchar_t **argv)
{
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);

    std::vector<std::string> utf8_args;
    utf8_args.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i)
    {
        utf8_args.push_back(wideToUtf8(argv[i]));
    }

    std::vector<char *> narrow_argv;
    narrow_argv.reserve(utf8_args.size());
    for (std::string &arg : utf8_args)
    {
        narrow_argv.push_back(arg.data());
    }

    return runMain(argc, narrow_argv.data());
}
#else
int main(int argc, char **argv)
{
    return runMain(argc, argv);
}
#endif