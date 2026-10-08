#include "dataset.hpp"
#include "inference/generation.hpp"
#include "nn/gpt.cuh"
#include "training/cross_entropy.cuh"
#include "training/adamw.cuh"
#include "io/serialization.hpp"
#include "training/training.cuh"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace cugpt::core;
using namespace cugpt::nn;
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

        float learning_rate = 3.0e-4f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float optimizer_eps = 1.0e-8f;
        float weight_decay = 0.01f;

        bool use_reduce_lr_on_plateau = false;
        ReduceLROnPlateau::Config plateau;
        PlateauMonitor plateau_monitor = PlateauMonitor::ValidationLoss;
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

    std::size_t parseNonNegativeSize(const char *value, const char *name)
    {
        try
        {
            std::size_t consumed = 0;
            const unsigned long long parsed = std::stoull(value, &consumed);
            if (consumed == 0 || value[consumed] != '\0')
            {
                throw std::invalid_argument("invalid integer");
            }
            return static_cast<std::size_t>(parsed);
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument(std::string("Invalid ") + name + ": " + value);
        }
    }

    float parseFloat(const char *value, const char *name)
    {
        try
        {
            std::size_t consumed = 0;
            const float parsed = std::stof(value, &consumed);
            if (consumed == 0 || value[consumed] != '\0' || !std::isfinite(parsed))
            {
                throw std::invalid_argument("invalid floating-point value");
            }
            return parsed;
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument(std::string("Invalid ") + name + ": " + value);
        }
    }

    ReduceLROnPlateauMode parsePlateauMode(const char *value)
    {
        const std::string mode = value;
        if (mode == "min")
        {
            return ReduceLROnPlateauMode::Min;
        }
        if (mode == "max")
        {
            return ReduceLROnPlateauMode::Max;
        }
        throw std::invalid_argument(std::string("Invalid lr-scheduler-mode: ") + value + " (expected min or max)");
    }

    ReduceLROnPlateauThresholdMode parsePlateauThresholdMode(const char *value)
    {
        const std::string mode = value;
        if (mode == "rel")
        {
            return ReduceLROnPlateauThresholdMode::Rel;
        }
        if (mode == "abs")
        {
            return ReduceLROnPlateauThresholdMode::Abs;
        }
        throw std::invalid_argument(
            std::string("Invalid lr-scheduler-threshold-mode: ") + value + " (expected rel or abs)");
    }

    PlateauMonitor parsePlateauMonitor(const char *value)
    {
        const std::string monitor = value;
        if (monitor == "train-loss")
        {
            return PlateauMonitor::TrainLoss;
        }
        if (monitor == "val-loss")
        {
            return PlateauMonitor::ValidationLoss;
        }
        if (monitor == "val-accuracy")
        {
            return PlateauMonitor::ValidationAccuracy;
        }
        throw std::invalid_argument(
            std::string("Invalid lr-scheduler-monitor: ") + value +
            " (expected train-loss, val-loss, or val-accuracy)");
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
            << "  --checkpoint PATH   Best model checkpoint (default: best_model.safetensors)\n"
            << "  --lr FLOAT           AdamW learning rate (default: 3e-4)\n"
            << "  --beta1 FLOAT\n"
            << "  --beta2 FLOAT\n"
            << "  --optimizer-eps FLOAT\n"
            << "  --weight-decay FLOAT\n"
            << "  --lr-scheduler {none|reduce-lr-on-plateau}\n"
            << "  --lr-scheduler-monitor {train-loss|val-loss|val-accuracy}\n"
            << "  --lr-scheduler-mode {min|max}\n"
            << "  --lr-scheduler-factor FLOAT\n"
            << "  --lr-scheduler-patience N\n"
            << "  --lr-scheduler-threshold FLOAT\n"
            << "  --lr-scheduler-threshold-mode {rel|abs}\n"
            << "  --lr-scheduler-cooldown N\n"
            << "  --lr-scheduler-min-lr FLOAT\n"
            << "  --lr-scheduler-eps FLOAT\n\n"
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
            else if (option == "--lr" && i + 1 < argc)
            {
                args.learning_rate = parseFloat(argv[++i], "lr");
            }
            else if (option == "--beta1" && i + 1 < argc)
            {
                args.beta1 = parseFloat(argv[++i], "beta1");
            }
            else if (option == "--beta2" && i + 1 < argc)
            {
                args.beta2 = parseFloat(argv[++i], "beta2");
            }
            else if (option == "--optimizer-eps" && i + 1 < argc)
            {
                args.optimizer_eps = parseFloat(argv[++i], "optimizer-eps");
            }
            else if (option == "--weight-decay" && i + 1 < argc)
            {
                args.weight_decay = parseFloat(argv[++i], "weight-decay");
            }
            else if (option == "--lr-scheduler" && i + 1 < argc)
            {
                const std::string scheduler = argv[++i];
                if (scheduler == "none")
                {
                    args.use_reduce_lr_on_plateau = false;
                }
                else if (scheduler == "reduce-lr-on-plateau" || scheduler == "reduce-on-plateau")
                {
                    args.use_reduce_lr_on_plateau = true;
                }
                else
                {
                    throw std::invalid_argument(
                        "Invalid lr-scheduler: " + scheduler + " (expected none or reduce-lr-on-plateau)");
                }
            }
            else if (option == "--lr-scheduler-monitor" && i + 1 < argc)
            {
                args.plateau_monitor = parsePlateauMonitor(argv[++i]);
            }
            else if (option == "--lr-scheduler-mode" && i + 1 < argc)
            {
                args.plateau.mode = parsePlateauMode(argv[++i]);
            }
            else if (option == "--lr-scheduler-factor" && i + 1 < argc)
            {
                args.plateau.factor = parseFloat(argv[++i], "lr-scheduler-factor");
            }
            else if (option == "--lr-scheduler-patience" && i + 1 < argc)
            {
                args.plateau.patience = parseNonNegativeSize(argv[++i], "lr-scheduler-patience");
            }
            else if (option == "--lr-scheduler-threshold" && i + 1 < argc)
            {
                args.plateau.threshold = parseFloat(argv[++i], "lr-scheduler-threshold");
            }
            else if (option == "--lr-scheduler-threshold-mode" && i + 1 < argc)
            {
                args.plateau.threshold_mode = parsePlateauThresholdMode(argv[++i]);
            }
            else if (option == "--lr-scheduler-cooldown" && i + 1 < argc)
            {
                args.plateau.cooldown = parseNonNegativeSize(argv[++i], "lr-scheduler-cooldown");
            }
            else if (option == "--lr-scheduler-min-lr" && i + 1 < argc)
            {
                args.plateau.min_lr = parseFloat(argv[++i], "lr-scheduler-min-lr");
            }
            else if (option == "--lr-scheduler-eps" && i + 1 < argc)
            {
                args.plateau.eps = parseFloat(argv[++i], "lr-scheduler-eps");
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

        GPT model = makeModel(ctx, dataset.tokenizer().vocabularySize());
        CrossEntropyLoss loss(ctx, dataset.tokenizer().padId());
        AdamW optimizer(
            ctx,
            model.parameters(),
            args.learning_rate,
            args.beta1,
            args.beta2,
            args.optimizer_eps,
            args.weight_decay);

        std::unique_ptr<ReduceLROnPlateau> lr_scheduler;
        if (args.use_reduce_lr_on_plateau)
        {
            lr_scheduler = std::make_unique<ReduceLROnPlateau>(optimizer, args.plateau);
        }

        TrainingConfig config;
        config.epochs = args.epochs;
        config.batch_size = args.batch_size;
        config.seed = args.seed;
        config.ignore_index = dataset.tokenizer().padId();
        config.print_every = 1;
        config.best_checkpoint_path = args.checkpoint_path;
        config.plateau_monitor = args.plateau_monitor;

        std::cout << "train_size=" << dataset.trainSize()
                  << " val_size=" << dataset.valSize()
                  << " sequence_length=" << dataset.sequenceLength() << '\n';

        const auto history = trainModel(ctx, model, loss, optimizer, lr_scheduler.get(), dataset, config);

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