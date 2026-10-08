# cuGPT

CUDA-accelerated GPT implementation written in C++/CUDA, with a Python reference implementation, a native Python binding, a small synthetic arithmetic dataset, training and inference CLI tools, SafeTensors checkpoints, unit tests, numerical-consistency checks, and benchmarking notebooks.

---

## Table of Contents

- [What is cuGPT?](#what-is-cugpt)
- [Highlights](#highlights)
- [Repository Layout](#repository-layout)
- [Architecture](#architecture)
- [Model Details](#model-details)
- [Attention and Causality](#attention-and-causality)
- [Synthetic Arithmetic Dataset](#synthetic-arithmetic-dataset)
- [Data Representation and Target Masking](#data-representation-and-target-masking)
- [Checkpoint Format](#checkpoint-format)
- [Python Reference Implementation](#python-reference-implementation)
- [Native Python Binding](#native-python-binding)
- [Requirements](#requirements)
- [Building the C++/CUDA Project](#building-the-ccuda-project)
- [Building and Installing the Python Package](#building-and-installing-the-python-package)
- [Running the CLI](#running-the-cli)
  - [Show Help](#show-help)
  - [Train](#train)
  - [Inference](#inference)
- [Training Configuration](#training-configuration)
- [Learning-Rate Scheduler](#learning-rate-scheduler)
- [Python API Examples](#python-api-examples)
- [CuPy / Zero-Copy CUDA Path](#cupy--zero-copy-cuda-path)
- [C++ API Overview](#c-api-overview)
- [Testing](#testing)
- [Numerical Consistency](#numerical-consistency)
- [Benchmarking](#benchmarking)
- [Reproducing the Notebook Workflow](#reproducing-the-notebook-workflow)
- [Performance Considerations](#performance-considerations)
- [Important Limitations](#important-limitations)
- [Troubleshooting](#troubleshooting)
- [Serialization / Compatibility Rules](#serialization--compatibility-rules)
- [End-to-End Quick Start](#end-to-end-quick-start)

---

## What is cuGPT?

`cuGPT` is a compact decoder-only Transformer implementation whose numerical core is implemented directly in CUDA/C++.

The project contains two implementations of essentially the same model:

1. **Reference implementation:** pure Python using NumPy on CPU or CuPy on GPU.
2. **CUDA implementation:** C++17 + CUDA, exposed through a Nanobind Python extension and also compiled as a standalone CLI executable.

The two implementations intentionally share the same model parameter naming convention and the same versioned SafeTensors checkpoint format. This makes it possible to train with one implementation, load the resulting checkpoint with the other, compare logits, and benchmark them side by side.

The included application is a synthetic arithmetic language task. Instead of training on a natural-language corpus, the repository generates expressions such as:

```text
<start> <сотни> четыре <десятки> восемь <единицы> три плюс <десятки> два равно <десятки> пять <единицы> пять <end>
```

The model learns to predict the sequence token-by-token. During training, labels before the `равно` token are masked with the padding token, so the effective objective focuses on generating the arithmetic result.

---

## Highlights

- Native Transformer implementation in CUDA/C++17.
- Decoder-only architecture with:
  - token embeddings,
  - sinusoidal positional encodings,
  - pre-normalized Transformer blocks,
  - multi-head self-attention,
  - causal softmax,
  - two-layer ReLU feed-forward networks,
  - residual connections,
  - final LayerNorm,
  - vocabulary projection.
- Hand-written CUDA kernels for several operations.
- cuBLAS-backed batched matrix multiplications inside attention.
- Forward and backward implementations, not inference-only code.
- AdamW optimizer implemented in CUDA.
- Cross-entropy with `ignore_index` support.
- Reduce-on-plateau learning-rate scheduler.
- SafeTensors serialization with explicit architecture metadata and strict tensor-name validation.
- Native Python extension built with Nanobind.
- CPU NumPy input path and GPU CuPy input path for the Python binding.
- Built-in greedy autoregressive generation.
- Standalone training/inference CLI.
- CTest-based unit tests for core neural-network components.
- Notebooks for training, benchmark comparison, and Python-vs-CUDA numerical consistency.

---

## Repository Layout

```text
.
├── CMakeLists.txt
├── pyproject.toml
├── requirements.txt
├── backend.py
├── dataset.py
├── model.py
├── main.ipynb
├── benchmark.ipynb
├── consistency.ipynb
│
├── bindings/
│   └── cugpt.cpp
│
├── include/
│   ├── core/
│   │   ├── cuda_context.cuh
│   │   ├── cuda_utils.cuh
│   │   ├── device_buffer.hpp
│   │   ├── module.hpp
│   │   ├── parameter.hpp
│   │   └── tensor.cuh
│   ├── data/
│   │   └── dataset.hpp
│   ├── inference/
│   │   └── generation.hpp
│   ├── io/
│   │   ├── safetensors.hpp
│   │   └── serialization.hpp
│   ├── nn/
│   │   ├── attention.cuh
│   │   ├── embedding.cuh
│   │   ├── feed_forward.cuh
│   │   ├── gpt.cuh
│   │   ├── layernorm.cuh
│   │   ├── linear.cuh
│   │   ├── positional_encoding.cuh
│   │   ├── relu.cuh
│   │   ├── softmax.cuh
│   │   └── transformer_block.cuh
│   ├── training/
│   │   ├── adamw.cuh
│   │   ├── cross_entropy.cuh
│   │   ├── reduce_lr_on_plateau.hpp
│   │   └── training.cuh
│   └── dataset.hpp
│
├── src/
│   ├── core/
│   ├── data/
│   ├── inference/
│   ├── io/
│   ├── nn/
│   └── training/
│
├── python/
│   └── cugpt/
│       ├── __init__.py
│       ├── __init__.pyi
│       └── loader.py
│
└── tests/
    ├── test_adamw.cpp
    ├── test_attention.cpp
    ├── test_embedding.cpp
    ├── test_feed_forward.cpp
    ├── test_gpt.cpp
    ├── test_layernorm.cpp
    ├── test_linear.cpp
    ├── test_loss.cpp
    ├── test_reduce.cu
    ├── test_relu.cpp
    ├── test_softmax.cpp
    └── test_transformer_block.cpp
```

### Main source areas

| Path | Purpose |
|---|---|
| `include/` | Public C++ headers for the CUDA core, model, training, data, inference, and I/O layers. |
| `src/` | CUDA/C++ implementation files. |
| `bindings/cugpt.cpp` | Nanobind extension that exposes `cugpt.GPT`. |
| `python/cugpt/` | Python package loader and typing stubs around the native extension. |
| `model.py` | Reference GPT implementation and Python checkpoint utilities. |
| `backend.py` | NumPy/CuPy backend abstraction used by the reference implementation. |
| `dataset.py` | Python arithmetic dataset generator. |
| `src/dataset.cpp` + `include/dataset.hpp` | Native C++ arithmetic tokenizer and dataset generator used by the CLI. |
| `src/main.cpp` | `cugpt train` / `cugpt inference` command-line interface. |
| `tests/` | Standalone unit-test executables registered with CTest. |
| `main.ipynb` | End-to-end research notebook: data, training, checkpointing, comparison and generation. |
| `consistency.ipynb` | Numerical equivalence tests between Python and CUDA implementations. |
| `benchmark.ipynb` | Forward-pass performance comparison. |

---

# Architecture

The CUDA model follows this pipeline:

```text
Token IDs [B, T]
      │
      ▼
Embedding [B, T, D]
      │
      ▼
Sinusoidal positional encoding (+ scaled PE)
      │
      ▼
┌───────────────────────────────────────────────────────────┐
│ Transformer Block × N                                     │
│                                                           │
│   x ──► LayerNorm ──► Causal Multi-Head Attention ───┐    │
│   │                                                  │    │
│   └──────────────────── Residual Add ◄───────────────┘    │
│                         │                                 │
│                         ▼                                 │
│                     LayerNorm                             │
│                         │                                 │
│                         ▼                                 │
│                  Linear(D → 4D)                           │
│                         │                                 │
│                        ReLU                               │
│                         │                                 │
│                  Linear(4D → D)                           │
│                         │                                 │
│   residual ◄────────────┴──────────── Residual Add        │
└───────────────────────────────────────────────────────────┘
      │
      ▼
Final LayerNorm
      │
      ▼
Linear(D → V)
      │
      ▼
Logits [B, T, V]
```

Where:

- `B` = batch size.
- `T` = sequence length.
- `D` = model dimension.
- `H` = number of attention heads.
- $\displaystyle D_h=\frac{D}{H}$ = dimension per head.
- `V` = vocabulary size.

The model requires $\displaystyle D\bmod H=0$.

## Default model used by the CLI

The CLI constructs:

```text
blocks             = 2
model_dim          = 128
num_heads          = 4
ffn_hidden         = model_dim * 4 = 512
positional_scale   = 0.1
```

For the default four-operation dataset, the vocabulary has 21 tokens, giving a default model shape of approximately:

```text
Embedding:        [21, 128]
Attention heads:  4 × 32 dimensions
FFN:              128 → 512 → 128
Final projection: 128 → 21
```

These values are hard-coded in `makeModel()` in `src/main.cpp`. If they are changed, the corresponding checkpoint architecture metadata changes as well.

---

# Model Details

## Token embedding

The embedding table has shape:

```text
[vocab_size, model_dim]
```

Embeddings are initialized from a zero-mean normal distribution with standard deviation:

```text
1 / sqrt(model_dim)
```

The CUDA embedding backward kernel uses `atomicAdd`, because multiple sequence positions can reference the same vocabulary row.

## Positional encoding

The project uses sinusoidal positional encoding, not learned positional embeddings.

For dimension `i` and position `p`, the implementation follows the usual even/odd sinusoidal pattern, then multiplies it by a configurable scalar (`positional_scale`) and adds it to the token embeddings.

The default scale is `0.1`.

The positional encoding is deterministic and has no trainable parameters.

## Transformer blocks

Each block is pre-normalized and has two residual branches:

$$
\begin{aligned}
r_1 &= x+\mathop{\text{Attention}}(\mathop{\text{LayerNorm}}(x)),\\
y   &= r_1+\mathop{\text{FFN}}(\mathop{\text{LayerNorm}}(r_1)).
\end{aligned}
$$

The feed-forward network is:

```text
Linear(model_dim, 4 * model_dim)
ReLU
Linear(4 * model_dim, model_dim)
```

Both linear layers include a bias.

## Residual output initialization

The constructor scales the attention output projection and the second FFN projection by:

```text
1 / sqrt(2 * number_of_blocks)
```

For the default two-block model this is `0.5`.

Only the output projection weights are scaled; the internal Q/K/V and first FFN projection initialization is left unchanged.

## LayerNorm

LayerNorm uses:

$$
\begin{aligned}
\mu &= \mathop{\text{mean}}(x),\\
\sigma^2 &= \mathop{\text{mean}}\!\left((x-\mu)^2\right),\\
\mathop{\text{inv\_std}} &= \frac{1}{\sqrt{\sigma^2+10^{-5}}},\\
y &= \gamma\odot(x-\mu)\mathop{\text{inv\_std}}+\beta.
\end{aligned}
$$

The serialized parameter names are `gain` and `bias`, matching the Python reference implementation.

## Multi-head attention

The C++ attention module contains four bias-free linear projections:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$
$$
O=\mathop{\text{Attention}}(Q,K,V)W_O.
$$

The projections are reshaped into:

```text
[B, H, T, Dh]
```

Attention scores are:

$$
S=\frac{QK^{\mathsf T}}{\sqrt{D_h}}.
$$

The attention kernel uses strided batched cuBLAS GEMMs for the matrix multiplications.

## Output projection

After the attention blocks, the model applies:

```text
LayerNorm
Linear(model_dim, vocab_size, bias=True)
```

The result is a tensor of shape:

```text
[B, T, vocab_size]
```

No softmax is applied by the GPT model itself. Cross-entropy computes the softmax probabilities as part of the loss, while generation uses `argmax` directly over the final logits.

---

# Attention and Causality

Causality is implemented differently in the two versions of the model, although they are intended to produce the same behavior.

### CUDA/C++ implementation

`src/nn/attention.cu` uses the C++ `Softmax` module with `causal_mask = true`. The CUDA softmax treats positions $\displaystyle j>t$ as $\displaystyle -\infty$ before normalization.

In other words, token `t` can attend only to positions $\displaystyle 0,\ldots,t$.

### Python implementation

The reference `Softmax` implementation itself is not causal. The causal behavior is supplied by adding an explicit attention mask in the notebook/training helpers:

```python
upper = xp.triu(xp.ones((seq_len, seq_len), dtype=xp.float32), k=1)
mask = xp.where(upper > 0, -xp.inf, 0.0)
mask = mask[xp.newaxis, xp.newaxis, :, :]
```

This distinction matters when calling the reference Python model directly: for autoregressive behavior, pass a causal mask.

---

# Synthetic Arithmetic Dataset

The built-in training task is intentionally deterministic and small enough to make it practical to test the complete training stack.

Supported operations are:

| Configuration name | Token | Operation |
|---|---|---|
| `plus` | `плюс` | `a + b` |
| `minus` | `минус` | `a - b` |
| `multiply` | `умножить на` | `a * b` |
| `divide` | `делить на` | integer division, with exact divisibility enforced by sampling |

Numbers are restricted to $\displaystyle 0,\ldots,999$, and generated results must also remain within $\displaystyle 0,\ldots,999$.

## Number representation

Numbers are represented by place tokens plus digit tokens.

Examples:

```text
0
→ <единицы> ноль
```

```text
5
→ <единицы> пять
```

```text
20
→ <десятки> два <единицы> ноль
```

```text
403
→ <сотни> четыре <единицы> три
```

```text
999
→ <сотни> девять <десятки> девять <единицы> девять
```

Leading zero places are omitted except that zero itself is represented explicitly as an `<единицы>` token followed by `ноль`.

## Expression format

A complete expression is:

```text
<start> NUMBER OPERATION NUMBER равно RESULT <end>
```

For example:

```text
<start> <сотни> четыре <десятки> восемь <единицы> три плюс <десятки> два равно <сотни> четыре <десятки> девять <единицы> три <end>
```

The CLI tokenizer works on the Russian token vocabulary even though the operation-selection API uses English names such as `plus` and `multiply`.

## Dataset sampling rules

### Addition

The generator samples `b` only when $\displaystyle a+b\leq \mathrm{max\_number}$.

### Subtraction

The generator samples $\displaystyle b\leq a$, so results remain non-negative.

### Multiplication

The generator enumerates valid pairs that satisfy $\displaystyle ab\leq \mathrm{max\_number}$ and then samples a subset.

### Division

The generator samples only exact integer divisions ($\displaystyle a\bmod b=0$) and never uses $\displaystyle b=0$.

---

# Data Representation and Target Masking

The training task is next-token prediction.

For every padded sequence:

```text
X = tokens[:-1]
Y = tokens[1:]
```

The label tensor `Y` is then modified so that all positions before the `равно` token are replaced by the padding ID.

That means the loss ignores the prompt portion and learns to produce the result.

Conceptually:

```text
Input:
    <start>  ... expression ...  равно  result  <end>

Targets:
    PAD      ... PAD             PAD    result  <end>
```

`CrossEntropyLoss` is configured with the padding token as `ignore_index` by the CLI.

This is important when interpreting the reported loss and sequence accuracy: the optimization target is the result region, not the entire prompt.

For the default dataset, the longest encoded expression has 22 tokens, so the autoregressive training tensors use a sequence length of 21.

---

# Checkpoint Format

Model checkpoints are stored as **SafeTensors** files.

The native C++ serializer writes:

- all model parameters as `float32` tensors,
- explicit model-architecture metadata,
- deterministic parameter names based on the module hierarchy.

The checkpoint format is versioned through:

```text
cugpt.format = cugpt:1
```

## Required metadata

The following keys are stored:

```text
cugpt.format
cugpt.vocab_size
cugpt.blocks
cugpt.model_dim
cugpt.n_heads
cugpt.positional_scale
```

For the default CLI model, the metadata is conceptually:

```text
cugpt.format             = cugpt:1
cugpt.vocab_size         = 21
cugpt.blocks             = 2
cugpt.model_dim          = 128
cugpt.n_heads            = 4
cugpt.positional_scale   = 0.1
```

## Strict loading behavior

`loadModel()` validates all of the following before copying weights:

1. checkpoint format is exactly supported (`cugpt:1`),
2. vocabulary size matches,
3. number of blocks matches,
4. model dimension matches,
5. number of heads matches,
6. positional scale matches,
7. every expected parameter name exists,
8. no unexpected parameter names are present,
9. every tensor is `float32`,
10. tensor shapes exactly match the destination tensors,
11. element counts and byte ranges are valid.

This strictness is intentional. A partially compatible checkpoint is rejected instead of silently reshaping or dropping weights.

## Parameter naming

The C++ module hierarchy intentionally mirrors the Python naming scheme. Typical names include:

```text
embedding_decoder.embeddings

decoder_blocks.0.layernorm1.gain
decoder_blocks.0.layernorm1.bias

decoder_blocks.0.multi_head_attention.query_linear.weights
decoder_blocks.0.multi_head_attention.key_linear.weights
decoder_blocks.0.multi_head_attention.value_linear.weights
decoder_blocks.0.multi_head_attention.final_projection.weights
decoder_blocks.0.layernorm2.gain
decoder_blocks.0.layernorm2.bias
decoder_blocks.0.ffn.linear1.weights
decoder_blocks.0.ffn.linear1.bias
decoder_blocks.0.ffn.linear2.weights
decoder_blocks.0.ffn.linear2.bias

decoder_blocks.1....

final_layernorm.gain
final_layernorm.bias
final_linear.weights
final_linear.bias
```

This shared naming is what makes the Python and C++ checkpoint utilities interoperable.

---

# Python Reference Implementation

The Python reference implementation lives primarily in:

```text
backend.py
dataset.py
model.py
```

It uses the same high-level architecture as the CUDA implementation.

## Backends

`backend.py` provides:

```python
Device.CPU
Device.GPU
```

The CPU backend uses NumPy.

The GPU backend uses CuPy and checks that at least one CUDA device is available.

Both backends expose:

```python
backend.xp
backend.device
backend.as_numpy(array)
backend.scatter_add(array, indices, values)
```

The model code is written against this small abstraction instead of hard-coding NumPy or CuPy operations.

## Creating the reference model

```python
from backend import Device, get_backend
from model import GPT

backend = get_backend(Device.GPU)

model = GPT(
    vocab_size=21,
    blocks=2,
    model_dim=128,
    n_heads=4,
    positional_scale=0.1,
    backend=backend,
)
```

For CPU execution:

```python
backend = get_backend(Device.CPU)
```

## Forward pass

The reference implementation expects an integer token tensor with shape:

```text
[B, T]
```

For autoregressive execution, supply a causal attention mask as demonstrated in `main.ipynb`, `benchmark.ipynb`, and `consistency.ipynb`.

Example:

```python
import numpy as np

B = 4
T = 16
xb = backend.xp.asarray(
    np.zeros((B, T), dtype=np.int64)
)

upper = backend.xp.triu(
    backend.xp.ones((T, T), dtype=backend.xp.float32),
    k=1,
)
mask = backend.xp.where(upper > 0, -backend.xp.inf, 0.0)
mask = mask[backend.xp.newaxis, backend.xp.newaxis, :, :]

logits = model(xb, mask=mask)
print(logits.shape)  # (4, 16, 21)
```

## Python checkpoint helpers

`model.py` provides:

```python
save_model(model, path)
load_model(model, path)
```

They use the same metadata fields as the C++ implementation and validate the metadata before loading tensors.

---

# Native Python Binding

The native extension is exposed as:

```python
from cugpt import GPT
```

The Python package automatically imports `setup_cuda()` before importing the native module.

`python/cugpt/loader.py` uses `cuda-pathfinder` to locate the CUDA runtime, cuBLAS, and cuBLASLt dynamic libraries. On Windows, the located directories are added with `os.add_dll_directory()`.

## Constructor

```python
GPT(
    vocab_size: int,
    blocks: int,
    model_dim: int,
    n_heads: int,
    positional_scale: float = 0.1,
)
```

Example:

```python
from cugpt import GPT

model = GPT(
    vocab_size=21,
    blocks=2,
    model_dim=128,
    n_heads=4,
)
```

## Loading a checkpoint

```python
model.load("best_model.safetensors")
```

## Forward input requirements

The binding deliberately exposes two overloads with the same Python method name:

```python
model.forward(tokens)
```

### NumPy / CPU path

Input must be:

- NumPy array,
- `int32`,
- 2-dimensional,
- C-contiguous.

The native implementation allocates a CUDA tensor and copies the tokens from host memory to device.

The returned array is:

- NumPy,
- `float32`,
- 3-dimensional,
- C-contiguous.

### CuPy / CUDA path

Input must be:

- CuPy array,
- `int32`,
- 2-dimensional,
- C-contiguous,
- allocated on CUDA device **0**.

The input CuPy allocation is borrowed directly by the CUDA core; no host-to-device token copy is performed.

The returned array is a CuPy `float32` array on device 0.

---

# Requirements

The build system establishes the following core requirements.

## Native build requirements

- CMake **4.4 or newer**.
- A C++17-compatible compiler.
- CUDA Toolkit capable of compiling CUDA C++17 code.
- CUDA runtime and cuBLAS development libraries discoverable by CMake.
- Python **3.10 or newer**.

The CMake project explicitly runs:

```cmake
find_package(CUDAToolkit REQUIRED)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
```

## Python packaging requirements

`pyproject.toml` declares:

```text
scikit-build-core >= 0.10
nanobind >= 2.4
cuda-pathfinder >= 1.8.2
```

The CMake configuration additionally fetches:

- Nanobind `v2.15.0` from GitHub.
- `safetensors-cpp` from the `main` branch of its Git repository.

Because these dependencies are fetched at configure time, the first build normally requires network access unless the sources are already available to CMake.

## About `requirements.txt`

The repository also contains a large pinned `requirements.txt` snapshot containing packages used by the notebook environment (including NumPy, pandas, matplotlib, Jupyter, CuPy, SafeTensors, tqdm, and related tooling).

For the package itself, `pyproject.toml` is the authoritative build metadata. The requirements snapshot should be treated as a notebook/development environment record rather than the minimal installation specification for the native package.

---

# Building the C++/CUDA Project

The project can be built directly with CMake without installing the Python wheel.

## Configure a Release build

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUGPT_BUILD_TESTS=ON
```

By default, if `CMAKE_CUDA_ARCHITECTURES` is not supplied, the project requests:

```text
native
```

You can override it explicitly when targeting a specific GPU generation. For example:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUGPT_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
```

Use the architecture value appropriate for your target CUDA GPU/toolchain.

## Build

```bash
cmake --build build --config Release
```

For single-configuration generators such as the default Unix Makefiles/Ninja style, the executable is normally produced under `build/`.

For Visual Studio multi-configuration builds, it is normally under the `Release` configuration directory.

The CLI target is named `cugpt_cli` but the produced executable is named:

```text
cugpt
```

## Build without tests

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUGPT_BUILD_TESTS=OFF
cmake --build build --config Release
```

This is also the default used by the Python wheel configuration in `pyproject.toml`.

---

# Building and Installing the Python Package

The recommended packaging path is through the `pyproject.toml` / scikit-build-core setup.

## Install from a local checkout

```bash
python -m pip install --upgrade pip
python -m pip install .
```

For an editable development installation:

```bash
python -m pip install -e .
```

The build system will compile the C++/CUDA core and the Nanobind extension and install the extension under the `cugpt` Python package.

## Verify the installation

```bash
python -c "import cugpt; print(cugpt.GPT)"
```

You should get the native `GPT` binding rather than an import error.

## Manual wheel build

A standard Python build frontend can be used as well:

```bash
python -m pip wheel . -w dist
```

Then install the generated wheel with:

```bash
python -m pip install dist/*.whl
```

---

# Running the CLI

The native executable provides two commands:

```text
cugpt train [options]
cugpt inference --checkpoint PATH --text TEXT [options]
```

---

## Show Help

```bash
cugpt --help
```

The CLI also accepts:

```bash
cugpt -h
```

The help text is built directly into `src/main.cpp`.

---

## Train

Minimal training command:

```bash
cugpt train
```

Default behavior:

```text
epochs                   = 15
batch_size               = 64
examples_per_operation   = 20000
seed                     = 42
checkpoint               = best_model.safetensors
learning_rate            = 3e-4
beta1                    = 0.9
beta2                    = 0.999
optimizer_eps            = 1e-8
weight_decay             = 0.01
scheduler                 = disabled
```

A typical explicit command is:

```bash
cugpt train \
  --epochs 15 \
  --batch-size 64 \
  --examples-per-operation 20000 \
  --seed 42 \
  --checkpoint best_model.safetensors \
  --lr 3e-4 \
  --weight-decay 0.01
```

At startup, the CLI reports the generated dataset sizes and sequence length, for example:

```text
train_size=...
val_size=...
sequence_length=...
```

At each configured logging interval it prints training metrics and, when validation is available, validation metrics.

## Best-checkpoint behavior

When a validation split is present, the training loop tracks sequence-level validation accuracy.

Whenever validation accuracy improves, the model is saved to the path passed through `--checkpoint`.

Example:

```text
Saved best checkpoint: best_model.safetensors (val_acc=...)
```

At the end of training, the CLI reports the best validation accuracy observed and the checkpoint path.

## Training history

After training, `trainModel()` attempts to write:

```text
history.csv
```

into the **current working directory**.

The columns are:

```text
step,train_loss,val_loss,val_acc,learning_rate
```

The file is useful for quick plotting in the notebooks.

---

## Inference

Inference requires both a checkpoint and a text prompt:

```bash
cugpt inference \
  --checkpoint best_model.safetensors \
  --text "<сотни> четыре <десятки> восемь <единицы> три плюс <десятки> два равно" \
  --max-new-tokens 10
```

The CLI internally constructs the same default architecture used during training:

```text
blocks           = 2
model_dim        = 128
n_heads          = 4
positional_scale = 0.1
```

It uses:

```text
MathTokenizer({"plus", "minus", "multiply", "divide"})
```

and then loads the checkpoint.

### Prompt handling

The native generation implementation automatically prepends:

```text
<start>
```

to the user-supplied text.

Therefore, the `--text` argument normally should **not** include `<start>` unless you deliberately want it duplicated.

The user prompt is expected to already describe the expression prefix, commonly including the `равно` token so that generation starts from the result position.

### Generation algorithm

Generation is purely greedy:

1. encode the prompt,
2. run the full model,
3. read the logits for the last position,
4. choose $\mathop{\text{arg\,max}}(\mathrm{logits}[-1])$,
5. append the token,
6. stop on `<end>` or after `max_new_tokens` steps.

There is no temperature, top-k, top-p, beam search, repetition penalty, or sampling implementation in the current inference code.

### Inference seed

The CLI parser accepts an inference seed field internally, but the current inference path does not use a random seed because decoding is deterministic greedy argmax.

---

# Training Configuration

`src/main.cpp` exposes the following training options.

| Option | Default | Meaning |
|---|---:|---|
| `--epochs` | `15` | Number of training epochs. |
| `--batch-size` | `64` | Samples per training batch. |
| `--examples-per-operation` | `20000` | Number of unique samples requested for each operation. |
| `--seed` | `42` | Dataset/training shuffle seed. |
| `--checkpoint` | `best_model.safetensors` | Output path for the best validation checkpoint. |
| `--lr` | `3e-4` | AdamW learning rate. |
| `--beta1` | `0.9` | Adam first-moment coefficient. |
| `--beta2` | `0.999` | Adam second-moment coefficient. |
| `--optimizer-eps` | `1e-8` | Numerical stability epsilon. |
| `--weight-decay` | `0.01` | Decoupled AdamW weight decay. |
| `--lr-scheduler` | `none` | Scheduler mode. |
| `--lr-scheduler-monitor` | `val-loss` | Metric used by the scheduler. |
| `--lr-scheduler-mode` | `min` | Whether lower or higher values are considered better. |
| `--lr-scheduler-factor` | `0.1` | Multiplicative LR reduction factor. |
| `--lr-scheduler-patience` | `10` | Allowed non-improving intervals before reduction. |
| `--lr-scheduler-threshold` | `1e-4` | Minimum significant change. |
| `--lr-scheduler-threshold-mode` | `rel` | Relative or absolute threshold. |
| `--lr-scheduler-cooldown` | `0` | Number of intervals to wait after a reduction. |
| `--lr-scheduler-min-lr` | `0` | Lower LR bound. |
| `--lr-scheduler-eps` | `1e-8` | Minimum change required to count as an LR update. |

## Important scheduler detail

The scheduler's default monitor is validation loss and the default mode is `min`, which is consistent.

If you monitor validation accuracy, you should switch the scheduler mode to `max`:

```bash
cugpt train \
  --lr-scheduler reduce-lr-on-plateau \
  --lr-scheduler-monitor val-accuracy \
  --lr-scheduler-mode max
```

---

# Learning-Rate Scheduler

`ReduceLROnPlateau` is implemented natively in C++.

Supported modes:

```text
min
max
```

Supported threshold modes:

```text
rel
abs
```

Example:

```bash
cugpt train \
  --lr-scheduler reduce-lr-on-plateau \
  --lr-scheduler-monitor val-loss \
  --lr-scheduler-mode min \
  --lr-scheduler-factor 0.5 \
  --lr-scheduler-patience 3 \
  --lr-scheduler-threshold 1e-4 \
  --lr-scheduler-threshold-mode rel \
  --lr-scheduler-cooldown 1 \
  --lr-scheduler-min-lr 1e-6
```

The scheduler is stepped once per training epoch.

It reduces the AdamW learning rate multiplicatively, but never below `min_lr`.

The scheduler reports an update such as:

```text
ReduceLROnPlateau: learning_rate 0.0003 -> 0.00015
```

---

# Python API Examples

## Train the reference implementation

The notebook uses the following general pattern:

```python
import numpy as np

from backend import Device, get_backend
from dataset import build_math_dataset
from model import GPT

backend = get_backend(Device.GPU)

data = build_math_dataset(
    max_number=999,
    include_operations=["plus", "minus", "multiply", "divide"],
    seed=42,
    val_fraction=0.2,
    examples_per_operation=20_000,
)

model = GPT(
    vocab_size=len(data.vocab),
    blocks=2,
    model_dim=128,
    n_heads=4,
    backend=backend,
)
```

The notebook then trains using its helper function, evaluates sequence-level accuracy, and writes `best_model.safetensors` whenever validation accuracy improves.

## Load the native model from Python

```python
from cugpt import GPT

model = GPT(
    vocab_size=21,
    blocks=2,
    model_dim=128,
    n_heads=4,
)

model.load("best_model.safetensors")
```

### CPU input to native CUDA model

```python
import numpy as np

prompt_ids = np.asarray([[1, 5, 6, 10, 20]], dtype=np.int32)
logits = model.forward(prompt_ids)

print(logits.shape)
print(logits.dtype)
```

The native implementation copies the NumPy input to GPU and returns a NumPy array after copying logits back.

### CuPy input to native CUDA model

```python
import cupy as cp

prompt_ids = cp.asarray([[1, 5, 6, 10, 20]], dtype=cp.int32)
logits = model.forward(prompt_ids)

print(logits.shape)
print(logits.dtype)
```

For the CuPy path, tokens are consumed directly from the device allocation and logits remain on the GPU.

---

# CuPy / Zero-Copy CUDA Path

The binding has two distinct data-transfer paths.

## NumPy path

```text
NumPy int32
    │
    │ host-to-device copy
    ▼
CUDA IntTensor
    │
    ▼
GPT forward
    │
    │ device-to-host copy
    ▼
NumPy float32 logits
```

This path is convenient and interoperable, but it necessarily transfers the input and output through host memory.

## CuPy path

```text
CuPy int32 on device 0
    │
    │ borrowed device pointer
    ▼
CUDA IntTensor view
    │
    ▼
GPT forward
    │
    ▼
CuPy float32 logits on device 0
```

No host copy is made for token input or returned logits.

### Synchronization note

The native core owns its own CUDA stream and currently does not expose stream interoperability to Python.

To preserve correctness, the CuPy binding therefore performs device/stream synchronization around foreign CuPy memory and before exposing the result.

That means the CuPy path is zero-copy with respect to host transfers, but it is **not fully asynchronous** with the caller's CuPy stream.

### Device limitation

The current CuPy binding explicitly supports **CUDA device 0 only**.

Passing a CuPy array from another device raises an error.

---

# C++ API Overview

The CUDA core is organized into small composable classes.

## Core

### `cugpt::core::Tensor`

Float tensor storage on CUDA device memory.

### `cugpt::core::IntTensor`

Integer tensor storage used for token IDs and integer targets.

### `cugpt::core::CudaContext`

Owns the CUDA execution context/stream and cuBLAS handle used by the model.

### `cugpt::core::Parameter`

Stores a trainable tensor and its gradient tensor.

### `cugpt::core::Module`

Provides:

- recursive module registration,
- flat parameter lists for optimizers,
- hierarchical parameter names for serialization,
- `zeroGrad()`.

## Neural network modules

Important classes include:

```text
cugpt::nn::Embedding
cugpt::nn::PositionalEncoding
cugpt::nn::LayerNorm
cugpt::nn::Linear
cugpt::nn::ReLU
cugpt::nn::Softmax
cugpt::nn::MultiHeadAttention
cugpt::nn::FeedForwardNetwork
cugpt::nn::TransformerBlock
cugpt::nn::GPT
```

## Training

```text
cugpt::training::CrossEntropyLoss
cugpt::training::AdamW
cugpt::training::ReduceLROnPlateau
cugpt::training::trainStep
cugpt::training::trainModel
cugpt::training::evaluate
```

## Serialization

```text
cugpt::io::saveModel
cugpt::io::loadModel
```

## Inference

```text
cugpt::inference::generateTokenIds
cugpt::inference::generateText
```

---

# Testing

The CMake project can build a test executable for each file under `tests/` and registers each executable with CTest.

## Configure with tests enabled

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUGPT_BUILD_TESTS=ON
```

## Build

```bash
cmake --build build --config Release
```

## Run all tests

```bash
ctest --test-dir build --output-on-failure
```

For multi-configuration generators such as Visual Studio, specify the configuration:

```bash
ctest --test-dir build -C Release --output-on-failure
```

## Current unit-test coverage

The repository contains tests for:

| Test | Component |
|---|---|
| `test_adamw` | AdamW optimizer updates and parameter behavior. |
| `test_attention` | Multi-head attention forward/backward path. |
| `test_embedding` | Embedding lookup and gradient accumulation. |
| `test_feed_forward` | Feed-forward block. |
| `test_gpt` | End-to-end model forward/backward behavior. |
| `test_layernorm` | LayerNorm forward/backward and gradients. |
| `test_linear` | Linear layer forward/backward and parameter gradients. |
| `test_loss` | Cross-entropy value and gradient. |
| `test_reduce` | CUDA reduction utility. |
| `test_relu` | ReLU forward/backward. |
| `test_softmax` | Softmax, causal masking and backward path. |
| `test_transformer_block` | Residual Transformer block and gradients. |

Many tests use hand-computed expected tensors and strict absolute tolerances, which makes them useful for catching low-level CUDA math regressions.

---

# Numerical Consistency

`consistency.ipynb` is specifically designed to compare the Python reference implementation and the CUDA implementation.

It loads the **same checkpoint** into both models and compares:

- mean absolute error,
- RMSE,
- maximum absolute error,
- 99th-percentile absolute error,
- relative errors,
- `numpy.allclose(...)`,
- argmax/prediction agreement.

The notebook generates cases covering:

- sequence lengths from 1 through 128,
- multiple batch sizes,
- random token sequences,
- all-zero sequences,
- constant-token sequences,
- maximal token IDs,
- many additional randomized shapes.

The notebook contains assertions requiring:

```text
allclose(atol=1e-4, rtol=1e-4)
argmax agreement >= 99.9%
```

It also performs an equivalence pass over the generated arithmetic dataset.

These checks provide a practical validation workflow for verifying that the C++ and Python implementations agree numerically.

---

# Benchmarking

`benchmark.ipynb` compares forward-pass execution across the reference and native implementations.

It builds a validation-only arithmetic dataset with 20,000 examples per operation and processes it with a batch size of 64.

The benchmark records:

- elapsed time,
- number of examples,
- number of batches,
- examples per second,
- batches per second,
- mean/std/min/max timings.

The notebook compares:

```text
Python GPT + NumPy
Python GPT + CuPy
cuGPT + NumPy input/output
cuGPT + CuPy input/output
```

The exact benchmark result depends on GPU model, CUDA version, compiler, driver, Python/CuPy versions, and runtime environment. The notebook intentionally measures the current machine rather than hard-coding a universal speedup claim.

---

# Reproducing the Notebook Workflow

`main.ipynb` is the best overview of the project from a research/experimentation perspective.

The general workflow is:

```text
1. Build arithmetic dataset
2. Inspect vocabulary and sequence lengths
3. Create reference GPT
4. Train reference model
5. Save best_model.safetensors
6. Plot loss / validation accuracy
7. Load the checkpoint into the Python reference model
8. Load the same checkpoint into native cuGPT
9. Run generation with both implementations
10. Run C++ CLI inference
```

The CLI portion of the notebook uses:

```bash
cugpt train --epochs 15 --batch-size 64 --checkpoint best_model.safetensors
```

and later:

```bash
cugpt inference --checkpoint best_model.safetensors --text "..."
```

The same checkpoint is then reused for consistency and benchmark experiments.

---

# Performance Considerations

## CUDA execution

The native implementation performs the actual tensor operations on the GPU and uses CUDA kernels for operations such as:

- embedding lookup,
- embedding gradient accumulation,
- LayerNorm,
- Softmax,
- ReLU,
- positional encoding,
- tensor add/scale operations,
- cross-entropy computation and reduction,
- AdamW updates.

Attention matrix products use cuBLAS strided batched GEMM operations.

## Memory format

The implementation uses contiguous row-major tensor storage. The Nanobind binding requires C-contiguous arrays so that native kernels can consume the pointers directly.

## Synchronization

The core is designed around a CUDA context with its own stream. Some host-visible operations synchronize that stream, including copying logits back to Python NumPy arrays and reading the scalar loss to the host.

The CuPy binding also synchronizes around interoperability because it currently does not share arbitrary caller streams with the native core.

## Autoregressive generation cost

Current generation recomputes the entire model for every new token:

```text
prompt of length T
→ forward(T)
→ append one token
→ forward(T+1)
→ append one token
→ ...
```

There is no KV cache.

For short arithmetic sequences this is perfectly workable, but for long contexts it is much less efficient than an implementation with cached keys/values.

---

# Important Limitations

This section is intentionally explicit because the repository is an experimental CUDA implementation rather than a production LLM runtime.

## 1. Greedy-only generation

Generation selects the highest-logit token with `argmax`.

There is currently no:

- temperature,
- top-k sampling,
- top-p sampling,
- beam search,
- repetition penalty,
- frequency penalty,
- presence penalty.

## 2. No KV cache

Every generation step recomputes attention for the entire current sequence.

## 3. Fixed application tokenizer

The CLI is tied to `MathTokenizer` and the included synthetic arithmetic vocabulary.

The model core itself is generic, but the CLI is not a general natural-language tokenizer pipeline.

## 4. Fixed CLI architecture

The current CLI always constructs the model with:

```text
2 blocks
128 model dimensions
4 attention heads
0.1 positional scale
```

It therefore relies on the checkpoint to match this architecture.

The underlying `GPT` C++ class is configurable, but `src/main.cpp` does not currently expose those dimensions as command-line options.

## 5. CUDA device 0 in Python binding

The CuPy binding explicitly rejects device IDs other than `0`.

The CLI/core does not provide a multi-GPU orchestration layer.

## 6. No mixed precision

The native serialization and core parameters are float32. There is no AMP, FP16, BF16, tensor-core-specific mixed precision path, or loss scaling implementation in this repository.

## 7. No distributed training

There is no data parallel, model parallel, NCCL, distributed optimizer, or multi-process training support.

## 8. Simple host-side batching

The training loop constructs host vectors for selected samples and then copies each batch to GPU memory. There is no asynchronous data loader, pinned-memory pipeline, prefetch queue, or overlap of host preprocessing with GPU computation.

## 9. No configurable early stopping

Training always runs for the requested number of epochs. The best checkpoint is saved, but there is no early-stopping termination criterion.

## 10. Checkpoint format is intentionally strict

A checkpoint with missing or extra tensor names, different tensor shapes, or mismatched architecture metadata is rejected rather than partially loaded.

This is good for correctness but means architecture edits require coordinated checkpoint migration or retraining.

## 11. External dependency fetching

CMake fetches `nanobind` and `safetensors-cpp` during configuration. Offline builds therefore require the dependencies to be made available to CMake in advance.

## 12. SafeTensors backend supports float32 model tensors

The custom C++ SafeTensors reader/writer explicitly accepts model tensors with `float32` dtype. Other weight dtypes are not accepted by the loading path.

---

# Troubleshooting

## `Could not find CUDAToolkit`

Make sure the CUDA Toolkit developer installation is available and that CMake can find it.

Typical checks:

```bash
nvcc --version
cmake --system-information | grep CUDA
```

On Windows, verify that the CUDA Toolkit is installed for the compiler/toolchain you are using and that CMake can locate it.

## CMake cannot fetch dependencies

The project fetches dependencies with `FetchContent`.

If configuration fails while cloning Nanobind or `safetensors-cpp`, check:

- Internet connectivity.
- Git availability.
- Proxy/firewall configuration.
- Whether your build environment permits fetching source dependencies.

For reproducible/offline builds, pre-populate the dependencies or vendor them rather than relying on network fetches.

## Python import fails inside `import cugpt`

The Python package performs CUDA dynamic-library discovery during import.

Verify that:

- CUDA libraries are installed,
- the `cuda-pathfinder` dependency is installed,
- the native extension was actually built and installed.

Try:

```bash
python -c "from cugpt import GPT; print(GPT)"
```

## CuPy says no GPU device is available

The `CudaBackend` checks:

```python
cp.cuda.runtime.getDeviceCount()
```

and raises if no CUDA device is found.

Check the GPU/driver installation separately before debugging the model code.

## CuPy binding rejects a token array

The native binding requires:

```text
int32
2 dimensions
C-contiguous
CUDA device 0
```

For example:

```python
tokens = cp.asarray(tokens, dtype=cp.int32)
tokens = cp.ascontiguousarray(tokens)
```

## Checkpoint architecture mismatch

The loader validates:

```text
vocab_size
blocks
model_dim
n_heads
positional_scale
```

Make sure the model object was constructed with exactly the architecture used to create the checkpoint.

For the default CLI checkpoint:

```python
GPT(
    vocab_size=21,
    blocks=2,
    model_dim=128,
    n_heads=4,
    positional_scale=0.1,
)
```

## Missing or unexpected checkpoint tensors

The serializer compares the exact parameter-name set between the model and checkpoint.

This usually means one of the following happened:

- a module was renamed,
- a parameter was added/removed,
- the module hierarchy changed,
- a checkpoint belongs to a different model implementation/version.

## Inference output looks wrong

Check the prompt format first.

The CLI automatically prepends:

```text
<start>
```

and expects the remainder to be compatible with `MathTokenizer`.

A typical arithmetic generation prompt ends at `равно`:

```text
<сотни> четыре <десятки> восемь <единицы> три плюс <десятки> два равно
```

Also remember that the current generator is greedy and deterministic.

## Loss or accuracy behaves unexpectedly

The dataset masks labels before `равно` with the padding ID. The loss is therefore focused on the result section.

Also note that validation **sequence accuracy** is all-or-nothing per example: a sequence counts as correct only when all non-ignored target positions are predicted correctly.

## Scheduler moves in the wrong direction

Check the metric and scheduler mode together.

For:

```text
train-loss
val-loss
```

use:

```text
mode = min
```

For:

```text
val-accuracy
```

use:

```text
mode = max
```

---

# Serialization / Compatibility Rules

The checkpoint contract is deliberately more strict than a typical loosely typed Python state dictionary.

A valid cuGPT checkpoint requires all of the following to stay aligned:

```text
architecture
parameter names
parameter shapes
parameter dtype
positional scale
vocabulary size
```

## Why strict metadata exists

Without architecture metadata, a checkpoint could accidentally be loaded into a model with a different:

- number of Transformer blocks,
- model dimension,
- number of heads,
- vocabulary size,
- positional scaling scheme.

The native loader fails early instead of producing silently corrupted predictions.

## Format versioning

The current format is:

```text
cugpt:1
```

If the serialization contract is changed incompatibly, the format version should be incremented and the loader should reject older/incompatible formats explicitly.

---

# End-to-End Quick Start

This section is the shortest path from a fresh checkout to a trained checkpoint and a generation run.

## 1. Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUGPT_BUILD_TESTS=ON

cmake --build build --config Release
```

## 2. Test

```bash
ctest --test-dir build --output-on-failure
```

## 3. Train

```bash
./build/cugpt train \
  --epochs 15 \
  --batch-size 64 \
  --checkpoint best_model.safetensors
```

On Windows multi-config builds, the executable may instead be located at a path similar to:

```text
build/Release/cugpt.exe
```

## 4. Generate

```bash
./build/cugpt inference \
  --checkpoint best_model.safetensors \
  --text "<сотни> четыре <десятки> восемь <единицы> три плюс <десятки> два равно" \
  --max-new-tokens 10
```

## 5. Load the same checkpoint from Python

```python
from cugpt import GPT

model = GPT(
    vocab_size=21,
    blocks=2,
    model_dim=128,
    n_heads=4,
)
model.load("best_model.safetensors")
```

This is the core demonstration of the repository: the same model architecture and checkpoint format are shared between the native CUDA runtime and the Python-facing workflow.