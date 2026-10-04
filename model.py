import math
from typing import Callable, Generator

import numpy as np
import numpy.typing as npt
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

xp: np = None
scatter_add: Callable[[npt.NDArray, npt.NDArray, npt.NDArray], None] = None
as_numpy: Callable[[npt.NDArray], npt.NDArray] = None


class Parameter:
    def __init__(self, data: npt.NDArray):
        self.data = data
        self.grad = xp.zeros_like(data)


class Module:
    @property
    def parameters(self) -> Generator[Parameter, None, None]:
        for _, parameter in self.named_parameters():
            yield parameter

    def named_parameters(
        self, prefix=""
    ) -> Generator[tuple[str, Parameter], None, None]:
        for name, value in self.__dict__.items():
            if value is None:
                continue

            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(value, Parameter):
                yield full_name, value
            elif isinstance(value, Module):
                yield from value.named_parameters(full_name)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        item_name = f"{full_name}.{i}"

                        yield from item.named_parameters(item_name)

    def trainable_parameters(self):
        return sum(parameter.data.size for parameter in self.parameters)

    def state_dict(self) -> dict[str, npt.NDArray]:
        state = {}

        for name, parameter in self.named_parameters():
            state[name] = xp.copy(parameter.data)

        return state

    def load_state_dict(self, state_dict: dict[str, npt.NDArray]):
        model_parameters = dict(self.named_parameters())

        missing_keys = [name for name in model_parameters if name not in state_dict]

        unexpected_keys = [name for name in state_dict if name not in model_parameters]

        if missing_keys or unexpected_keys:
            raise ValueError(
                "State dict mismatch:\n"
                f"Missing keys: {missing_keys}\n"
                f"Unexpected keys: {unexpected_keys}"
            )

        for name, parameter in model_parameters.items():
            value = state_dict[name]

            if parameter.data.shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"model={parameter.data.shape}, "
                    f"state_dict={value.shape}"
                )

            parameter.data[...] = value

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad[...] = 0


class ReLU(Module):
    _inputs: npt.NDArray = None

    def __init__(self):
        pass

    def __call__(self, inputs: npt.NDArray) -> npt.NDArray:
        self._inputs = inputs

        return xp.maximum(0, inputs)

    def backward(self, grad: npt.NDArray) -> npt.NDArray:
        return xp.where(self._inputs > 0, grad, 0)


class Softmax(Module):
    _outputs: npt.NDArray = None

    def __init__(self):
        pass

    def __call__(self, inputs: npt.NDArray) -> npt.NDArray:
        max_logit = xp.max(inputs, axis=-1, keepdims=True)

        self._outputs = xp.exp(inputs - max_logit) / xp.sum(
            xp.exp(inputs - max_logit), axis=-1, keepdims=True
        )

        return self._outputs

    def backward(self, grad: npt.NDArray) -> npt.NDArray:
        return grad * self._outputs - self._outputs * xp.sum(
            grad * self._outputs, axis=-1, keepdims=True
        )


class PositionalEncoding(Module):
    def __init__(self, embedding_length: int, scale=1.0):
        self._embedding_length = embedding_length
        self._scale = scale

    def __call__(self, inputs: npt.NDArray) -> npt.NDArray:
        T = inputs.shape[1]

        pos = xp.arange(0, T, dtype=int)[:, xp.newaxis]

        denominator = (
            xp.arange(0, self._embedding_length) // 2 * 2
        ) / self._embedding_length
        denominator = 1 / (10_000**denominator)
        denominator = denominator[xp.newaxis, ...]

        inner = pos * denominator
        pe_sin = xp.sin(inner)
        pe_cos = xp.cos(inner)

        pe_sin[:, xp.arange(1, self._embedding_length, 2)] = 0
        pe_cos[:, xp.arange(0, self._embedding_length, 2)] = 0

        positional_encoding = pe_sin + pe_cos

        return inputs + positional_encoding[xp.newaxis, ...] * self._scale

    def backward(self, grad: npt.NDArray) -> npt.NDArray:
        return grad


class EmbeddingLayer(Module):
    _inputs: npt.NDArray = None

    def __init__(self, vocab_size: int, embedding_dim: int):
        scale = 1.0 / math.sqrt(embedding_dim)

        self.embeddings = Parameter(xp.random.randn(vocab_size, embedding_dim) * scale)

    def __call__(self, inputs: npt.NDArray) -> npt.NDArray:
        self._inputs = inputs
        return self.embeddings.data[inputs]

    def backward(self, grad: npt.NDArray):
        scatter_add(self.embeddings.grad, self._inputs, grad)


class Linear(Module):
    _inputs: npt.NDArray = None

    def __init__(self, in_features: int, out_features: int, include_bias=True):
        limit = math.sqrt(6 / (in_features))

        self.weights = Parameter(
            xp.random.uniform(-limit, limit, (in_features, out_features))
        )
        self.bias = Parameter(xp.zeros((1, out_features))) if include_bias else None

        self.include_bias = include_bias

    def __call__(self, inputs: npt.NDArray) -> npt.NDArray:
        self._inputs = inputs

        out = xp.matmul(inputs, self.weights.data)

        if self.include_bias:
            out += self.bias.data

        return out

    def backward(self, grad: npt.NDArray) -> npt.NDArray:
        inputs_flat = self._inputs.reshape(-1, self._inputs.shape[-1])
        grad_flat = grad.reshape(-1, grad.shape[-1])

        dL_dA = xp.matmul(grad, self.weights.data.T)
        dL_dW = xp.matmul(inputs_flat.T, grad_flat)

        if self.include_bias:
            dL_dB = xp.sum(grad_flat, axis=0, keepdims=True)
            self.bias.grad += dL_dB

        self.weights.grad += dL_dW

        return dL_dA


class FeedForwardNetwork(Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        self.linear1 = Linear(in_features, hidden_features)
        self.activation = ReLU()
        self.linear2 = Linear(hidden_features, out_features)

    def __call__(self, inputs: npt.NDArray) -> npt.NDArray:
        x = self.linear1(inputs)
        x = self.activation(x)
        x = self.linear2(x)

        return x

    def backward(self, grad: npt.NDArray) -> npt.NDArray:
        grad = self.linear2.backward(grad)
        grad = self.activation.backward(grad)
        grad = self.linear1.backward(grad)

        return grad


class LayerNorm(Module):
    _mean: npt.NDArray = None
    _mu: npt.NDArray = None
    _var: npt.NDArray = None
    _std: npt.NDArray = None

    _outputs: npt.NDArray = None

    def __init__(self, in_features: int):
        self._in_features = in_features

        self.gain = Parameter(xp.ones((1, in_features)))
        self.bias = Parameter(xp.zeros((1, in_features)))

    def __call__(self, inputs: npt.NDArray) -> npt.NDArray:
        epsilon = 1e-5

        self._mean = (1 / self._in_features) * xp.sum(inputs, axis=-1, keepdims=True)
        self._mu = inputs - self._mean
        self._var = (1 / self._in_features) * xp.sum(
            xp.square(self._mu), axis=-1, keepdims=True
        )
        self._std = xp.sqrt(self._var + epsilon)
        self._outputs = self.gain.data * (self._mu / self._std) + self.bias.data

        return self._outputs

    def backward(self, grad: npt.NDArray) -> npt.NDArray:
        """
        [Reference](https://robotchinwag.com/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/)
        """
        inv_std = 1 / self._std
        inv_std3 = inv_std**3

        n = self._in_features

        g = grad * self.gain.data

        dL_dX = g * inv_std - xp.sum(
            g[..., xp.newaxis]
            / n
            * (
                inv_std[..., xp.newaxis]
                + self._mu[..., xp.newaxis]
                * self._mu[..., xp.newaxis, :]
                * inv_std3[..., xp.newaxis]
            ),
            axis=-2,
        )

        dL_dGain = xp.sum(grad * self._mu * inv_std, axis=(0, 1))
        dL_dB = xp.sum(grad, axis=(0, 1))

        self.gain.grad += dL_dGain
        self.bias.grad += dL_dB

        return dL_dX


class MultiHeadAttention(Module):
    _queries: npt.NDArray = None
    _keys: npt.NDArray = None
    _values: npt.NDArray = None
    _attention_weights: npt.NDArray = None

    def __init__(self, in_features: int, out_features: int, n_heads: int):
        self._in_features = in_features
        self._out_features = out_features
        self._n_heads = n_heads

        self.query_linear = Linear(in_features, out_features, include_bias=False)
        self.key_linear = Linear(in_features, out_features, include_bias=False)
        self.value_linear = Linear(in_features, out_features, include_bias=False)

        self.softmax = Softmax()

        self.final_projection = Linear(out_features, out_features, include_bias=False)

    def __call__(self, inputs: npt.NDArray, mask: npt.NDArray = None) -> npt.NDArray:
        B, T, I = inputs.shape
        D = self._out_features
        H = self._n_heads
        Dh = D // H

        assert I == self._in_features
        assert D % H == 0

        # [B, T, I] @ [I, D] -> [B, T, D]
        queries = self.query_linear(inputs)
        keys = self.key_linear(inputs)
        values = self.value_linear(inputs)

        # [B, T, H, Dh]
        shape = (B, T, H, Dh)

        # [B, T, D] -> [B, T, H, Dh]
        queries = xp.reshape(queries, shape)
        keys = xp.reshape(keys, shape)
        values = xp.reshape(values, shape)

        # [B, T, H, Dh] -> [B, H, T, Dh]
        queries = xp.transpose(queries, axes=[0, 2, 1, 3])
        keys = xp.transpose(keys, axes=[0, 2, 1, 3])
        values = xp.transpose(values, axes=[0, 2, 1, 3])

        # [B, H, T, Dh] -> [B, H, Dh, T]
        t_keys = xp.transpose(keys, axes=[0, 1, 3, 2])

        # [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]
        # S = Q @ K^T
        attention_scores = xp.matmul(queries, t_keys)

        # Divide by sqrt(Dh)
        attention_scores = attention_scores / xp.sqrt(keys.shape[-1])

        if mask is not None:
            attention_scores += mask

        # Softmax
        attention_weights = self.softmax(attention_scores)

        # [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
        # O = A @ V
        final_vector = xp.matmul(attention_weights, values)

        # [B, H, T, Dh] -> [B, T, H, Dh]
        final_vector = xp.transpose(final_vector, axes=[0, 2, 1, 3])

        # [B, T, H, Dh] -> [B, T, D]
        final_vector = xp.reshape(final_vector, [B, T, self._out_features])

        # [B, T, D] @ [D, D] -> [B, T, D]
        final_vector = self.final_projection(final_vector)

        self._queries = queries
        self._keys = keys
        self._values = values
        self._attention_weights = attention_weights

        return final_vector

    def backward(self, grad: npt.NDArray) -> npt.NDArray:
        B, T, D = grad.shape
        H = self._n_heads
        Dh = D // H

        assert D == self._out_features
        assert D % H == 0

        grad = self.final_projection.backward(grad)

        # [B, T, H, Dh]
        shape = (B, T, H, Dh)

        # [B, T, D] -> [B, T, H, Dh]
        grad = xp.reshape(grad, shape)

        # [B, T, H, Dh] -> [B, H, T, Dh]
        grad = xp.transpose(grad, axes=[0, 2, 1, 3])

        # [B, H, T, T] -> [B, H, T, T]
        t_attention_weights = xp.transpose(self._attention_weights, axes=[0, 1, 3, 2])

        # [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
        # dL/dV = A^T @ dL/dO
        dL_dValues = xp.matmul(t_attention_weights, grad)

        # [B, H, T, Dh] -> [B, H, Dh, T]
        t_values = xp.transpose(self._values, axes=[0, 1, 3, 2])

        # [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]
        # dL/dA = dL/dO @ V^T
        dL_dAttention_Weights = xp.matmul(grad, t_values)

        # dL/dS = softmax_backward(dL/dA) / sqrt(Dh)
        dL_dAttention_Scores = self.softmax.backward(dL_dAttention_Weights) / xp.sqrt(
            self._keys.shape[-1]
        )

        # [B, H, T, T] -> [B, H, T, T]
        t_dL_dAttention_Scores = xp.transpose(dL_dAttention_Scores, axes=[0, 1, 3, 2])

        # [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
        # dL/dK = dL/dS^T @ Q
        dL_dKeys = xp.matmul(t_dL_dAttention_Scores, self._queries)

        # [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
        # dL/dQ = dL/dS @ K
        dL_dQueries = xp.matmul(dL_dAttention_Scores, self._keys)

        # [B, H, T, Dh] -> [B, T, H, Dh]
        dL_dValues = xp.transpose(dL_dValues, axes=[0, 2, 1, 3])
        dL_dKeys = xp.transpose(dL_dKeys, axes=[0, 2, 1, 3])
        dL_dQueries = xp.transpose(dL_dQueries, axes=[0, 2, 1, 3])

        # [B, T, H, Dh] -> [B, T, D]
        dL_dValues = xp.reshape(dL_dValues, [B, T, self._out_features])
        dL_dKeys = xp.reshape(dL_dKeys, [B, T, self._out_features])
        dL_dQueries = xp.reshape(dL_dQueries, [B, T, self._out_features])

        dL_dA_value_linear = self.value_linear.backward(dL_dValues)
        dL_dA_key_linear = self.key_linear.backward(dL_dKeys)
        dL_dA_query_linear = self.query_linear.backward(dL_dQueries)

        grad_in = dL_dA_value_linear + dL_dA_key_linear + dL_dA_query_linear

        return grad_in


class TransformerBlock(Module):
    def __init__(self, model_dim: int, n_heads: int, res_weight_init_scale=1.0):
        self.layernorm1 = LayerNorm(model_dim)
        self.multi_head_attention = MultiHeadAttention(model_dim, model_dim, n_heads)

        self.layernorm2 = LayerNorm(model_dim)
        self.ffn = FeedForwardNetwork(model_dim, model_dim * 4, model_dim)

        self.ffn.linear2.weights.data *= res_weight_init_scale
        self.multi_head_attention.final_projection.weights.data *= res_weight_init_scale

    def __call__(self, inputs: npt.NDArray, mask: npt.NDArray = None) -> npt.NDArray:
        out = inputs + self.multi_head_attention(self.layernorm1(inputs), mask=mask)
        out = out + self.ffn(self.layernorm2(out))

        return out

    def backward(self, grad: npt.NDArray) -> npt.NDArray:
        grad = grad + self.layernorm2.backward(self.ffn.backward(grad.copy()))
        grad = grad + self.layernorm1.backward(
            self.multi_head_attention.backward(grad.copy())
        )

        return grad


class GPT(Module):
    def __init__(
        self,
        vocab_size: int,
        blocks: int,
        model_dim: int,
        n_heads: int,
        positional_scale=0.1,
    ):
        self.vocab_size = vocab_size
        self.blocks = blocks
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.positional_scale = positional_scale

        self.embedding_decoder = EmbeddingLayer(vocab_size, embedding_dim=model_dim)
        self.positional_encodings_decoder = PositionalEncoding(
            embedding_length=model_dim, scale=positional_scale
        )

        res_weight_init_scale = (2 * blocks) ** (-0.5)

        self.decoder_blocks = [
            TransformerBlock(model_dim, n_heads, res_weight_init_scale)
            for _ in range(blocks)
        ]

        self.final_layernorm = LayerNorm(model_dim)
        self.final_linear = Linear(model_dim, self.vocab_size)

    def __call__(self, tokens: npt.NDArray, mask: npt.NDArray = None) -> npt.NDArray:
        out = self.embedding_decoder(tokens)
        out = self.positional_encodings_decoder(out)

        for decoder in self.decoder_blocks:
            out = decoder(inputs=out, mask=mask)

        out = self.final_layernorm(out)

        return self.final_linear(out)

    def backward(self, grad: npt.NDArray) -> npt.NDArray:
        grad = self.final_linear.backward(grad)
        grad = self.final_layernorm.backward(grad)

        for decoder in reversed(self.decoder_blocks):
            grad = decoder.backward(grad)

        grad = self.positional_encodings_decoder.backward(grad)
        self.embedding_decoder.backward(grad)

        return grad


def get_metadata(model: GPT):
    metadata = {
        "cugpt.format": "cugpt:1",
        "cugpt.vocab_size": str(model.vocab_size),
        "cugpt.blocks": str(model.blocks),
        "cugpt.model_dim": str(model.model_dim),
        "cugpt.n_heads": str(model.n_heads),
        "cugpt.positional_scale": str(model.positional_scale),
    }

    return metadata


def assert_metadata(model: GPT, metadata: dict[str, str]):
    expected = get_metadata(model)

    for key, value in expected.items():
        assert key in metadata, f"Missing metadata: {key}"
        assert value == metadata[key], (
            f"Invalid metadata {key}: " f"expected {value!r}, got {metadata[key]!r}"
        )


def save_model(model: GPT, path: str):
    state = {
        name: as_numpy(parameter.data).astype(np.float32)
        for name, parameter in model.named_parameters()
    }

    metadata = get_metadata(model)

    save_file(state, path, metadata)


def load_model(model: GPT, path: str):
    with safe_open(path, framework="numpy") as f:
        metadata = f.metadata() or {}

    assert_metadata(model, metadata)

    loaded = load_file(path)

    state = {name: xp.asarray(value) for name, value in loaded.items()}

    model.load_state_dict(state)
