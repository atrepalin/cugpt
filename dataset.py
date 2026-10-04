from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt


@dataclass
class Dataset:
    train_texts: list
    val_texts: list
    x_train: npt.NDArray
    y_train: npt.NDArray
    x_val: npt.NDArray
    y_val: npt.NDArray
    vocab: list[str]
    token_to_id: dict[int, str]
    id_to_token: dict[int, str]
    pad_id: int
    max_len: int
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]


DIGITS = [
    "ноль",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
]

PLACE_TOKENS = {
    100: "<сотни>",
    10: "<десятки>",
    1: "<единицы>",
}

OPERATIONS = {
    "plus": ("плюс", lambda a, b: a + b),
    "minus": ("минус", lambda a, b: a - b),
    "multiply": ("умножить на", lambda a, b: a * b),
    "divide": ("делить на", lambda a, b: a // b),
}


def number_to_tokens(n: int | str):
    n = int(n)
    if not 0 <= n <= 999:
        raise ValueError("number must be in [0, 999]")

    if n == 0:
        return [PLACE_TOKENS[1], DIGITS[0]]

    digits_by_place = [
        (100, (n // 100) % 10),
        (10, (n // 10) % 10),
        (1, n % 10),
    ]

    nonzero_indices = [i for i, (_, digit) in enumerate(digits_by_place) if digit != 0]
    first_nonzero = nonzero_indices[0]
    last_nonzero = nonzero_indices[-1]

    tokens = []
    for place, digit in digits_by_place[first_nonzero : last_nonzero + 1]:
        tokens.extend([PLACE_TOKENS[place], DIGITS[digit]])
    return tokens


def make_expression(a: int, b: int, op_name: str):
    op_token, operation = OPERATIONS[op_name]
    result = operation(a, b)

    if not 0 <= result <= 999:
        raise ValueError("result must be in [0, 999]")

    tokens = (
        ["<start>"]
        + number_to_tokens(a)
        + [op_token]
        + number_to_tokens(b)
        + ["равно"]
        + number_to_tokens(result)
        + ["<end>"]
    )
    return " ".join(tokens)


def _sample_pairs(op_name: str, max_number: int, count: int, rng: np.random.Generator):
    if op_name in {"multiply", "divide"}:
        if op_name == "multiply":
            all_pairs = [
                (a, b)
                for a in range(1, max_number + 1)
                for b in range(max_number // a + 1)
            ]
            all_pairs.extend((0, b) for b in range(max_number + 1))
        else:
            all_pairs = [
                (a, b)
                for a in range(max_number + 1)
                for b in range(1, max_number + 1)
                if a % b == 0
            ]

        all_pairs = list(set(all_pairs))
        rng.shuffle(all_pairs)
        return set(all_pairs[: min(count, len(all_pairs))])

    pairs = set()
    while len(pairs) < count:
        a = int(rng.integers(0, max_number + 1))

        if op_name == "plus":
            b = int(rng.integers(0, max_number - a + 1))
        elif op_name == "minus":
            b = int(rng.integers(0, a + 1))
        else:
            raise ValueError(f"Unknown operation: {op_name}")

        pairs.add((a, b))

    return pairs


def build_math_dataset(
    max_number=999,
    include_operations: list = None,
    seed=42,
    val_fraction=0.1,
    examples_per_operation=20_000,
):
    if max_number != 999:
        raise ValueError("This dataset format supports numbers from 0 to 999.")
    if include_operations is None:
        include_operations = list(OPERATIONS.keys())

    rng = np.random.default_rng(seed)
    examples = []

    for op_name in include_operations:
        if op_name not in OPERATIONS:
            raise ValueError(f"Unknown operation: {op_name}")

        pairs = _sample_pairs(
            op_name,
            max_number=max_number,
            count=examples_per_operation,
            rng=rng,
        )
        examples.extend(make_expression(a, b, op_name) for a, b in pairs)

    rng.shuffle(examples)
    split = int(len(examples) * (1.0 - val_fraction))
    train_texts = examples[:split]
    val_texts = examples[split:]

    special_tokens = ["<pad>", "<start>", "<end>"]
    place_tokens = ["<сотни>", "<десятки>", "<единицы>"]
    operation_tokens = [OPERATIONS[name][0] for name in include_operations]
    structural_tokens = ["равно"]

    vocab = (
        special_tokens + place_tokens + operation_tokens + structural_tokens + DIGITS
    )

    vocab = list(dict.fromkeys(vocab))

    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for token, i in token_to_id.items()}

    token_patterns = sorted(
        [tok for tok in vocab if tok != "<pad>"],
        key=len,
        reverse=True,
    )

    def tokenize(text: str):
        tokens = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue

            matched = False
            for tok in token_patterns:
                if text.startswith(tok, i):
                    tokens.append(tok)
                    i += len(tok)
                    matched = True
                    break

            if not matched:
                raise ValueError(f"Unknown token at position {i}: {text[i:]!r}")

        return tokens

    def encode(text: str):
        return [token_to_id[tok] for tok in tokenize(text)]

    def decode(ids: list[int]):
        return " ".join(
            id_to_token[int(i)] for i in ids if int(i) != token_to_id["<pad>"]
        )

    encoded_train = [encode(text) for text in train_texts]
    encoded_val = [encode(text) for text in val_texts]
    max_len = max(map(len, encoded_train + encoded_val))
    pad_id = token_to_id["<pad>"]

    def pad_sequences(sequences):
        data = np.full((len(sequences), max_len), pad_id, dtype=np.int64)
        for i, seq in enumerate(sequences):
            data[i, : len(seq)] = seq
        return data

    train_seq = pad_sequences(encoded_train)
    val_seq = pad_sequences(encoded_val)

    x_train, y_train = train_seq[:, :-1].copy(), train_seq[:, 1:].copy()
    x_val, y_val = val_seq[:, :-1].copy(), val_seq[:, 1:].copy()

    equals_id = token_to_id["равно"]

    def mask_labels_before_result(labels, sequences):
        for i, seq in enumerate(sequences):
            eq_positions = np.flatnonzero(seq == equals_id)
            if len(eq_positions) != 1:
                raise ValueError(
                    "Each sequence must contain exactly one 'равно' token."
                )

            eq_pos = int(eq_positions[0])

            labels[i, :eq_pos] = pad_id

    mask_labels_before_result(y_train, train_seq)
    mask_labels_before_result(y_val, val_seq)

    return Dataset(
        train_texts=train_texts,
        val_texts=val_texts,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        vocab=vocab,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_id=pad_id,
        max_len=max_len,
        encode=encode,
        decode=decode,
    )
