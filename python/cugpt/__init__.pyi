from typing import Protocol, Any

class Array(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> Any: ...

class GPT:
    def __init__(
        self,
        vocab_size: int,
        blocks: int,
        model_dim: int,
        n_heads: int,
        positional_scale: float = 0.1,
    ) -> None: ...
    def load(self, path: str) -> None: ...
    def forward(self, tokens: Array) -> Array: ...
