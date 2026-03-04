from typing import Protocol, Any
from torch import Tensor


class ReIdEmbeddingModel(Protocol):

    def extract_embedding(self, frame, box) -> Tensor:
        ...