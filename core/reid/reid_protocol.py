from typing import Protocol, Any


class ReId(Protocol):

    def extrace_embedding(self, frame, box) -> Any:
        ...