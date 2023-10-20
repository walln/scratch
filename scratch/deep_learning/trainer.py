from typing import Protocol


class Trainer(Protocol):
    def train(self) -> None:
        ...
