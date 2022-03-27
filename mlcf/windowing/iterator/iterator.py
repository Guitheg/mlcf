from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


__all__ = [
    "WindowIterator"
]


# TODO (doc) review
# TODO (enhancement) Window Iterator Time Series implementation
class WindowIterator(ABC):
    """WindowIterator is an abstract class.
    Every class which attempt to iterate over a dataframe with window
    should be inherit from this WindowIterator.

    Attributes:
        __window_index (int): the window index used to iterate over the window.
    """
    def __init__(self):
        self.__window_index = 0

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self) -> WindowIterator:
        self.__window_index = 0
        return self

    def __next__(self):
        if self.__window_index < len(self):
            item = self[self.__window_index]
            self.__window_index += 1
            return item
        else:
            raise StopIteration

    @property
    @abstractmethod
    def n_window(self) -> int:
        pass

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @property
    @abstractmethod
    def features(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @abstractmethod
    def copy(self) -> WindowIterator:
        pass

    @classmethod
    @abstractmethod
    def read(self, filepath: Path) -> WindowIterator:
        pass

    @abstractmethod
    def write(self, dirpath: Path, filename: str) -> Path:
        pass
