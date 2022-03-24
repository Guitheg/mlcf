from __future__ import annotations
from abc import ABC, abstractmethod

# TODO: (doc)

__all__ = [
    "WindowIterator"
]


# TODO: (enhancement) Window Iterator Time Series implementation
class WindowIterator(ABC):
    def __init__(self):
        self.__window_index = 0

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def __iter__(self):
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
    def n_window(self):
        pass

    @property
    @abstractmethod
    def ndim(self):
        pass

    @property
    @abstractmethod
    def features(self):
        pass

    @property
    @abstractmethod
    def width(self):
        pass

    @abstractmethod
    def copy(self):
        pass

    @classmethod
    @abstractmethod
    def read(self, filepath) -> WindowIterator:
        pass

    @abstractmethod
    def write(self, dirpath, filename):
        pass
