from enum import Enum


class ListEnum(Enum):

    @classmethod
    def list_value(self):
        return [item.value for item in list(self)]
