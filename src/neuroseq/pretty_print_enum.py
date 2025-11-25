from enum import Enum


class PrettyPrintEnum(Enum):
    def __str__(self):
        return self.name
