from enum import Enum
from enum import IntEnum


class PrettyPrintIntEnum(IntEnum):
    def __str__(self):
        return self.name


class PrettyPrintEnum(Enum):
    def __str__(self):
        return self.name
