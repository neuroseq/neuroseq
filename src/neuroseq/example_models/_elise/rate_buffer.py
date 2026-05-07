""" Buffer module. """

# /usr/bin/env python3


from abc import ABC, abstractmethod
from numbers import Number

import numpy as np
import numpy.typing as npt

Array = npt.NDArray

DTYPE = np.float64


class BufferBase(ABC):
    """Define the interface for the buffer."""

    __slots__ = ["size", "depth", "buf"]

    def __init__(self, size: int, depth: int, val: Number | Array) -> None:
        """Initialize the buffer with given parameters.

        :param size: int, Size of the buffer
        :param depth: int, Depth of the buffer
        :param val: Number or Array, Initial value to fill the buffer
        """
        self.size = size
        self.depth = depth
        self.buf = np.array([])  # Override me!

    def __repr__(self) -> str:
        """Repr method."""
        return f"buffer of size {self.size}, depth {self.depth}:\n{self.buf}"  # noqa

    @abstractmethod
    def __getitem__(self, key):
        """Interface for getitem."""
        pass

    @abstractmethod
    def roll(self, val: Array) -> None:
        """Interface for the roll operation."""
        pass

    @abstractmethod
    def get(self, delay: Array) -> Array:
        """Interface for the get operation."""
        pass

    # TODO: Maybe we want to add some dunders like __add__
    # But this might be an overkill...
    # Let's see if it's practical in the code...


class Buffer(BufferBase):
    """Rolling buffer with inhomogeneous access."""

    __slots__ = ["size", "depth", "buf", "_i", "_size_range"]

    def __init__(self, size: int, depth: int, val: Number | npt.NDArray) -> None:
        """Initialize the buffer with given parameters.

        :param size: int, Size of the buffer
        :param depth: int, Depth of the buffer
        :param val: Number or npt.NDArray, Initial value(s) to fill the buffer
        """
        super().__init__(size, depth, val)

        self._i = 0
        self._size_range = np.arange(self.size)

        if isinstance(val, Number):
            self.buf = np.full((depth, size), val, dtype=DTYPE)
        elif isinstance(val, np.ndarray):
            if val.shape != (size,):
                raise ValueError(
                    f"'val' should have the shape {(self.size,)} but has {val.shape}."
                )
            self.buf = np.tile(val, (depth, 1)).astype(DTYPE)

    def _assemble(self) -> npt.NDArray:
        """Reassemble the internal buffer to be contiguous."""
        idx = (self._i + 1) % self.depth
        assembled_buffer = np.concatenate([self.buf[idx:], self.buf[:idx]])
        return assembled_buffer

    def __getitem__(self, key: slice | int | tuple[int, int]) -> npt.NDArray | float:
        """Access the ordered buffer.

        Always returns a copy, since the buffer is non-contiguously stored
        and needs to be reassembled.
        """
        assembled_buffer = self._assemble()
        return assembled_buffer[key]

    def roll(self, val: npt.NDArray) -> None:
        """Insert a new values to the Buffer.

        :param val: numpy array, Value to be inserted
        :return: None
        """
        if val.shape != (self.size,):
            raise ValueError(
                f"'val' should have the shape {(self.size,)} but has {val.shape}."
            )
        self._i += 1
        idx = self._i % self.depth
        self.buf[idx] = val

    def get(self, delay: npt.NDArray) -> npt.NDArray:
        """Get the current delayed values of the buffer.

        The values in delay must be between 0 and depth.
        Delay stores the delays as integer indices.

        :param delay: numpy array, Array of delay values as indexes

        :return: numpy array, current delayed values
        """
        if delay.shape != (self.size,):
            raise ValueError(
                f"'delay' should have the shape {(self.size,)} but has {delay.shape}."
            )
        if not np.issubdtype(delay.dtype, np.integer):
            raise ValueError(
                f"'delay' should have the dtype 'interger', but has {delay.dtype}."
            )
        idx = (-delay + self._i) % self.depth
        return self.buf[idx, self._size_range]
