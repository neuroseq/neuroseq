#!/usr/bin/env python3
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, eta):
        self.eta = eta

    @abstractmethod
    def get_update(self, weights, step):
        raise NotImplementedError("Subclasses must implement this method")


class SimpleUpdater(Optimizer):
    def get_update(self, weights, dwdt):
        return dwdt * self.eta
