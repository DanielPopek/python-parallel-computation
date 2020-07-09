"""Script for generation of artificial datasets."""
from __future__ import annotations
import abc
from typing import List

import numpy as np


class ScikitPredictor(abc.ABC):
    """Replace."""

    @abc.abstractmethod
    def fit(self, X, y):
        """Replace."""
        pass

    @abc.abstractmethod
    def predict(self, X):
        """Replace."""
        pass


class LinearRegression(ScikitPredictor):
    """Replace."""

    def __init__(self):
        """Replace."""
        self._coef = None

    @abc.abstractmethod
    def fit(self, X: List[float], y: List[float]) -> LinearRegression:
        """Replace."""
        pass

    def predict(self, X: List[float]) -> np.ndarray:
        """Replace."""
        if self._coef is None:
            raise RuntimeError('Please fit model before prediction')

        return self._coef[0] + self._coef[1] * np.array(X)
