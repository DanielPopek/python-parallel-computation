"""Script for generation of artificial datasets."""
from typing import List
from lr import base
import numpy as np


class LinearRegressionNumpy(base.LinearRegression):
    """Replace."""

    def fit(self, X: List[float], y: List[float]) -> base.LinearRegression:
        """Replace."""
        xs = np.asarray(X)
        ys = np.asarray(y)
        x_avg = np.sum(xs) / xs.shape[0]
        y_avg = np.sum(ys) / ys.shape[0]
        xs_decreased = xs - x_avg
        ys_decreased = ys - y_avg
        nominator = np.sum(xs_decreased * ys_decreased)
        denominator = np.sum(xs_decreased * xs_decreased)
        a = nominator / denominator
        b = y_avg - a * x_avg
        self._coef = [b, a]
        pass
