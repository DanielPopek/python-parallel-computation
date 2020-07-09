"""Script for generation of artificial datasets."""
from typing import List

from lr import base

import threading


class LinearRegressionThreads(base.LinearRegression):
    """Replace."""

    def mult_summing_function(self, first_collection,
                              second_collection, results, index):
        """Replace."""
        sum = 0
        n = len(first_collection)
        if second_collection is not None:
            for i in range(n):
                sum += first_collection[i] * second_collection[i]
        else:
            for i in range(n):
                sum += first_collection[i]
        results[index] = sum

    def fit(self, X: List[float], y: List[float]) -> base.LinearRegression:
        """Replace."""
        n = len(X)

        results = [None] * 4

        x_sum = threading.Thread(target=self.mult_summing_function,
                                 args=(X, None, results, 0))

        y_sum = threading.Thread(target=self.mult_summing_function,
                                 args=(y, None, results, 1))

        x_squared_sum = threading.Thread(target=self.mult_summing_function,
                                         args=(X, X, results, 2))

        x_y_mult_sum = threading.Thread(target=self.mult_summing_function,
                                        args=(X, y, results, 3))

        x_sum.start()
        y_sum.start()
        x_squared_sum.start()
        x_y_mult_sum.start()

        x_sum.join()
        y_sum.join()
        x_squared_sum.join()
        x_y_mult_sum.join()

        x_sum = results[0]
        y_sum = results[1]
        x_squared_sum = results[2]
        x_y_mult_sum = results[3]

        nominator = (n * x_y_mult_sum - x_sum * y_sum)
        a = nominator / (n * x_squared_sum - x_sum * x_sum)
        b = (y_sum / n) - a * (x_sum / n)
        self._coef = [b, a]
