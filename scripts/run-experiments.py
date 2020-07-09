"""Script for time measurement experiments on linear regression models."""
import argparse
from typing import List
from typing import Tuple
from typing import Type

import pandas as pd
import matplotlib.pyplot as plt

import lr
import time

MODEL_NAMES = ["NUMPY", "SEQUENTIAL", "THREADS", "PROCESSES"]


def get_args() -> argparse.Namespace:
    """Parses script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets-dir',
        required=True,
        help='Name of directory with generated datasets',
        type=str,
    )

    return parser.parse_args()


def run_experiments(
        models: List[Type[lr.base.LinearRegression]],
        datasets: List[Tuple[List[float], List[float]]],
) -> pd.DataFrame:
    """Runs experimnets and for each datesets size and prepares statistics."""
    result_dataframe = pd.DataFrame(columns=['Model', 'DatasetSize', 'Time'])
    for model in models:
        model_name = MODEL_NAMES[models.index(model)]
        model = model()
        for dataset in datasets:
            start_point = time.time()
            model.fit(X=dataset[0], y=dataset[1])
            time_elapsed = time.time() - start_point
            result_dataframe = result_dataframe.append(
                {'Model': model_name, 'DatasetSize': len(dataset[0]),
                 'Time': time_elapsed}, ignore_index=True)
    result_dataframe.to_csv("results.csv")
    return result_dataframe


def make_plot(results: pd.DataFrame) -> None:
    """Draws a plot."""
    results = results.pivot(index='DatasetSize', columns='Model', values='Time')
    results.plot()
    plt.savefig('results2.png')
    plt.show()
    pass


def main() -> None:
    """Runs script."""
    args = get_args()
    data_path = args.datasets_dir

    models = [
        lr.LinearRegressionNumpy,
        lr.LinearRegressionSequential,
        lr.LinearRegressionThreads,
        lr.LinearRegressionProcess
    ]

    whole_dataset = pd.read_csv(data_path)
    sizes = [1000 * i for i in range(1, 11)]
    datasets = [(whole_dataset[0:size]['0'].tolist(),
                 whole_dataset[0:size]['1'].tolist()) for size in sizes]
    results = run_experiments(models, datasets)
    make_plot(results)


if __name__ == '__main__':
    main()
