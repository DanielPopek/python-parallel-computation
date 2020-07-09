"""Script for generation of artificial datasets."""
import argparse
from typing import List
from typing import Tuple
import numpy as np
import pandas as pd

XS_RANGE = 10
NOISE_SCALING_FACTOR = 0.9
A_FACTOR = 3
B_FACTOR = 2


def get_args() -> argparse.Namespace:
    """Parses script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-samples',
        required=True,
        help='Number of samples to generate',
        type=int,
    )
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Name of directory to save generated data',
        type=str,
    )

    return parser.parse_args()


def generate_data(num_samples: int) -> Tuple[List[float], List[float]]:
    """Generated X, y with given number of data samples."""
    xs = np.random.uniform(0, XS_RANGE, num_samples)
    noise = np.random.normal(0, NOISE_SCALING_FACTOR, size=(1, num_samples))[0]
    ys = A_FACTOR * xs + B_FACTOR + noise
    return (list(xs), list(ys))


def main() -> None:
    """Runs script."""
    attribites = get_args()
    num_samples, directory = attribites.num_samples, attribites.out_dir
    x, y = generate_data(num_samples)
    df = pd.DataFrame(list(zip(x, y)))
    df.to_csv(directory)
    pass


if __name__ == '__main__':
    main()
