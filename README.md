# Parallelization of computations in Python

## Description
The aim of this project was to compare performance of seqential and parrel approach to data processing in python.

Subtasks:
- Artificial dataset generator.
```bash
$ python3 scripts/data-generator.py --num-samples <num-samples> --out-dir </path/to/datasets>
```
- 4 versions od linear regression models:
    - Sequential computations (baseline)
    - Numpy
    - Threaded computation parallelization
    - Process-based computation parallelization
- Generate plots, which show the execution times of the above models with respect to the size of the dataset
```bash
$ PYTHONPATH=. python3 scripts/run-experiments.py --datasets-dir </path/to/datasets>
```
- Code passes tox tests 
```bash
$ tox -v
```

