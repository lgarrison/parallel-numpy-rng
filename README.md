# parallel-numpy-rng
[![tests](https://github.com/lgarrison/parallel-numpy-rng/actions/workflows/test.yml/badge.svg)](https://github.com/lgarrison/parallel-numpy-rng/actions/workflows/test.yml)

A multi-threaded random number generator backed by Numpy RNG, with parallelism provided by Numba.

## Overview
Uses the "fast-forward" capability of the PCG-family of RNG, as exposed by the
new-style Numpy RNG API, to generate random numbers in a multi-threaded manner. The key
is that you get the same random numbers regardless of how many threads were used.

Only a two types of random numbers are supported right now: uniform and normal. More
could be added if there is demand, although some kinds, like bounded random ints, are
hard to parallelize in the approach used here.

## Example + Performance
```python
import numpy as np
import parallel_numpy_rng

seed = 123
parallel_rng = parallel_numpy_rng.default_rng(seed)
numpy_rng = np.random.default_rng(seed)

%timeit numpy_rng.random(size=10**9, dtype=np.float32)                           # 2.85 s
%timeit parallel_rng.random(size=10**9, dtype=np.float32, nthread=1)             # 3.34 s
%timeit parallel_rng.random(size=10**9, dtype=np.float32, nthread=128)           # 67.8 ms

%timeit numpy_rng.standard_normal(size=10**8, dtype=np.float32)                  # 1.12 s
%timeit parallel_rng.standard_normal(size=10**8,dtype=np.float32, nthread=1)     # 1.85 s
%timeit parallel_rng.standard_normal(size=10**8, dtype=np.float32, nthread=128)  # 43.5 ms
```

Note that the `parallel_rng` is slower than Numpy when using a single thread, because the parallel implementation requires a slower algorithm in some cases (this is especially noticeable for float64 and normals)

## Installation
The code works and is [reasonably well tested](./test_parallel_numpy_rng.py), so it's probably ready for use.  It can be installed from PyPI:
```console
$ pip install parallel-numpy-rng
```
