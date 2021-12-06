# parallel-numpy-rng
![tests](https://github.com/lgarrison/parallel-numpy-rng/actions/workflows/test.yml/badge.svg)

A multi-threaded random number generator backed by Numpy RNG, with parallelism provided by Numba.

## Overview
Uses the "fast-forward" capability of the PCG-family of RNG, as exposed by the
new-style Numpy RNG API, to generate random numbers in a multi-threaded manner. The key
is that you get the same random numbers regardless of how many threads were used.

## Example + Performance
```python
import numpy as np
import parallel_numpy_rng

seed = 123
parallel_rng = parallel_numpy_rng.default_rng(seed)
numpy_rng = np.random.default_rng(seed)

%timeit numpy_rng.random(size=10**9, dtype=np.float32)                           # 2.89 s
%timeit parallel_rng.random(size=10**9, dtype=np.float32, nthread=1)             # 3.35 s
%timeit parallel_rng.random(size=10**9, dtype=np.float32, nthread=128)           # 73.9 ms

%timeit numpy_rng.standard_normal(size=10**8, dtype=np.float32)                  # 1.13 s
%timeit parallel_rng.standard_normal(size=10**8,dtype=np.float32, nthread=1)     # 1.87 s
%timeit parallel_rng.standard_normal(size=10**8, dtype=np.float32, nthread=128)  # 36.6 ms
```

Note that the `parallel_rng` is slower than Numpy when using a single thread, because the parallel implementation requires a slower algorithm in some cases (this is especially noticeable for float64 and normals)

## Status
The code works and is [reasonably well tested](./test_parallel_numpy_rng.py), so it's probably ready for use. I haven't decided on a distribution method yet; maybe it will just live here, or maybe it's worth spinning out into its own PyPI repo.

Only a two types of random numbers are supported right now: uniform and normal. More could be added if there is demand, although some kinds, like bounded random ints, are hard to parallelize in the approach used here.
