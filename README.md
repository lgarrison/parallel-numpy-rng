# parallel-numpy-rng
[![tests](https://github.com/lgarrison/parallel-numpy-rng/actions/workflows/test.yml/badge.svg)](https://github.com/lgarrison/parallel-numpy-rng/actions/workflows/test.yml) [![PyPI](https://img.shields.io/pypi/v/parallel-numpy-rng)](https://pypi.org/project/parallel-numpy-rng/)

A multi-threaded random number generator backed by NumPy RNG, with parallelism provided by Numba.

## Overview
This package uses the "fast-forward" capability of the [PCG family of RNG](https://www.pcg-random.org),
as exposed by the [new-style NumPy RNG API](https://numpy.org/doc/stable/reference/random/index.html),
to generate arrays of random numbers in a multi-threaded manner. The result depends only on the random
number seed, not the number of threads.

Only a two types of random numbers are supported right now: uniform and normal. More
could be added if there is demand, although some kinds, like bounded random ints, are
hard to parallelize in the approach used here.

The uniform randoms are the same as NumPy produces for a given seed, although the
random normals are not.

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

Note that the `parallel_rng` is slower than NumPy when using a single thread, because the parallel implementation requires a slower algorithm in some cases (this is especially noticeable for float64 and normals)

## Installation
The code works and is [reasonably well tested](./test_parallel_numpy_rng.py), so it's probably ready for use.  It can be installed from PyPI:
```console
$ pip install parallel-numpy-rng
```

## Details
Random number generation can be slow, even with modern algorithms like PCG, so it's helpful to
be able to use multiple threads. The easy way to do this is to give each thread a different
seed, but then the RNG sequence will depend on how many threads you used and how you did the
seed offset. It would be nice if the RNG sequence could be the output of a single logical 
sequence (i.e. the stream resulting from a single seed), and the number of threads could
just be an implementation detail.

The key capability to enable this is cheap fast-forwarding of the underlying RNG.  For example,
if we want to generate *N* random numbers with 2 threads, we know the first thread will do *N/2* calls
to the RNG, thus advancing its internal state that many times. Therefore, we would like to start
the second thread's RNG in a state that is already advanced *N/2* times, but without actually making
*N/2* calls to get there.

This is known as fast-forwarding, or jump-ahead. [NumPy's new RNG API](https://numpy.org/doc/stable/reference/random/index.html)
(as of NumPy 1.17) uses the PCG RNG that supports this feature, and NumPy exposes an [`advance()`
method](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.PCG64.advance.html#numpy.random.PCG64.advance)
which implements it.  The new API also exposes CFFI bindings to get PCG random values,
so we can implement the core loop, including parallelism, in a Numba-compiled function
that can call the RNG via a low-level function pointer.

An interesting consequence of using fast-forwarding is that each random value must be generated
with a known number of calls to the underlying RNG, so that we know how many steps to advance
the RNG state by. This means we can't use rejection sampling, which makes a variable number of
calls.  Fortunately, inverse-transform sampling can usually substitute, or more specific methods
like Box-Muller. These can be slower than rejection sampling (or whatever NumPy uses) with one
thread, but even just two threads more than makes up for the difference.
