# parallel-numpy-rng
![tests](https://github.com/lgarrison/parallel-numpy-rng/actions/workflows/test.yml/badge.svg)

A multi-threaded random number generator, backed by Numpy RNG.

## Overview
Uses the "fast-forward" capability of the PCG-family of RNG, as exposed by the
new-style Numpy RNG API, to generate random numbers in a multi-threaded manner. The key
is that you get the same random numbers regardless of how many threads were used.

## Example
```python
import numpy as np
from parallel_numpy_rng import MTGenerator
p = np.random.PCG64(123)  # or PCG64DXSM
mtg = MTGenerator(p)
r1 = mtg.random(size=16, nthread=2, dtype=np.float32)
r2 = mtg.standard_normal(size=16, nthread=2, dtype=np.float32)
```
