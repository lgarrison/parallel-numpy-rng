# parallel-numpy-rng
A multi-threaded random number generator, backed by Numpy RNG.

## Overview
Uses the "fast-forward" capability of the PCG-family of RNG, as exposed by the
new-style Numpy RNG API, to generate random numbers in a multi-threaded manner. The key
is that you get the same random numbers regardless of how many threads were used.
