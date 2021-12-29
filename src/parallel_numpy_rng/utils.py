'''
Utilities to help generate random values from bits
'''

import numpy as np
from numba import njit

# TODO: for very low latency cases, there may be benefit to inlining these

def generate_int_to_float(bitgen):
    '''Return a nested dict of numba functions, keyed by 'zero'/'nonzero' then dtype'''
    
    # initialize the numba functions to convert ints to floats
    _next_uint32_pcg64 = bitgen().ctypes.next_uint32
    _next_uint64_pcg64 = bitgen().ctypes.next_uint64

    @njit(fastmath=True)
    def _next_float_pcg64(state):
        '''Random float in the semi-open interval [0,1)'''
        return np.float32(np.int32(_next_uint32_pcg64(state) >> np.uint32(8)) * (np.float32(1.) / np.float32(16777216.)))

    @njit(fastmath=True)
    def _next_float_pcg64_nonzero(state):
        '''Random float in the semi-open interval (0,1]'''
        return np.float32((np.int32(_next_uint32_pcg64(state) >> np.uint32(8)) + np.int32(1))  * (np.float32(1.) / np.float32(16777216.)))

    @njit(fastmath=True)
    def _next_double_pcg64(state):
        '''Random double in the semi-open interval [0,1)'''
        return np.float64(np.int64(_next_uint64_pcg64(state) >> np.uint64(11)) * (np.float64(1.) / np.float64(9007199254740992.)))

    @njit(fastmath=True)
    def _next_double_pcg64_nonzero(state):
        '''Random double in the semi-open interval (0,1]'''
        return np.float64((np.int64(_next_uint64_pcg64(state) >> np.uint64(11)) + np.int64(1))  * (np.float64(1.) / np.float64(9007199254740992.)))
    
    _next_float = {'zero': {np.float32:_next_float_pcg64, np.float64:_next_double_pcg64},
                   'nonzero': {np.float32:_next_float_pcg64_nonzero, np.float64:_next_double_pcg64_nonzero},
                  }
    return _next_float
