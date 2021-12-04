import os

import numpy as np
import pytest

# TODO: add PCG64DXSM test

# TODO: repeatedly testing N < Nthread isn't exercising anything new
maxthreads = len(os.sched_getaffinity(0))
all_Nthreads = [1,2,3,4,maxthreads] + \
                list(range(5,maxthreads,12)) + \
                list(range(5,maxthreads,11))
all_Nthreads = sorted(list(set(filter(lambda n: n <= maxthreads, all_Nthreads))))

@pytest.fixture(scope='module')
def allN(request):
    '''~1000 values up to 10^5'''
    _rng = np.random.default_rng(123)
    Ntest = [1,2,3,10,100,1000]
    Ntest += list((10**(_rng.random(size=1000)*5)).astype(int))
    Ntest = sorted(list(set(Ntest)))
    return Ntest

@pytest.fixture(scope='module')
def someN(request):
    '''~100 values up to 10^4'''
    _rng = np.random.default_rng(123)
    Ntest = [1,2,3,10,100,1000]
    Ntest += list((10**(_rng.random(size=100)*4)).astype(int))
    Ntest = sorted(list(set(Ntest)))
    return Ntest


@pytest.fixture(scope='module', params=[123,0xDEADBEEF], ids=['seed1','seed2'])
def seed(request):
    return request.param


@pytest.fixture(scope='module', params=all_Nthreads)
def nthread(request):
    return request.param


@pytest.fixture(scope='module', params=[np.float32,np.float64])
def dtype(request):
    return request.param


@pytest.fixture(scope='module', params=['random','standard_normal'])
def funcname(request):
    return request.param


def test_threads(allN, seed, nthread, dtype, funcname):
    '''do different nthreads give the same answer?
    '''
    
    from parallel_numpy_rng import MTGenerator
    
    for N in allN:
        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        func = getattr(mtg,funcname)
        s = func(size=N, nthread=1, dtype=dtype)

        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        func = getattr(mtg,funcname)
        p = func(size=N, nthread=nthread, dtype=dtype)

        assert np.all(s == p)
    
    
def test_resume(someN, seed, nthread, dtype, funcname):
    '''Test that generating an array of randoms with one call
    or several give the same answer
    '''
    
    from parallel_numpy_rng import MTGenerator
    
    rng = np.random.default_rng(seed)
    
    for N in someN:
        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        func = getattr(mtg,funcname)
        a = func(size=N, nthread=nthread, dtype=dtype)
        
        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        func = getattr(mtg,funcname)
        
        res = np.empty(N, dtype=dtype)
        i = 0
        print(f'N: {N}')
        while i < N:
            # TODO: going to be slow for large N!
            n = rng.integers(low=1,high=N-i+1)
            print(f'i,n: {i},{n}')
            res[i:i+n] = func(size=n, nthread=nthread, dtype=dtype)
            i += n

        assert np.all(a == res)

    
def test_uniform_matches_numpy(someN, seed, nthread, dtype):
    '''Both Numpy and MTGenerator call the PCG uniform double generator, so
    they actually produce identical streams. Floats are almost the same,
    except we have to reimplement the uint32->float part. So it will be
    close but not exact.
    
    This isn't a property we guarantee or really need to preserve, but
    it's a sign that everything is working as expected.
    '''
    from parallel_numpy_rng import MTGenerator
    
    for N in someN:
        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        a = mtg.random(size=N, nthread=nthread, dtype=dtype)
        
        rng = np.random.Generator(np.random.PCG64(seed))
        b = rng.random(size=N, dtype=dtype)
        
        if dtype == np.float64:
            assert np.all(a == b)
        elif dtype == np.float32:
            assert np.allclose(a, b,atol=1e-7)
        else:
            raise ValueError(dtype)
