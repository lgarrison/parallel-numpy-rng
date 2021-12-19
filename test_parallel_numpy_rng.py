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
        if N < nthread-1:
            # don't repeatedly test N < nthread
            continue
            
        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        func = getattr(mtg,funcname)
        s = func(size=N, nthread=1, dtype=dtype)

        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        func = getattr(mtg,funcname)
        p = func(size=N, nthread=nthread, dtype=dtype)

        # In theory, different numbers of threads will yield bit-wise identical answers
        # But in practice, the last digit changes sometimes. This is probably because
        # different code paths are taken based on alignment.
        # We will use atol because our values are all O(unity)
        if dtype == np.float32:
            assert np.allclose(s, p, atol=1e-7, rtol=0.)
        elif dtype == np.float64:
            assert np.allclose(s, p, atol=1e-15, rtol=0.)
    
    
def test_resume(someN, seed, nthread, dtype, funcname):
    '''Test that generating an array of randoms with one call
    or several give the same answer
    '''
    
    from parallel_numpy_rng import MTGenerator
    
    rng = np.random.default_rng(seed)
    
    for N in someN:
        if N < nthread-1:
            # don't repeatedly test N < nthread
            continue
            
        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        func = getattr(mtg,funcname)
        a = func(size=N, nthread=nthread, dtype=dtype)
        
        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        func = getattr(mtg,funcname)
        
        res = np.empty(N, dtype=dtype)
        i = 0
        while i < N:
            n = rng.integers(low=1,high=N-i+1)
            res[i:i+n] = func(size=n, nthread=nthread, dtype=dtype)
            i += n

        if dtype == np.float32:
            assert np.allclose(a, res, atol=1e-7, rtol=0.)
        elif dtype == np.float64:
            assert np.allclose(a, res, atol=1e-15, rtol=0.)
            

def test_mixing_threads(someN, seed, nthread, dtype):
    '''Test that changing the number of threads mid-stream
    doesn't matter.  Only standard normal holds any interesting
    external state.
    '''
    funcname = 'standard_normal'
    from parallel_numpy_rng import MTGenerator
    
    rng = np.random.default_rng(seed)
    maxthreads = nthread
    del nthread
    
    for N in someN:
        if N < maxthreads-1:
            # don't repeatedly test N < nthread
            continue
            
        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        func = getattr(mtg,funcname)
        a = func(size=N, nthread=maxthreads, dtype=dtype)
        
        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        func = getattr(mtg,funcname)
        
        res = np.empty(N, dtype=dtype)
        i = 0
        tstart = np.linspace(0, N, maxthreads+1, endpoint=True, dtype=int)
        # sweep from 1 to maxthreads
        for t in range(maxthreads):
            n = tstart[t+1]-tstart[t]
            res[i:i+n] = func(size=n, nthread=t+1, dtype=dtype)
            i += n

        if dtype == np.float32:
            assert np.allclose(a, res, atol=1e-7, rtol=0.)
        elif dtype == np.float64:
            assert np.allclose(a, res, atol=1e-15, rtol=0.)
            
            
def test_mixing_func(someN, seed, nthread, dtype):
    '''Test interleaving random and standard_normal works for different N/thread
    '''
    
    from parallel_numpy_rng import MTGenerator
    
    rng = np.random.default_rng(seed)
    
    for N in someN:
        if N < nthread-1:
            # don't repeatedly test N < nthread
            continue
            
        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        coin_seed = rng.integers(2**16)
        coin_rng = np.random.default_rng(coin_seed)
            
        nchunk = max(2,N//100)
        serial = np.empty(N, dtype=dtype)
        i = 0
        tstart = np.linspace(0, N, nchunk+1, endpoint=True, dtype=int)
        for t in range(nchunk):
            n = tstart[t+1]-tstart[t]
            # in each chunk, flip a coin to decide the function
            func = mtg.random if coin_rng.integers(2) else mtg.standard_normal
            serial[i:i+n] = func(size=n, nthread=1, dtype=dtype)
            i += n
            
            
        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        coin_rng = np.random.default_rng(coin_seed)
        
        parallel = np.empty(N, dtype=dtype)
        i = 0
        for t in range(nchunk):
            n = tstart[t+1]-tstart[t]
            func = mtg.random if coin_rng.integers(2) else mtg.standard_normal
            parallel[i:i+n] = func(size=n, nthread=nthread, dtype=dtype)
            i += n
        
        if dtype == np.float32:
            assert np.allclose(serial, parallel, atol=1e-7, rtol=0.)
        elif dtype == np.float64:
            assert np.allclose(serial, parallel, atol=1e-15, rtol=0.)

    
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
        if N < maxthreads-1:
            # don't repeatedly test N < nthread
            continue
            
        pcg = np.random.PCG64(seed)
        mtg = MTGenerator(pcg)
        a = mtg.random(size=N, nthread=nthread, dtype=dtype)
        
        rng = np.random.Generator(np.random.PCG64(seed))
        b = rng.random(size=N, dtype=dtype)
        
        if dtype == np.float64:
            assert np.allclose(a, b, atol=1e-15, rtol=0.)
        elif dtype == np.float32:
            assert np.allclose(a, b, atol=1e-7)
        else:
            raise ValueError(dtype)


def test_finite_normals_float32():
    '''
    If the floats fed to Box-Muller can include 0, it will produce infinity.
    We use the interval (0,1] to avoid this.
    
    In theory, we ought to test float64 the same way. But it's hard to find
    a PCG state that produces 53 zeros...
    
    https://github.com/lgarrison/parallel-numpy-rng/issues/1
    '''
    from parallel_numpy_rng import MTGenerator
    pcg = np.random.PCG64(1194)
    mtg = MTGenerator(pcg)
    a = mtg.standard_normal(size=20000, nthread=maxthreads, dtype=np.float32)
    assert np.all(np.isfinite(a))
