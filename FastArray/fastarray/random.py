"""
Random number generation for FastArray
"""
import numpy as np
from .fastarray import FastArray


# Random number generation functions
def rand(*args):
    """Random values in a given shape."""
    result = np.random.rand(*args)
    return FastArray(result)


def randn(*args):
    """Return a sample (or samples) from the "standard normal" distribution."""
    result = np.random.randn(*args)
    return FastArray(result)


def randint(low, high=None, size=None, dtype=int):
    """Return random integers from low (inclusive) to high (exclusive)."""
    result = np.random.randint(low, high=high, size=size, dtype=dtype)
    return FastArray(result)


def random(size=None):
    """Return random floats in the half-open interval [0.0, 1.0)."""
    result = np.random.random(size)
    return FastArray(result)


def choice(a, size=None, replace=True, p=None):
    """Generate a random sample from a given 1-D array."""
    result = np.random.choice(a, size=size, replace=replace, p=p)
    if isinstance(result, np.ndarray):
        return FastArray(result)
    else:
        return result


def uniform(low=0.0, high=1.0, size=None):
    """Draw samples from a uniform distribution."""
    result = np.random.uniform(low=low, high=high, size=size)
    return FastArray(result)


def normal(loc=0.0, scale=1.0, size=None):
    """Draw random samples from a normal (Gaussian) distribution."""
    result = np.random.normal(loc=loc, scale=scale, size=size)
    return FastArray(result)


def beta(a, b, size=None):
    """Draw samples from a Beta distribution."""
    result = np.random.beta(a, b, size=size)
    return FastArray(result)


def gamma(shape, scale=1.0, size=None):
    """Draw samples from a Gamma distribution."""
    result = np.random.gamma(shape, scale=scale, size=size)
    return FastArray(result)


# Seeding
def seed(seed=None):
    """Seed the generator."""
    np.random.seed(seed)


# Permutations
def shuffle(x):
    """Modify a sequence in-place by shuffling its contents."""
    if isinstance(x, FastArray):
        # Convert to numpy array, shuffle, then back to FastArray
        arr = x._decompress()
        np.random.shuffle(arr)
        return FastArray(arr, compression=x.compression_type)
    else:
        np.random.shuffle(x)
        return x


def permutation(x):
    """Randomly permute a sequence, or return a permuted range."""
    if isinstance(x, FastArray):
        arr = x._decompress()
        result = np.random.permutation(arr)
        return FastArray(result, compression=x.compression_type)
    else:
        result = np.random.permutation(x)
        if isinstance(result, np.ndarray):
            return FastArray(result)
        else:
            return result