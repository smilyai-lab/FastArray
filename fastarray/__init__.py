"""
FastArray - A compressed array library for AI models
Fast drop-in replacement for NumPy with automatic compression
Enhanced for TPU V5e-8 with BF16 support
"""
from .fastarray import FastArray, CompressionType
from .compression import compress_array, decompress_array
from . import numpy_api  # Import all NumPy-compatible functions

# Export the same API as NumPy for drop-in compatibility
from numpy import *

# Override array creation functions to return FastArray objects
from .fastarray import array, zeros, ones, empty, full
from . import linalg
from . import random
from . import backend
from . import index
from . import memory
from . import jax_fastarray as jax_integration  # Updated to use new JAX-fastarray
from . import jax_training_integration

__version__ = "1.0.0"
__all__ = ["FastArray", "array", "zeros", "ones", "empty", "full", "CompressionType",
           "compress_array", "decompress_array", "backend", "index", "memory", "jax_integration", "jax_training_integration"]