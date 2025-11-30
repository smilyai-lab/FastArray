"""
FastArray - A compressed array library for AI models
Fast drop-in replacement for NumPy with automatic compression
"""
from .fastarray import FastArray
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
from . import jax_integration
from . import jax_training_integration

__version__ = "0.1.0"
__all__ = ["FastArray", "array", "zeros", "ones", "empty", "full",
           "compress_array", "decompress_array", "backend", "index", "memory", "jax_integration", "jax_training_integration"]