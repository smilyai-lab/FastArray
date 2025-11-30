"""
Backend system for FastArray supporting CPU/GPU/TPU operations
"""
import numpy as np
from typing import Optional, Any
import platform

# Backend types
CPU = "cpu"
GPU = "gpu" 
TPU = "tpu"

# Try to import acceleration libraries
try:
    import cupy as cp  # For GPU operations
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

try:
    import jax.numpy as jnp  # For TPU operations via JAX
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None

try:
    import tensorflow as tf  # Alternative TPU support
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None


class BackendManager:
    """
    Manages operations across different backends (CPU, GPU, TPU)
    """
    
    def __init__(self):
        self.current_backend = CPU
        self.device = None
        
        # Auto-detect available backends
        self.available_backends = [CPU]
        
        if CUDA_AVAILABLE:
            self.available_backends.append(GPU)
            
        if JAX_AVAILABLE or TF_AVAILABLE:
            self.available_backends.append(TPU)
    
    def set_backend(self, backend: str, device: Optional[Any] = None):
        """
        Set the current backend for operations
        """
        if backend not in self.available_backends:
            raise ValueError(f"Backend {backend} not available. Available: {self.available_backends}")
        
        self.current_backend = backend
        self.device = device
    
    def get_backend_array(self, array):
        """
        Convert an array to the appropriate backend format
        """
        if self.current_backend == CPU:
            # Return as numpy array
            if hasattr(array, '_decompress'):
                return array._decompress()
            elif isinstance(array, np.ndarray):
                return array
            else:
                return np.array(array)
        
        elif self.current_backend == GPU and CUDA_AVAILABLE:
            # Convert to cupy array
            if hasattr(array, '_decompress'):
                numpy_arr = array._decompress()
                return cp.asarray(numpy_arr)
            elif isinstance(array, np.ndarray):
                return cp.asarray(array)
            elif hasattr(array, 'get'):  # Already a cupy array
                return array
            else:
                return cp.asarray(array)
        
        elif self.current_backend == TPU:
            # Convert to JAX array if available, otherwise TensorFlow
            if JAX_AVAILABLE:
                if hasattr(array, '_decompress'):
                    numpy_arr = array._decompress()
                    return jnp.array(numpy_arr)
                elif isinstance(array, np.ndarray):
                    return jnp.array(array)
                else:
                    return jnp.array(array)
            elif TF_AVAILABLE:
                if hasattr(array, '_decompress'):
                    numpy_arr = array._decompress()
                    return tf.convert_to_tensor(numpy_arr)
                elif isinstance(array, np.ndarray):
                    return tf.convert_to_tensor(array)
                else:
                    return tf.convert_to_tensor(array)
        
        # Fallback to CPU
        if hasattr(array, '_decompress'):
            return array._decompress()
        else:
            return np.asarray(array)
    
    def perform_operation(self, op_func, *args, **kwargs):
        """
        Perform an operation using the current backend
        """
        if self.current_backend == CPU:
            # Just use numpy operations
            return op_func(*args, **kwargs)
        
        elif self.current_backend == GPU and CUDA_AVAILABLE:
            # Convert inputs to cupy, perform operation, convert back to numpy
            cp_args = []
            for arg in args:
                if isinstance(arg, np.ndarray) or hasattr(arg, '_decompress'):
                    cp_args.append(self.get_backend_array(arg))
                else:
                    cp_args.append(arg)
            
            result = op_func(*cp_args, **kwargs)
            
            # Convert back to numpy for storage in FastArray
            if hasattr(result, 'get'):  # cupy array
                return result.get()
            else:
                return np.asarray(result)
        
        elif self.current_backend == TPU:
            # Use JAX or TensorFlow
            if JAX_AVAILABLE:
                jnp_args = []
                for arg in args:
                    if isinstance(arg, np.ndarray) or hasattr(arg, '_decompress'):
                        jnp_args.append(self.get_backend_array(arg))
                    else:
                        jnp_args.append(arg)
                
                result = op_func(*jnp_args, **kwargs)
                
                # Convert back to numpy
                return np.asarray(result)
            
            elif TF_AVAILABLE:
                tf_args = []
                for arg in args:
                    tf_args.append(self.get_backend_array(arg))
                
                result = op_func(*tf_args, **kwargs)
                
                # Convert back to numpy
                if hasattr(result, 'numpy'):
                    return result.numpy()
                else:
                    return np.asarray(result)
        
        # Fallback to CPU
        return op_func(*args, **kwargs)


# Global backend manager instance
_backend_manager = BackendManager()


def get_backend_manager():
    """Get the global backend manager instance"""
    return _backend_manager


def set_backend(backend: str, device: Optional[Any] = None):
    """Set the global backend"""
    _backend_manager.set_backend(backend, device)


def get_current_backend():
    """Get the current backend"""
    return _backend_manager.current_backend


def get_available_backends():
    """Get available backends"""
    return _backend_manager.available_backends