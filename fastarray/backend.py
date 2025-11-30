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
    from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
    from jax.experimental import mesh_utils
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None
    NamedSharding = None
    Mesh = None
    mesh_utils = None

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
        self.jax_mesh = None  # For TPU mesh operations

        # Auto-detect available backends
        self.available_backends = [CPU]

        if CUDA_AVAILABLE:
            self.available_backends.append(GPU)

        if JAX_AVAILABLE:
            # Check if JAX can access accelerators
            try:
                devices = jax.devices()
                if devices:
                    self.available_backends.append(TPU)
            except:
                pass  # JAX might be available but accelerators not accessible

        if TF_AVAILABLE and not JAX_AVAILABLE:
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    self.available_backends.append(GPU)
            except:
                pass

    def set_backend(self, backend: str, device: Optional[Any] = None, mesh_shape: Optional[tuple] = None):
        """
        Set the current backend for operations
        """
        if backend not in self.available_backends:
            raise ValueError(f"Backend {backend} not available. Available: {self.available_backends}")

        self.current_backend = backend
        self.device = device

        # Set up JAX mesh if specified for TPU operations
        if backend == TPU and JAX_AVAILABLE and mesh_shape and mesh_utils:
            try:
                mesh_devices = mesh_utils.create_device_mesh(mesh_shape)
                self.jax_mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
            except Exception as e:
                print(f"Warning: Could not create JAX mesh: {e}")
                self.jax_mesh = None

    def get_backend_array(self, array, sharding_rule=None):
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

        elif self.current_backend == TPU and JAX_AVAILABLE:
            # Convert to JAX array for TPU operations
            if hasattr(array, '_decompress'):
                numpy_arr = array._decompress()
                jax_arr = jnp.array(numpy_arr)
            elif isinstance(array, np.ndarray):
                jax_arr = jnp.array(array)
            else:
                jax_arr = jnp.array(array)

            # If a sharding rule is provided, apply it
            if sharding_rule and self.jax_mesh:
                try:
                    sharding = NamedSharding(self.jax_mesh, sharding_rule)
                    jax_arr = jax.device_put(jax_arr, sharding)
                except:
                    # If sharding fails, return normal array
                    pass

            return jax_arr

        elif self.current_backend == TPU and TF_AVAILABLE and not JAX_AVAILABLE:
            # Use TensorFlow for TPU (fallback)
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

    def perform_operation(self, op_func, *args, sharding_rule=None, **kwargs):
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

        elif self.current_backend == TPU and JAX_AVAILABLE:
            # Use JAX for operations
            jnp_args = []
            for arg in args:
                if isinstance(arg, np.ndarray) or hasattr(arg, '_decompress'):
                    jnp_args.append(self.get_backend_array(arg, sharding_rule))
                else:
                    jnp_args.append(arg)

            result = op_func(*jnp_args, **kwargs)

            # Convert back to numpy (this preserves values while converting type)
            # For large arrays that will be used in further JAX operations, user should work directly with jax arrays
            return np.asarray(result)

        elif self.current_backend == TPU and TF_AVAILABLE and not JAX_AVAILABLE:
            # Use TensorFlow operations
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


def set_backend(backend: str, device: Optional[Any] = None, mesh_shape: Optional[tuple] = None):
    """Set the global backend with optional mesh configuration for TPU"""
    _backend_manager.set_backend(backend, device, mesh_shape)


def get_current_backend():
    """Get the current backend"""
    return _backend_manager.current_backend


def get_available_backends():
    """Get available backends"""
    return _backend_manager.available_backends


def to_jax_array(fastarray, sharding_rule=None):
    """Convert a FastArray directly to a JAX array, useful for TPU operations"""
    if JAX_AVAILABLE:
        backend_manager = get_backend_manager()
        return backend_manager.get_backend_array(fastarray, sharding_rule)
    else:
        raise RuntimeError("JAX is not available. Please install JAX to use JAX arrays.")


def set_tpu_mesh(mesh_shape):
    """Set up a TPU mesh for sharded operations"""
    if JAX_AVAILABLE and mesh_utils:
        try:
            mesh_devices = mesh_utils.create_device_mesh(mesh_shape)
            mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
            return mesh
        except Exception as e:
            print(f"Warning: Could not create TPU mesh: {e}")
            return None
    else:
        print("JAX is not available. TPU mesh creation requires JAX.")
        return None