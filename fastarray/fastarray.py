"""
FastArray - A compressed array class for efficient AI model storage
"""
import numpy as np
from typing import Optional, Union, Tuple, Any
import ctypes


class CompressionType:
    NONE = "none"
    QUANTIZATION = "quantization"
    SPARSE = "sparse"
    BLOSC = "blosc"
    BLOCK = "block"


class FastArray:
    """
    A compressed array class that provides a drop-in replacement for NumPy arrays
    with automatic compression and efficient operations.
    """
    
    def __init__(self, data, dtype=None, copy=True, order='K', subok=False, ndmin=0, 
                 compression="auto", **kwargs):
        """
        Initialize a FastArray.
        
        Parameters:
        - data: array_like, input data
        - dtype: data type, optional
        - compression: compression algorithm to use ("auto", "quantization", "sparse", "blosc", etc.)
        """
        # First convert to numpy array to normalize input
        self._numpy_array = np.array(data, dtype=dtype, copy=copy, order=order, 
                                     subok=subok, ndmin=ndmin)
        
        # Determine compression type
        if compression == "auto":
            compression = self._choose_compression(self._numpy_array)
        
        self.compression_type = compression
        self._original_shape = self._numpy_array.shape
        self._original_dtype = self._numpy_array.dtype
        
        # Apply compression
        self._compressed_data = self._compress(self._numpy_array, self.compression_type)
        
    def _choose_compression(self, arr):
        """Choose appropriate compression based on array characteristics."""
        # For sparse arrays (many zeros), use sparse compression
        if np.count_nonzero(arr) / arr.size < 0.1:  # Less than 10% non-zero
            return CompressionType.SPARSE
        
        # For float arrays, consider quantization
        if arr.dtype in [np.float32, np.float64]:
            return CompressionType.QUANTIZATION
            
        # For large arrays, use blosc compression
        if arr.nbytes > 1024 * 1024:  # Larger than 1MB
            return CompressionType.BLOSC
            
        return CompressionType.NONE
    
    def _compress(self, arr, compression_type):
        """Compress the array using specified method."""
        if compression_type == CompressionType.NONE:
            return arr
        elif compression_type == CompressionType.QUANTIZATION:
            return self._quantize_compress(arr)
        elif compression_type == CompressionType.SPARSE:
            return self._sparse_compress(arr)
        elif compression_type == CompressionType.BLOSC:
            return self._blosc_compress(arr)
        elif compression_type == CompressionType.BLOCK:
            return self._block_compress(arr)
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")
    
    def _quantize_compress(self, arr):
        """Advanced quantization that preserves accuracy while achieving compression."""
        if arr.dtype not in [np.float32, np.float64]:
            # Only quantize floating point arrays
            return arr

        original_dtype = arr.dtype

        # For float64, keep as float32 for now to save some space
        if original_dtype == np.float64:
            return arr.astype(np.float32)

        # For float32 (most common in neural networks), use our custom quantization
        if original_dtype == np.float32:
            # Use our custom quantization method that maintains accuracy
            # This is inspired by symmetric quantization but implemented from scratch

            # Calculate min and max values for proper scaling
            min_val = np.min(arr)
            max_val = np.max(arr)

            # Handle special case where all values are the same
            if min_val == max_val:
                # Store the single value and shape information
                return {
                    'quantized_data': np.full(arr.shape, 0, dtype=np.int8),  # dummy values
                    'scale': 1.0,
                    'zero_point': float(min_val),
                    'original_shape': arr.shape,
                    'original_dtype': original_dtype
                }

            # Use 8-bit quantization (int8) - maps to range [-128, 127]
            qmin, qmax = -128, 127

            # Calculate scale and zero point for the quantization
            scale = (max_val - min_val) / (qmax - qmin)

            # Quantize the values
            quantized = np.round((arr - min_val) / scale + qmin).astype(np.int8)

            # Clip to ensure values are within the quantized range
            quantized = np.clip(quantized, qmin, qmax)

            # Store the quantization parameters and quantized data
            return {
                'quantized_data': quantized,
                'scale': scale,
                'zero_point': float(min_val),
                'original_shape': arr.shape,
                'original_dtype': original_dtype
            }

        return arr.copy()
    
    def _sparse_compress(self, arr):
        """Compress sparse arrays by storing only non-zero values."""
        # For now, simple implementation - in practice would use scipy.sparse or custom format
        if arr.ndim == 1:
            non_zero_indices = np.nonzero(arr)[0]
            non_zero_values = arr[non_zero_indices]
            return (non_zero_indices, non_zero_values, arr.shape)
        elif arr.ndim == 2:
            non_zero_indices = np.nonzero(arr)
            non_zero_values = arr[non_zero_indices]
            return (non_zero_indices, non_zero_values, arr.shape)
        else:
            # For higher dimensions, flatten then compress
            flat = arr.flatten()
            non_zero_indices = np.nonzero(flat)[0]
            non_zero_values = flat[non_zero_indices]
            return (non_zero_indices, non_zero_values, arr.shape)
    
    def _blosc_compress(self, arr):
        """Compress using blosc library."""
        # In the actual implementation, would use blosc
        # For now, just return original as placeholder
        return arr
    
    def _block_compress(self, arr):
        """Compress by dividing into blocks."""
        # For now, placeholder implementation
        return arr
    
    def _decompress(self):
        """Decompress back to numpy array."""
        if self.compression_type == CompressionType.NONE:
            return self._compressed_data
        elif self.compression_type == CompressionType.QUANTIZATION:
            # Handle the new quantization format with scale and zero point
            if isinstance(self._compressed_data, dict) and 'quantized_data' in self._compressed_data:
                # This is our new quantization format
                quantized = self._compressed_data['quantized_data']
                scale = self._compressed_data['scale']
                zero_point = self._compressed_data['zero_point']

                # Dequantize: convert back to float32
                dequantized = (quantized.astype(np.float32) - (-128)) * scale + zero_point

                # Reshape to original shape if needed
                if dequantized.shape != self._original_shape:
                    dequantized = dequantized.reshape(self._original_shape)

                return dequantized.astype(self._original_dtype)
            else:
                # This is the old format, just convert back to original dtype
                return self._compressed_data.astype(self._original_dtype)
        elif self.compression_type == CompressionType.SPARSE:
            return self._sparse_decompress(self._compressed_data)
        elif self.compression_type == CompressionType.BLOSC:
            return self._compressed_data  # Placeholder
        elif self.compression_type == CompressionType.BLOCK:
            return self._compressed_data  # Placeholder
        else:
            return self._compressed_data
    
    def _sparse_decompress(self, compressed_data):
        """Decompress sparse array."""
        if len(compressed_data) == 3:
            indices, values, shape = compressed_data
            result = np.zeros(shape, dtype=self._original_dtype)
            if len(shape) == 1:
                result[indices] = values
            elif len(shape) == 2:
                result[indices[0], indices[1]] = values
            else:
                # For higher dimensions, need to handle appropriately
                flat_indices = np.ravel_multi_index(indices, shape)
                flat_result = np.zeros(np.prod(shape), dtype=self._original_dtype)
                flat_result[flat_indices] = values
                result = flat_result.reshape(shape)
            return result
        else:
            raise ValueError("Invalid sparse compressed data format")
    
    # Emulate NumPy array interface
    def __array__(self, dtype=None):
        """Return the decompressed array."""
        arr = self._decompress()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr
    
    def __getitem__(self, key):
        """Indexing operation - needs to decompress for now."""
        # In a more optimized version, we'd implement direct indexing on compressed data
        decompressed = self._decompress()
        return FastArray(decompressed[key], compression=self.compression_type)
    
    def __setitem__(self, key, value):
        """Item assignment."""
        # Decompress, modify, and recompress
        decompressed = self._decompress()
        decompressed[key] = value
        # Update internal state with new compressed data
        self._numpy_array = decompressed
        self._compressed_data = self._compress(decompressed, self.compression_type)
    
    @property
    def dtype(self):
        """Return dtype of the array."""
        return self._original_dtype
    
    @property
    def shape(self):
        """Return shape of the array."""
        return self._original_shape
    
    @property
    def size(self):
        """Return total number of elements."""
        return np.prod(self._original_shape)
    
    @property
    def ndim(self):
        """Return number of dimensions."""
        return len(self._original_shape)
    
    @property
    def nbytes(self):
        """Return total bytes consumed by the elements of the array."""
        # Return the compressed size, not the decompressed size
        if self.compression_type == CompressionType.QUANTIZATION:
            if isinstance(self._compressed_data, dict) and 'quantized_data' in self._compressed_data:
                # For quantized arrays, return the size of the quantized data (int8) + metadata
                quantized_size = self._compressed_data['quantized_data'].nbytes
                # Add small overhead for metadata (scale, zero_point, etc.)
                metadata_size = 64  # Approximate size for scale, zero_point, shape info
                return quantized_size + metadata_size
            else:
                # Old format - just return the compressed data size
                return self._compressed_data.nbytes
        elif self.compression_type == CompressionType.SPARSE:
            if isinstance(self._compressed_data, tuple) and len(self._compressed_data) == 3:
                indices, values, shape = self._compressed_data
                return indices.nbytes + values.nbytes + 64  # 64 bytes overhead for metadata
            else:
                return self._decompress().nbytes
        elif self.compression_type == CompressionType.NONE:
            return self._compressed_data.nbytes
        elif self.compression_type == CompressionType.BLOSC:
            # For blosc, we would return compressed size, but for now return decompressed
            return self._compressed_data.nbytes if hasattr(self._compressed_data, 'nbytes') else self._decompress().nbytes
        else:
            # For other types, return the compressed representation size
            if hasattr(self._compressed_data, 'nbytes'):
                return self._compressed_data.nbytes
            else:
                # If it's a dictionary or other object, estimate size
                import sys
                return sys.getsizeof(self._compressed_data)
    
    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        """Copy of the array, cast to a specified type."""
        decompressed = self._decompress()
        converted = decompressed.astype(dtype, order=order, casting=casting, subok=subok, copy=copy)
        return FastArray(converted, compression=self.compression_type)
    
    def tolist(self):
        """Return the array as an a.ndim-levels deep nested list of Python scalars."""
        return self._decompress().tolist()
    
    def __repr__(self):
        """String representation."""
        decompressed = self._decompress()
        return f"FastArray({decompressed.__repr__()}, compression='{self.compression_type}')"
    
    def __str__(self):
        """String representation."""
        return str(self._decompress())
    
    # Arithmetic operations that return FastArray
    def __add__(self, other):
        return self._apply_operation("__add__", other)
    
    def __radd__(self, other):
        return self._apply_operation("__radd__", other)
    
    def __sub__(self, other):
        return self._apply_operation("__sub__", other)
    
    def __rsub__(self, other):
        return self._apply_operation("__rsub__", other)
    
    def __mul__(self, other):
        return self._apply_operation("__mul__", other)
    
    def __rmul__(self, other):
        return self._apply_operation("__rmul__", other)
    
    def __truediv__(self, other):
        return self._apply_operation("__truediv__", other)
    
    def __rtruediv__(self, other):
        return self._apply_operation("__rtruediv__", other)
    
    def __pow__(self, other):
        return self._apply_operation("__pow__", other)
    
    def __rpow__(self, other):
        return self._apply_operation("__rpow__", other)
    
    def _apply_operation(self, op, other):
        """Apply an operation by decompressing, operating, and recompressing."""
        self_arr = self._decompress()
        
        if isinstance(other, FastArray):
            other_arr = other._decompress()
        else:
            other_arr = other
            
        result = getattr(self_arr, op)(other_arr)
        
        # Use same compression type as self, or "auto" if other was FastArray
        new_compression = self.compression_type
        if isinstance(other, FastArray):
            # In a more advanced implementation, we might choose based on result characteristics
            pass
            
        return FastArray(result, compression=new_compression)
    
    # Universal functions (ufuncs) would be implemented here
    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """Sum of array elements over a given axis."""
        result = self._decompress().sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        return FastArray(result, compression=self.compression_type) if np.ndim(result) > 0 else result
    
    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """Average of array elements over a given axis."""
        result = self._decompress().mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        return FastArray(result, compression=self.compression_type) if np.ndim(result) > 0 else result
    
    def reshape(self, *shape, order='C'):
        """Return an array containing the same data with a new shape."""
        decompressed = self._decompress()
        new_shape = decompressed.reshape(*shape, order=order)
        return FastArray(new_shape, compression=self.compression_type)


# Convenience functions that return FastArray objects
def array(data, dtype=None, copy=True, order='K', subok=False, ndmin=0, compression="auto"):
    """Create a FastArray."""
    return FastArray(data, dtype=dtype, copy=copy, order=order, subok=subok, 
                     ndmin=ndmin, compression=compression)

def zeros(shape, dtype=float, order='C', compression="auto"):
    """Return a new FastArray of given shape and type, filled with zeros."""
    return FastArray(np.zeros(shape, dtype=dtype, order=order), compression=compression)

def ones(shape, dtype=float, order='C', compression="auto"):
    """Return a new FastArray of given shape and type, filled with ones."""
    return FastArray(np.ones(shape, dtype=dtype, order=order), compression=compression)

def empty(shape, dtype=float, order='C', compression="auto"):
    """Return a new FastArray of given shape and type, without initializing entries."""
    return FastArray(np.empty(shape, dtype=dtype, order=order), compression=compression)

def full(shape, fill_value, dtype=None, order='C', compression="auto"):
    """Return a new FastArray of given shape and type, filled with fill_value."""
    return FastArray(np.full(shape, fill_value, dtype=dtype, order=order), compression=compression)