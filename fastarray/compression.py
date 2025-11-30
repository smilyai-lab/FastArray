"""
Compression utilities for FastArray
"""
import numpy as np


def compress_array(arr, method="auto"):
    """
    Compress an array using specified method.
    
    Parameters:
    - arr: numpy array or FastArray to compress
    - method: compression method ("auto", "quantization", "sparse", "blosc")
    
    Returns:
    - Compressed representation of the array
    """
    if method == "auto":
        # Use the same logic as in FastArray class
        if np.count_nonzero(arr) / arr.size < 0.1:  # Sparse
            method = "sparse"
        elif arr.dtype in [np.float32, np.float64]:
            method = "quantization"
        elif arr.nbytes > 1024 * 1024:  # Larger than 1MB
            method = "blosc"
        else:
            method = "none"
    
    if method == "quantization":
        if arr.dtype == np.float64:
            return arr.astype(np.float32)
        elif arr.dtype == np.float32:
            return arr.astype(np.float16)
        else:
            return arr
    
    elif method == "sparse":
        # Store non-zero elements only
        mask = arr != 0
        values = arr[mask]
        indices = np.where(mask)
        return {
            'values': values,
            'indices': indices,
            'shape': arr.shape,
            'dtype': arr.dtype
        }
    
    elif method == "blosc":
        # Placeholder - in real implementation, use blosc
        try:
            import blosc
            arr_bytes = arr.tobytes()
            compressed = blosc.compress(arr_bytes, typesize=arr.dtype.itemsize)
            return {
                'compressed_data': compressed,
                'shape': arr.shape,
                'dtype': arr.dtype
            }
        except ImportError:
            # If blosc not available, fall back to quantization
            return compress_array(arr, method="quantization")
    
    else:  # method == "none"
        return arr


def decompress_array(compressed_data, method="auto"):
    """
    Decompress an array.
    
    Parameters:
    - compressed_data: compressed representation
    - method: compression method used
    
    Returns:
    - Decompressed numpy array
    """
    if method == "quantization":
        # For quantization, we might need to restore precision depending on implementation
        return compressed_data
    
    elif method == "sparse":
        if isinstance(compressed_data, dict) and 'values' in compressed_data:
            shape = compressed_data['shape']
            dtype = compressed_data['dtype']
            values = compressed_data['values']
            indices = compressed_data['indices']
            
            result = np.zeros(shape, dtype=dtype)
            result[indices] = values
            return result
        else:
            return compressed_data  # Not actually compressed
    
    elif method == "blosc":
        if isinstance(compressed_data, dict) and 'compressed_data' in compressed_data:
            try:
                import blosc
                arr_bytes = blosc.decompress(compressed_data['compressed_data'])
                shape = compressed_data['shape']
                dtype = compressed_data['dtype']
                
                # Convert bytes back to numpy array
                result = np.frombuffer(arr_bytes, dtype=dtype)
                return result.reshape(shape)
            except ImportError:
                # If blosc not available, return as is
                return compressed_data
        else:
            return compressed_data
    
    else:  # method == "none" or not compressed
        return compressed_data