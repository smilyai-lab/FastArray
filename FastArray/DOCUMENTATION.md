# FastArray Documentation

## Overview

FastArray is a compressed array library designed for AI models that serves as a drop-in replacement for NumPy. It provides automatic compression and efficient operations for large arrays used in machine learning and AI applications.

## Installation

```bash
pip install fastarray
```

## Basic Usage

```python
import fastarray as fa

# Create a compressed array - automatically compressed based on content
arr = fa.array([1, 2, 3, 4, 5])
print(arr)  # FastArray([1 2 3 4 5], compression='quantization')

# Operations work just like NumPy
result = arr * 2
print(result)  # FastArray([2 4 6 8 10], compression='quantization')

# Large arrays get automatically compressed
large_arr = fa.zeros((1000, 1000))  # Uses Blosc compression for large arrays
```

## Core Features

### 1. Automatic Compression

FastArray automatically selects the most appropriate compression method based on the characteristics of your data:

- **Quantization**: For numerical arrays, reduces precision (e.g., float32 to float16)
- **Sparse Storage**: For arrays with many zeros, stores only non-zero values with indices
- **Blosc Compression**: For large arrays, uses the Blosc compression library
- **No Compression**: For small arrays where compression overhead isn't worth it

### 2. Drop-in NumPy Compatibility

All NumPy functions and methods work with FastArray:

```python
import fastarray as fa
import numpy as np

# All NumPy operations work seamlessly
a = fa.array([[1, 2], [3, 4]])
b = fa.array([[5, 6], [7, 8]])

result = fa.dot(a, b)  # Linear algebra operations
mean_val = fa.mean(a)  # Statistical operations
reshaped = a.reshape(1, 4)  # Array manipulation
```

### 3. Backend Acceleration

FastArray supports multiple backends for optimized performance:

- **CPU**: Standard NumPy-like operations
- **GPU**: Accelerated operations via CuPy
- **TPU**: Accelerated operations via JAX or TensorFlow

```python
import fastarray as fa

# Set backend to GPU (if available)
fa.backend.set_backend('gpu')

# Perform operations on GPU
large_arr = fa.ones((1000, 1000))
result = fa.dot(large_arr, large_arr)  # Computed on GPU
```

## API Reference

### FastArray Class

The `FastArray` class is the core of the library and provides the same interface as NumPy's `ndarray`.

#### Constructor

`fa.array(data, dtype=None, copy=True, order='K', subok=False, ndmin=0, compression="auto")`

Creates a new FastArray with automatic compression.

Parameters:
- `data`: Input data (list, tuple, numpy array, etc.)
- `dtype`: Data type of the array
- `copy`: Whether to copy the input data
- `order`: Memory layout ('C', 'F', etc.)
- `subok`: Whether to allow sub-classes
- `ndmin`: Minimum number of dimensions
- `compression`: Compression method ("auto", "quantization", "sparse", "blosc", "none")

#### Properties

- `.shape`: Shape of the array
- `.dtype`: Data type of the array
- `.size`: Number of elements
- `.ndim`: Number of dimensions
- `.nbytes`: Number of bytes used
- `.compression_type`: Type of compression applied

#### Methods

Most NumPy array methods are supported:

- `.astype(dtype)`: Convert to different data type
- `.reshape(*shape)`: Change shape of the array
- `.sum()`, `.mean()`, `.min()`, `.max()`: Reduction operations
- `.tolist()`: Convert to Python list

### Creation Functions

FastArray provides the same array creation functions as NumPy:

- `fa.array()`: Create array from data
- `fa.zeros(shape)`: Create array of zeros
- `fa.ones(shape)`: Create array of ones
- `fa.empty(shape)`: Create uninitialized array
- `fa.full(shape, value)`: Create array filled with value
- `fa.arange(start, stop, step)`: Create range of values
- `fa.linspace(start, stop, num)`: Create evenly spaced values

### Linear Algebra

FastArray includes all common linear algebra operations:

```python
import fastarray as fa

a = fa.array([[1, 2], [3, 4]])
b = fa.array([[5, 6], [7, 8]])

# Matrix operations
dot_product = fa.dot(a, b)
determinant = fa.linalg.det(a)
eigenvals, eigenvecs = fa.linalg.eig(a)
svd_result = fa.linalg.svd(a)
```

Available functions include:
- `fa.dot()`, `fa.vdot()`, `fa.outer()`, `fa.inner()`
- `fa.linalg.det()`, `fa.linalg.inv()`, `fa.linalg.svd()`, `fa.linalg.eig()`
- And many more linear algebra operations

### Random Number Generation

FastArray provides random number generation similar to NumPy:

```python
# Basic random generation
random_arr = fa.random.rand(3, 4)  # 3x4 array of random values [0, 1)
normal_arr = fa.random.randn(100)  # 100 values from normal distribution
integers = fa.random.randint(0, 10, size=5)  # 5 random integers [0, 10)
```

### Memory Management

FastArray provides functions for managing memory usage:

#### Array Offloading

```python
import fastarray as fa

large_arr = fa.ones((10000, 10000))  # Very large array

# Offload to disk to free memory
file_path = fa.memory.offload_array_to_disk(large_arr, "big_matrix")

# Load back when needed
restored_arr = fa.memory.load_array_from_disk_offloaded(file_path)
```

#### Memory Mapping

```python
# Create memory-mapped array (accessed from disk)
mapped_arr = fa.memory.memory_map_array(large_arr, "mapped_file.dat")
```

#### Index System

FastArray includes an index system to manage multiple stored arrays:

```python
# Save array with metadata
fa.index.save_array_to_disk(my_array, "model_weights", 
                           metadata={"layer": "conv1", "epoch": 10})

# List saved arrays
saved_names = fa.index.list_saved_arrays()

# Load by name
loaded_array = fa.index.load_array_from_disk("model_weights")

# Get array information
info = fa.index.get_file_manager().index.get_array_info("model_weights")
```

## Compression Algorithms

### Quantization

Reduces precision to save memory while maintaining accuracy for AI workloads:

```python
# Automatically quantizes float32 to float16
quantized_arr = fa.array([1.1, 2.2, 3.3], compression="quantization")
```

### Sparse Storage

Efficiently stores arrays with many zeros:

```python
# Automatically detects sparse arrays
sparse_data = [0, 0, 0, 4, 0, 0, 7, 0, 0, 10]
sparse_arr = fa.array(sparse_data)  # Will use sparse compression
```

### Blosc Compression

Uses the Blosc library for general-purpose compression:

```python
# For large arrays, Blosc provides excellent compression ratios
large_arr = fa.zeros((1000, 1000), compression="blosc")
```

## Performance Tips

### 1. Choose the Right Compression

For different use cases:
- **AI Models**: Use `'auto'` for automatic selection based on array characteristics
- **Sparse Models**: Use `'sparse'` if your arrays have many zeros
- **Numerical Data**: Use `'quantization'` to reduce precision
- **Large Arrays**: Use `'blosc'` for maximum compression

### 2. Backend Selection

- Use CPU backend for general operations
- Use GPU backend for large matrix operations
- Use TPU backend for specific TPU-optimized operations

### 3. Memory Management

- Offload arrays to disk when not actively used
- Use the index system to organize multiple saved arrays
- Monitor memory usage with `fa.memory.get_memory_usage_info()`

## Examples

### Example 1: Training with Large Model Weights

```python
import fastarray as fa

# Load large model weights automatically compressed
weights = fa.load_array_from_disk("large_model_weights")

# Perform training operations efficiently
for batch in data_loader:
    # Operations work seamlessly with compressed arrays
    output = fa.dot(weights, batch)
    loss = calculate_loss(output, targets)
    
    # Update weights (compression handled automatically)
    weights = update_weights(weights, gradients)
```

### Example 2: Attention Matrices

```python
import fastarray as fa

# Attention matrices can be very large but often sparse
def create_attention_matrix(queries, keys):
    # Compute attention scores
    scores = fa.dot(queries, fa.transpose(keys))
    
    # Apply softmax
    attention = fa.softmax(scores)
    
    # Often attention matrices are sparse (many near-zero values)
    # FastArray automatically handles this efficiently
    return attention

# Large attention matrix handled efficiently
large_attention = create_attention_matrix(
    fa.random.randn(10000, 512),  # Large query matrix
    fa.random.randn(10000, 512)   # Large key matrix
)
```

## Troubleshooting

### Common Issues

1. **Performance**: If operations are slower than expected, check that the appropriate backend is selected.

2. **Memory**: If memory usage is high, consider using the offloading system or different compression methods.

3. **Compatibility**: If a NumPy function doesn't work as expected, ensure you're using the version imported from fastarray.

## Contributing

FastArray is an open-source project. Contributions are welcome! Please see our contributing guide for more information.