# FastArray

FastArray is a compressed array library designed for AI models that serves as a drop-in replacement for NumPy. It provides automatic compression and efficient operations for large arrays used in machine learning and AI applications.

## Features

- **Drop-in NumPy replacement**: Use FastArray anywhere you would use NumPy arrays
- **Automatic compression**: Automatically selects the best compression method based on array characteristics
- **Multiple compression strategies**: Quantization, sparse storage, Blosc compression, and more
- **GPU/TPU support**: Optimized for accelerated computing hardware
- **Memory efficient**: Significantly reduces memory usage for large AI models
- **Fast operations**: Optimized operations even on compressed data

## Installation

```bash
pip install fastarray
```

## Quick Start

```python
import fastarray as fa

# Create a compressed array - automatically compressed based on content
arr = fa.array([1, 2, 3, 4, 5])
print(arr)  # FastArray([1 2 3 4 5], compression='quantization')

# Operations work just like NumPy
result = arr * 2
print(result)  # FastArray([2 4 6 8 10], compression='quantization')

# Large arrays get automatically compressed
large_arr = fa.zeros((1000, 1000))  # Uses compression for large arrays

# Linear algebra operations work seamlessly
a = fa.array([[1, 2], [3, 4]])
b = fa.array([[5, 6], [7, 8]])
result = fa.dot(a, b)  # [[19 22], [43 50]]
```

## Compression Methods

FastArray automatically selects the best compression method:

- **Quantization**: Reduces precision (e.g., float32 to float16) for numerical arrays
- **Sparse**: Stores only non-zero values for sparse arrays (arrays with many zeros)
- **Blosc**: General-purpose compression for large arrays

## AI Model Example

FastArray is designed for AI workloads like large language models:

```python
import fastarray as fa

# Working with attention matrices (often sparse)
attention_scores = fa.random.randn(4096, 4096)  # Large attention matrix
attention_weights = fa.softmax(attention_scores)  # Still efficiently compressed

# Model weights can be automatically compressed
model_weights = fa.random.randn(10000, 512)  # Large weight matrix
output = fa.dot(model_weights, input_data)  # Efficient computation

# Save/load model weights with compression
fa.index.save_array_to_disk(model_weights, "model_weights",
                           metadata={"layer": "attention", "size": "7B"})
loaded_weights = fa.index.load_array_from_disk("model_weights")
```

## API Compatibility

FastArray maintains full NumPy API compatibility. All NumPy functions and methods that work on `np.ndarray` will work the same way with `fa.FastArray`.

## Performance

FastArray is specifically designed for:
- Large neural network weight matrices
- Attention matrices in transformers
- Training and inference on modest hardware
- TPU/GPU accelerated computing

## Documentation

For complete documentation, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

FastArray is released under the MIT License. See the [LICENSE](LICENSE) file for more details.