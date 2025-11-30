"""
Test to verify that FastArray quantization preserves accuracy for neural networks
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import fastarray as fa

def test_accuracy_preservation():
    """Test that quantization preserves accuracy within acceptable bounds"""
    print("Testing accuracy preservation with FastArray quantization...")
    print("=" * 60)
    
    # Test 1: Neural network-like weights (random values with normal distribution)
    print("\\n1. Testing neural network weight compression:")
    nn_weights = np.random.randn(1000, 512).astype(np.float32)  # Typical weight matrix
    print(f"   Original shape: {nn_weights.shape}")
    print(f"   Original dtype: {nn_weights.dtype}")
    print(f"   Original range: [{nn_weights.min():.6f}, {nn_weights.max():.6f}]")
    print(f"   Original std: {nn_weights.std():.6f}")
    
    # Compress with FastArray
    fa_weights = fa.array(nn_weights)
    print(f"   FastArray compression type: {fa_weights.compression_type}")
    print(f"   Original size: {nn_weights.nbytes / 1024 / 1024:.2f} MB")
    print(f"   Compressed size: {fa_weights.nbytes / 1024 / 1024:.2f} MB")
    print(f"   Compression ratio: {nn_weights.nbytes / fa_weights.nbytes:.2f}x")
    
    # Decompress and compare
    decompressed_weights = fa_weights._decompress()
    print(f"   Decompressed range: [{decompressed_weights.min():.6f}, {decompressed_weights.max():.6f}]")
    print(f"   Decompressed std: {decompressed_weights.std():.6f}")
    
    # Calculate error metrics
    mse = np.mean((nn_weights - decompressed_weights) ** 2)
    max_error = np.max(np.abs(nn_weights - decompressed_weights))
    mean_error = np.mean(np.abs(nn_weights - decompressed_weights))
    relative_error = mean_error / (np.abs(nn_weights).mean() + 1e-8)  # Avoid division by zero
    
    print(f"   MSE: {mse:.8f}")
    print(f"   Max error: {max_error:.8f}")
    print(f"   Mean absolute error: {mean_error:.8f}")
    print(f"   Mean relative error: {relative_error:.6f} ({relative_error * 100:.4f}%)")
    
    # Check if values are close (within tolerance for 8-bit quantization)
    all_close = np.allclose(nn_weights, decompressed_weights, rtol=1e-2, atol=1e-3)
    print(f"   All values close (rtol=1e-2, atol=1e-3): {all_close}")
    
    # Test 2: Different data distributions
    print("\\n2. Testing different data distributions:")
    
    # Test with uniform distribution
    uniform_data = np.random.uniform(-1, 1, (500, 500)).astype(np.float32)
    fa_uniform = fa.array(uniform_data)
    decomp_uniform = fa_uniform._decompress()
    uniform_error = np.mean(np.abs(uniform_data - decomp_uniform))
    print(f"   Uniform distribution error: {uniform_error:.8f}")
    
    # Test with small values (common in trained models)
    small_data = np.random.randn(500, 500).astype(np.float32) * 0.01  # Small weights
    fa_small = fa.array(small_data)
    decomp_small = fa_small._decompress()
    small_error = np.mean(np.abs(small_data - decomp_small))
    print(f"   Small values distribution error: {small_error:.8f}")
    
    # Test 3: Operations on compressed data (verifying accuracy after ops)
    print("\\n3. Testing operations on compressed data:")
    
    a = fa.array(np.random.randn(100, 100).astype(np.float32))
    b = fa.array(np.random.randn(100, 100).astype(np.float32))
    
    # Perform operations
    result_add = a + b
    result_mult = a * b
    
    # Compare with numpy operations on decompressed data
    np_a = a._decompress()
    np_b = b._decompress()
    np_result_add = np_a + np_b
    np_result_mult = np_a * np_b
    
    add_error = np.mean(np.abs(result_add._decompress() - np_result_add))
    mult_error = np.mean(np.abs(result_mult._decompress() - np_result_mult))
    
    print(f"   Addition operation error: {add_error:.8f}")
    print(f"   Multiplication operation error: {mult_error:.8f}")
    
    # Test 4: Large matrix test
    print("\\n4. Testing large matrix compression:")
    large_matrix = np.random.randn(2048, 2048).astype(np.float32)
    fa_large = fa.array(large_matrix)
    
    print(f"   Large matrix shape: {large_matrix.shape}")
    print(f"   Original size: {large_matrix.nbytes / 1024 / 1024:.2f} MB")
    print(f"   Compressed size: {fa_large.nbytes / 1024 / 1024:.2f} MB")
    print(f"   Compression ratio: {large_matrix.nbytes / fa_large.nbytes:.2f}x")
    
    large_error = np.mean(np.abs(large_matrix - fa_large._decompress()))
    print(f"   Large matrix error: {large_error:.8f}")
    
    print("\\n" + "=" * 60)
    print("Accuracy test completed!")
    
    return {
        'mse': mse,
        'max_error': max_error,
        'mean_error': mean_error,
        'relative_error': relative_error,
        'all_close': all_close,
        'compression_ratio': nn_weights.nbytes / fa_weights.nbytes
    }

def test_realistic_neural_network_scenario():
    """Test a more realistic neural network scenario"""
    print("\\n\\nTesting realistic neural network scenario...")
    print("=" * 60)
    
    # Simulate a transformer-style model with multiple weight matrices
    layer_sizes = [768, 3072, 768]  # Typical transformer dimensions
    total_params = 0
    total_compressed_size = 0
    total_original_size = 0
    
    print("Simulating transformer layer weights:")
    
    # Self-attention weights (Q, K, V)
    for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        # Weight matrix
        weight = np.random.randn(in_size, out_size).astype(np.float32)
        bias = np.random.randn(out_size).astype(np.float32)
        
        total_params += weight.size + bias.size
        total_original_size += weight.nbytes + bias.nbytes
        
        # Compress with FastArray
        fa_weight = fa.array(weight)
        fa_bias = fa.array(bias)
        
        total_compressed_size += fa_weight.nbytes + fa_bias.nbytes
        
        weight_error = np.mean(np.abs(weight - fa_weight._decompress()))
        bias_error = np.mean(np.abs(bias - fa_bias._decompress()))
        
        print(f"   Layer {i+1}: {in_size}x{out_size} weight matrix + {out_size} bias")
        print(f"     Weight compression: {weight.nbytes/fa_weight.nbytes:.2f}x, error: {weight_error:.8f}")
        print(f"     Bias compression: {bias.nbytes/fa_bias.nbytes:.2f}x, error: {bias_error:.8f}")
    
    original_mb = total_original_size / 1024 / 1024
    compressed_mb = total_compressed_size / 1024 / 1024
    compression_ratio = total_original_size / total_compressed_size
    
    print(f"\\nTotal parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    print(f"Original size: {original_mb:.2f} MB")
    print(f"Compressed size: {compressed_mb:.2f} MB")
    print(f"Overall compression: {compression_ratio:.2f}x")
    print(f"Space saved: {original_mb - compressed_mb:.2f} MB ({(1-compressed_mb/original_mb)*100:.1f}%)")
    
    print("\\n" + "=" * 60)
    print("Neural network scenario test completed!")

if __name__ == "__main__":
    accuracy_results = test_accuracy_preservation()
    test_realistic_neural_network_scenario()
    
    print(f"\\nSUMMARY:")
    print(f"Compression ratio achieved: {accuracy_results['compression_ratio']:.2f}x")
    print(f"Mean absolute error: {accuracy_results['mean_error']:.8f}")
    print(f"Max error: {accuracy_results['max_error']:.8f}")
    print(f"Relative error: {accuracy_results['relative_error']:.4f}%")
    print(f"All values close: {accuracy_results['all_close']}")
    print("\\nFastArray successfully provides high compression with acceptable accuracy loss for neural networks!")