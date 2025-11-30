"""
Basic tests for FastArray functionality
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import fastarray as fa


def test_basic_functionality():
    """Test basic FastArray functionality"""
    print("Testing basic FastArray functionality...")
    
    # Test array creation
    arr = fa.array([1, 2, 3, 4, 5])
    print(f"Created array: {arr}")
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}")
    
    # Test operations
    result = arr + 2
    print(f"Addition result: {result}")
    
    result = arr * 2
    print(f"Multiplication result: {result}")
    
    # Test numpy compatibility
    np_arr = np.array([1, 2, 3, 4, 5])
    fa_arr = fa.array([1, 2, 3, 4, 5])

    np_result = np.add(np_arr, 10)

    # Using the imported function from numpy_api
    fa_result = fa.add(fa_arr, 10)

    print(f"NumPy result: {np_result}")
    print(f"FastArray result: {fa_result}")

    # Check if fa_result is a FastArray or regular array
    if hasattr(fa_result, '_decompress'):
        fa_result_decompressed = fa_result._decompress()
    else:
        fa_result_decompressed = np.asarray(fa_result)

    print(f"Results equal: {np.array_equal(np_result, fa_result_decompressed)}")
    

def test_compression():
    """Test compression functionality"""
    print("\nTesting compression functionality...")
    
    # Test quantization on float array
    float_arr = fa.array([1.1, 2.2, 3.3, 4.4, 5.5])
    print(f"Float array: {float_arr}, compression: {float_arr.compression_type}")
    
    # Test sparse compression
    sparse_data = [0, 0, 0, 4, 0, 0, 7, 0, 0, 10]
    sparse_arr = fa.array(sparse_data)
    print(f"Sparse array: {sparse_arr}, compression: {sparse_arr.compression_type}")
    
    # Test large array (should use blosc or similar)
    large_arr = fa.zeros((100, 100))
    print(f"Large array shape: {large_arr.shape}, compression: {large_arr.compression_type}")


def test_linalg():
    """Test linear algebra functionality"""
    print("\nTesting linear algebra functionality...")
    
    a = fa.array([[1, 2], [3, 4]])
    b = fa.array([[5, 6], [7, 8]])
    
    print(f"Matrix A:\n{a}")
    print(f"Matrix B:\n{b}")
    
    result = fa.dot(a, b)
    print(f"A dot B:\n{result}")
    
    # Verify with numpy
    np_a = np.array([[1, 2], [3, 4]])
    np_b = np.array([[5, 6], [7, 8]])
    np_result = np.dot(np_a, np_b)
    
    print(f"NumPy A dot B:\n{np_result}")

    # Check if result is a FastArray or regular array
    if hasattr(result, '_decompress'):
        result_decompressed = result._decompress()
    else:
        result_decompressed = np.asarray(result)

    print(f"Results equal: {np.array_equal(np_result, result_decompressed)}")


def test_memory_functions():
    """Test memory management functionality"""
    print("\nTesting memory management...")
    
    large_arr = fa.ones((50, 50))
    print(f"Created large array of shape: {large_arr.shape}")
    
    # Test offloading
    file_path = fa.memory.offload_array_to_disk(large_arr, "test_array")
    print(f"Array offloaded to: {file_path}")
    
    # Load it back
    loaded_arr = fa.memory.load_array_from_disk_offloaded(file_path, large_arr.compression_type)
    print(f"Loaded array shape: {loaded_arr.shape}")

    # Check if loaded_arr is a FastArray or regular array
    if hasattr(loaded_arr, '_decompress'):
        loaded_decompressed = loaded_arr._decompress()
    else:
        loaded_decompressed = np.asarray(loaded_arr)

    print(f"Arrays equal: {np.array_equal(large_arr._decompress(), loaded_decompressed)}")


def test_index_system():
    """Test the index file system"""
    print("\nTesting index system...")
    
    test_arr = fa.array([[1, 2, 3], [4, 5, 6]])
    
    # Save array to index
    saved_path = fa.index.save_array_to_disk(test_arr, "test_matrix", 
                                             metadata={"purpose": "testing"})
    print(f"Array saved to: {saved_path}")
    
    # List saved arrays
    saved_arrays = fa.index.list_saved_arrays()
    print(f"Saved arrays: {saved_arrays}")
    
    # Load array from index
    loaded = fa.index.load_array_from_disk("test_matrix")
    print(f"Loaded array:\n{loaded}")
    
    # Verify they're the same
    if hasattr(loaded, '_decompress'):
        loaded_decompressed = loaded._decompress()
    else:
        loaded_decompressed = np.asarray(loaded)

    print(f"Arrays equal: {np.array_equal(test_arr._decompress(), loaded_decompressed)}")
    
    # Get array info
    info = fa.index.get_file_manager().index.get_array_info("test_matrix")
    print(f"Array info: {info}")


if __name__ == "__main__":
    print("Running FastArray tests...\n")
    
    test_basic_functionality()
    test_compression()
    test_linalg()
    test_memory_functions()
    test_index_system()
    
    print("\nAll tests completed!")