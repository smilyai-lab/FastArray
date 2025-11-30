"""
Memory management for FastArray - memory mapping and offloading capabilities
"""
import numpy as np
import os
from typing import Optional, Union
from .fastarray import FastArray
from .compression import compress_array, decompress_array


class MemoryManager:
    """
    Handles memory mapping and offloading of FastArray objects
    """
    
    def __init__(self):
        self.mapped_arrays = {}  # Track memory-mapped arrays
        self.offload_directory = "./fastarray_offload/"
        
        if not os.path.exists(self.offload_directory):
            os.makedirs(self.offload_directory)
    
    def memory_map(self, array: FastArray, filename: str, mode: str = 'r+') -> FastArray:
        """
        Create a memory-mapped FastArray that accesses data on disk
        
        Parameters:
        - array: FastArray to memory map
        - filename: file to store the array data
        - mode: file access mode ('r', 'r+', 'w+', 'c')
        
        Returns:
        - Memory-mapped FastArray
        """
        # First decompress and save the array to disk in numpy format
        numpy_array = array._decompress()
        full_path = os.path.join(self.offload_directory, filename)
        
        # Save using numpy's memmap capability
        numpy.save(full_path, numpy_array)
        
        # Now create a memory map to the saved file
        mmap_array = np.memmap(full_path + '.npy', dtype=numpy_array.dtype, 
                               mode=mode, shape=numpy_array.shape)
        
        # Create a new FastArray that wraps the memory-mapped array
        # We'll track this in our mapped arrays dict
        mmap_fastarray = FastArray(mmap_array, compression=array.compression_type)
        self.mapped_arrays[filename] = {
            'array': mmap_fastarray,
            'file_path': full_path + '.npy',
            'original_compression': array.compression_type
        }
        
        return mmap_fastarray
    
    def offload_to_disk(self, array: FastArray, name: str, 
                        compression: Optional[str] = None) -> str:
        """
        Offload a FastArray to disk to free up memory
        
        Parameters:
        - array: FastArray to offload
        - name: name for the offloaded array
        - compression: compression method to use (defaults to array's compression)
        
        Returns:
        - Path where the array was stored
        """
        if compression is None:
            compression = array.compression_type
        
        # Determine the best storage format based on compression type
        if compression == "sparse":
            # Save separately as this might have special handling
            file_path = os.path.join(self.offload_directory, f"{name}_sparse.npz")
            compressed_data = array._compressed_data
            if isinstance(compressed_data, tuple) and len(compressed_data) == 3:
                indices, values, shape = compressed_data
                np.savez_compressed(file_path, indices=indices, values=values, shape=shape)
            else:
                # Fallback to default saving
                numpy_array = array._decompress()
                np.savez_compressed(file_path, data=numpy_array)
        else:
            # Use compressed numpy format for efficiency
            file_path = os.path.join(self.offload_directory, f"{name}.npz")
            numpy_array = array._decompress()
            np.savez_compressed(file_path, data=numpy_array)
        
        return file_path
    
    def load_offloaded(self, file_path: str, compression: str = "auto") -> FastArray:
        """
        Load an offloaded FastArray from disk
        
        Parameters:
        - file_path: path to the offloaded array
        - compression: compression method that was used
        
        Returns:
        - Loaded FastArray
        """
        # Determine if it's a sparse save or regular save
        if "_sparse" in os.path.basename(file_path):
            # Load sparse format
            loaded = np.load(file_path)
            if 'indices' in loaded and 'values' in loaded and 'shape' in loaded:
                indices = loaded['indices']
                values = loaded['values']
                shape = tuple(loaded['shape'])
                
                # Reconstruct the array
                dtype = values.dtype
                reconstructed = np.zeros(shape, dtype=dtype)
                if len(shape) == 1:
                    reconstructed[indices] = values
                elif len(shape) == 2:
                    reconstructed[indices[0], indices[1]] = values
                else:
                    # Handle higher dimensions
                    flat_indices = np.ravel_multi_index(indices, shape)
                    flat_result = np.zeros(np.prod(shape), dtype=dtype)
                    flat_result[flat_indices] = values
                    reconstructed = flat_result.reshape(shape)
                
                return FastArray(reconstructed, compression=compression)
        else:
            # Load regular compressed format
            loaded = np.load(file_path)
            array_data = loaded['data']
            return FastArray(array_data, compression=compression)
    
    def swap_to_disk(self, array: FastArray, name: str, 
                     temp: bool = True) -> Union[str, FastArray]:
        """
        Swap array to disk temporarily, optionally returning a placeholder
        
        Parameters:
        - array: FastArray to swap out
        - name: name for the swapped array
        - temp: if True, return file path; if False, return reloaded array
        
        Returns:
        - File path (if temp=True) or reloaded FastArray (if temp=False)
        """
        file_path = self.offload_to_disk(array, name, array.compression_type)
        
        if temp:
            return file_path
        else:
            # Clear the original array from memory and reload
            del array  # This helps free memory
            return self.load_offloaded(file_path, array.compression_type)
    
    def get_memory_usage(self) -> dict:
        """
        Get information about memory usage and mapped arrays
        
        Returns:
        - Dictionary with memory usage information
        """
        mapped_info = {}
        total_size = 0
        
        for name, info in self.mapped_arrays.items():
            arr = info['array']
            size = arr.nbytes if hasattr(arr, 'nbytes') else 0
            total_size += size
            mapped_info[name] = {
                'size_bytes': size,
                'shape': arr.shape if hasattr(arr, 'shape') else None,
                'dtype': str(arr.dtype) if hasattr(arr, 'dtype') else None,
                'file_path': info['file_path']
            }
        
        return {
            'mapped_arrays': mapped_info,
            'total_mapped_size': total_size,
            'num_mapped_arrays': len(self.mapped_arrays),
            'offload_directory': self.offload_directory
        }


# Global memory manager instance
_memory_manager = MemoryManager()


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    return _memory_manager


def memory_map_array(array: FastArray, filename: str, mode: str = 'r+') -> FastArray:
    """Create a memory-mapped version of a FastArray"""
    return _memory_manager.memory_map(array, filename, mode)


def offload_array_to_disk(array: FastArray, name: str, 
                         compression: Optional[str] = None) -> str:
    """Offload a FastArray to disk"""
    return _memory_manager.offload_to_disk(array, name, compression)


def load_array_from_disk_offloaded(file_path: str, compression: str = "auto") -> FastArray:
    """Load a FastArray that was previously offloaded to disk"""
    return _memory_manager.load_offloaded(file_path, compression)


def swap_array_to_disk(array: FastArray, name: str, 
                      temp: bool = True) -> Union[str, FastArray]:
    """Swap array to disk temporarily"""
    return _memory_manager.swap_to_disk(array, name, temp)


def get_memory_usage_info() -> dict:
    """Get memory usage information"""
    return _memory_manager.get_memory_usage()