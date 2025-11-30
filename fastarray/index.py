"""
Index file system for managing compressed arrays
This module provides functionality to store, retrieve, and index compressed arrays on disk
"""
import numpy as np
import os
import pickle
import json
from typing import Dict, List, Tuple, Any, Optional
from .fastarray import FastArray
from .compression import compress_array, decompress_array


class ArrayIndex:
    """
    Index system for managing compressed arrays on disk
    """
    
    def __init__(self, index_path: str = "fastarray_index.json"):
        self.index_path = index_path
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load the index from disk, or create a new one if it doesn't exist"""
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "arrays": {},
                "metadata": {
                    "version": "1.0",
                    "created": str(np.datetime64('now'))
                }
            }
    
    def _save_index(self):
        """Save the index to disk"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def add_array(self, array: FastArray, name: str, 
                  path: Optional[str] = None, 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a compressed array to the index system
        
        Parameters:
        - array: FastArray to store
        - name: name to identify the array
        - path: optional path to store the array (if None, auto-generate)
        - metadata: optional metadata to store with the array
        
        Returns:
        - The path where the array was stored
        """
        if path is None:
            path = f"fastarray_{name}_{len(self.index['arrays'])}.npz"
        
        # Compress and save the array
        if array.compression_type == "sparse":
            compressed_data = array._compressed_data  # Already in sparse format
            format_type = "sparse"
        else:
            # For other compression types, just save the numpy representation
            compressed_data = array._decompress()
            format_type = "numpy"
        
        # Save to disk
        if format_type == "sparse":
            # Save sparse format separately
            sparse_path = path.replace('.npz', '_sparse.pkl')
            with open(sparse_path, 'wb') as f:
                pickle.dump(compressed_data, f)
            actual_path = sparse_path
        else:
            # Save using numpy's savez_compressed for efficiency
            np.savez_compressed(path, data=compressed_data)
            actual_path = path
        
        # Update index
        self.index["arrays"][name] = {
            "path": actual_path,
            "compression_type": array.compression_type,
            "shape": array.shape,
            "dtype": str(array.dtype),
            "size_bytes": array.nbytes,
            "format_type": format_type,
            "metadata": metadata or {},
            "timestamp": str(np.datetime64('now'))
        }
        
        # Save updated index
        self._save_index()
        
        return actual_path
    
    def get_array(self, name: str) -> FastArray:
        """
        Retrieve an array from the index system
        
        Parameters:
        - name: name of the array to retrieve
        
        Returns:
        - FastArray instance
        """
        if name not in self.index["arrays"]:
            raise KeyError(f"Array '{name}' not found in index")
        
        array_info = self.index["arrays"][name]
        path = array_info["path"]
        compression_type = array_info["compression_type"]
        format_type = array_info["format_type"]
        
        # Load the data
        if format_type == "sparse":
            # Load sparse format
            with open(path, 'rb') as f:
                compressed_data = pickle.load(f)
            # Recreate the array from sparse representation
            if isinstance(compressed_data, tuple) and len(compressed_data) == 3:
                indices, values, shape = compressed_data
                temp_array = np.zeros(shape, dtype=array_info["dtype"])
                if len(shape) == 1:
                    temp_array[indices] = values
                elif len(shape) == 2:
                    temp_array[indices[0], indices[1]] = values
                else:
                    flat_indices = np.ravel_multi_index(indices, shape)
                    flat_result = np.zeros(np.prod(shape), dtype=array_info["dtype"])
                    flat_result[flat_indices] = values
                    temp_array = flat_result.reshape(shape)
            else:
                temp_array = decompress_array(compressed_data, method=compression_type)
        else:
            # Load numpy format
            loaded = np.load(path)
            temp_array = loaded['data']
        
        # Return as FastArray
        return FastArray(temp_array, compression=compression_type)
    
    def list_arrays(self) -> List[str]:
        """List all arrays in the index"""
        return list(self.index["arrays"].keys())
    
    def remove_array(self, name: str):
        """Remove an array from the index and delete its file"""
        if name not in self.index["arrays"]:
            raise KeyError(f"Array '{name}' not found in index")
        
        path = self.index["arrays"][name]["path"]
        
        # Delete the file
        if os.path.exists(path):
            os.remove(path)
        
        # Remove from index
        del self.index["arrays"][name]
        
        # Save updated index
        self._save_index()
    
    def get_array_info(self, name: str) -> Dict[str, Any]:
        """Get information about an array"""
        if name not in self.index["arrays"]:
            raise KeyError(f"Array '{name}' not found in index")
        
        return self.index["arrays"][name].copy()
    
    def update_metadata(self, name: str, metadata: Dict[str, Any]):
        """Update metadata for an array"""
        if name not in self.index["arrays"]:
            raise KeyError(f"Array '{name}' not found in index")
        
        self.index["arrays"][name]["metadata"].update(metadata)
        
        # Save updated index
        self._save_index()


class ArrayFileManager:
    """
    File management system for compressed arrays
    """
    
    def __init__(self, base_path: str = "./fastarray_storage/"):
        self.base_path = base_path
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        self.index = ArrayIndex(os.path.join(base_path, "index.json"))
    
    def save_array(self, array: FastArray, name: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save an array with automatic file management"""
        return self.index.add_array(array, name, metadata=metadata)
    
    def load_array(self, name: str) -> FastArray:
        """Load an array by name"""
        return self.index.get_array(name)
    
    def delete_array(self, name: str):
        """Delete an array by name"""
        self.index.remove_array(name)
    
    def list_arrays(self) -> List[str]:
        """List all stored arrays"""
        return self.index.list_arrays()
    
    def array_exists(self, name: str) -> bool:
        """Check if an array exists"""
        return name in self.index.index["arrays"]


# Global array file manager instance
_file_manager = ArrayFileManager()


def get_file_manager() -> ArrayFileManager:
    """Get the global file manager instance"""
    return _file_manager


def save_array_to_disk(array: FastArray, name: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> str:
    """Save a FastArray to disk with automatic indexing"""
    return _file_manager.save_array(array, name, metadata)


def load_array_from_disk(name: str) -> FastArray:
    """Load a FastArray from disk using its name"""
    return _file_manager.load_array(name)


def delete_array_from_disk(name: str):
    """Delete a FastArray from disk"""
    _file_manager.delete_array(name)


def list_saved_arrays() -> List[str]:
    """List all arrays saved to disk"""
    return _file_manager.list_arrays()


def array_exists_on_disk(name: str) -> bool:
    """Check if an array exists on disk"""
    return _file_manager.array_exists(name)