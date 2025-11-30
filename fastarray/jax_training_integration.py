"""
FastArray Integration Module for JAX/TPU Training
This module provides functions to integrate FastArray compression with JAX training workflows
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax import tree_util
import numpy as np
import fastarray as fa
from fastarray.jax_integration import to_jax_array


def compress_parameters_with_fastarray(params):
    """
    Compress model parameters using FastArray for storage efficiency
    """
    def compress_param(param):
        if isinstance(param, jnp.ndarray):
            # Convert JAX array to numpy, then to FastArray for compression
            np_param = np.array(param)
            fa_param = fa.array(np_param, compression="quantization")
            return fa_param
        else:
            # If already FastArray or other type, return as is
            return param
    
    return tree_util.tree_map(compress_param, params)


def decompress_params_to_jax(compressed_params, mesh=None, sharding_rules=None):
    """
    Convert compressed FastArray parameters back to JAX arrays for computation
    """
    def decompress_to_jax(path, fa_param):
        if isinstance(fa_param, fa.FastArray):
            # Convert FastArray to JAX array
            jax_param = to_jax_array(fa_param)
            
            # Apply sharding if mesh and rules are provided
            if mesh is not None and sharding_rules is not None:
                # Create sharding specification for this parameter
                path_str = '/'.join(str(p.key) for p in path)
                
                # Find appropriate sharding rule
                spec = P()  # Default: replicate
                for pattern, rule in sharding_rules.items():
                    if pattern in path_str:
                        spec = rule
                        break
                
                sharding = NamedSharding(mesh, spec)
                jax_param = jax.device_put(jax_param, sharding)
            
            return jax_param
        else:
            # If already a JAX array, return as is
            return jnp.array(fa_param)
    
    return tree_util.tree_map_with_path(decompress_to_jax, compressed_params)


def create_compressed_train_state(model, rng, dummy_input_shape, config, mesh, sharding_rules, optimizer):
    """
    Create a training state with parameters compressed using FastArray
    """
    # Initialize model parameters
    dummy_input = jnp.ones(dummy_input_shape, dtype=jnp.int32)
    variables = model.init(rng, dummy_input, training=False)
    params = variables['params']
    
    # Compress parameters using FastArray
    compressed_params = compress_parameters_with_fastarray(params)
    
    # Calculate compression statistics
    original_size = sum(p.size * p.dtype.itemsize for p in tree_util.tree_leaves(params))
    compressed_size = sum(p.nbytes if hasattr(p, 'nbytes') else 0 for p in tree_util.tree_leaves(compressed_params))
    
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    print(f"🔄 Parameter compression: {original_size/1024/1024:.2f}MB -> {compressed_size/1024/1024:.2f}MB ({compression_ratio:.1f}x)")
    
    # Convert compressed parameters back to sharded JAX arrays for training
    jax_params = decompress_params_to_jax(compressed_params, mesh, sharding_rules)
    
    # Create training state with compressed parameters
    from flax.training.train_state import TrainState
    
    class CompressedTrainState(TrainState):
        dropout_rng: jax.random.PRNGKey
        compressed_params: dict = None  # Store original compressed params
    
    state = CompressedTrainState.create(
        apply_fn=model.apply,
        params=jax_params,
        tx=optimizer,
        dropout_rng=rng,
        compressed_params=compressed_params  # Keep compressed version for saving
    )
    
    return state


def save_compressed_state(state, checkpoint_dir, step):
    """
    Save the compressed parameters to disk
    """
    from flax.training import checkpoints
    
    # Save the compressed parameters (these are smaller)
    compressed_params = state.compressed_params
    
    # Convert FastArrays to numpy for saving
    def fa_to_numpy(fa_obj):
        if isinstance(fa_obj, fa.FastArray):
            return fa_obj._decompress()
        return fa_obj
    
    numpy_params = tree_util.tree_map(fa_to_numpy, compressed_params)
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=numpy_params,
        step=step,
        overwrite=True
    )


def load_compressed_state(model, checkpoint_path, mesh, sharding_rules):
    """
    Load compressed state and convert back to JAX arrays
    """
    from flax.training import checkpoints
    
    # Load the saved parameters
    saved_params = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)
    
    # Compress them using FastArray again
    compressed_params = compress_parameters_with_fastarray(saved_params)
    
    # Convert to JAX arrays with sharding
    jax_params = decompress_params_to_jax(compressed_params, mesh, sharding_rules)
    
    return jax_params, compressed_params


def get_compression_stats(params):
    """
    Calculate compression statistics for a set of parameters
    """
    original_size = 0
    compressed_size = 0
    
    def calculate_sizes(param):
        nonlocal original_size, compressed_size
        if isinstance(param, jnp.ndarray):
            original_size += param.size * param.dtype.itemsize
        elif isinstance(param, fa.FastArray):
            original_size += param._decompress().size * param._decompress().dtype.itemsize
            compressed_size += param.nbytes
    
    tree_util.tree_map(calculate_sizes, params)
    
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    saved_mb = (original_size - compressed_size) / 1024 / 1024
    
    return {
        'original_size_mb': original_size / 1024 / 1024,
        'compressed_size_mb': compressed_size / 1024 / 1024,
        'saved_mb': saved_mb,
        'compression_ratio': compression_ratio
    }


# Integration functions for the training script
def integrate_fastarray_in_training(model, rng, dummy_input_shape, config, mesh, sharding_rules, optimizer):
    """
    Integrate FastArray into the training workflow
    """
    print("🔄 Initializing model with FastArray compression...")
    
    # Create compressed training state
    state = create_compressed_train_state(model, rng, dummy_input_shape, config, mesh, sharding_rules, optimizer)
    
    # Calculate and display stats
    stats = get_compression_stats(state.compressed_params)
    print(f"✅ Model initialized with {stats['compression_ratio']:.1f}x compression")
    print(f"   Saved {stats['saved_mb']:.2f}MB of memory")
    
    return state