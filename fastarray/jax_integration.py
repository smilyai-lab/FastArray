"""
JAX/TPU integration for FastArray
Enables using FastArray compressed arrays with JAX for TPU training
"""
import numpy as np
from typing import Optional, Any, Tuple, Dict
from .fastarray import FastArray
from .backend import JAX_AVAILABLE, to_jax_array, set_backend, get_current_backend


class JAXFastArray:
    """
    Wrapper to seamlessly integrate FastArray with JAX for TPU operations
    """
    
    def __init__(self, fastarray: FastArray, sharding_rule=None):
        """
        Initialize a JAXFastArray from a FastArray
        
        Args:
            fastarray: The FastArray to wrap
            sharding_rule: JAX PartitionSpec for sharding across devices
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not available. Please install JAX for TPU operations.")
        
        # Import locally to avoid issues when JAX isn't available
        from jax.sharding import PartitionSpec as P

        self._fastarray = fastarray
        self._sharding_rule = sharding_rule
        
        # Convert to JAX array
        self._jax_array = to_jax_array(fastarray, sharding_rule)
        
    def to_jax(self):
        """Get the underlying JAX array"""
        return self._jax_array
    
    def to_fastarray(self):
        """Get the underlying FastArray"""
        return self._fastarray
    
    def to_numpy(self):
        """Get the decompressed data as a numpy array"""
        return self._fastarray._decompress()
    
    @property
    def shape(self):
        """Shape of the array"""
        return self._jax_array.shape
    
    @property
    def dtype(self):
        """Data type of the array"""
        return self._jax_array.dtype
    
    @property
    def sharding_rule(self):
        """Sharding rule for this array"""
        return self._sharding_rule


def create_jax_from_fastarray(fastarray: FastArray, sharding_rule=None):
    """
    Create a JAX array directly from a FastArray with optional sharding
    
    Args:
        fastarray: The FastArray to convert
        sharding_rule: Optional sharding specification
        
    Returns:
        JAX array ready for TPU operations
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not available. Please install JAX for TPU operations.")
    
    return to_jax_array(fastarray, sharding_rule)


def create_sharding_rules_for_model(vocab_size: int, d_model: int, ff_dim: int, n_heads: int, model_parallel: int):
    """
    Create appropriate sharding rules for transformer models
    
    Args:
        vocab_size: Model vocabulary size
        d_model: Model dimension
        ff_dim: Feed-forward dimension
        n_heads: Number of attention heads
        model_parallel: Model parallelism degree
        
    Returns:
        Dictionary of parameter sharding rules
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not available. Sharding rules require JAX.")
    
    # Import locally so function works when JAX is available
    from jax.sharding import PartitionSpec as P
    
    d_shardable = d_model % model_parallel == 0
    ff_shardable = ff_dim % model_parallel == 0
    vocab_shardable = vocab_size % model_parallel == 0

    rules = {
        'embed_tokens/embedding': P(None, 'model') if vocab_shardable else P(None, None),
        'q_proj/kernel': P(None, 'model') if d_shardable else P(None, None),
        'k_proj/kernel': P(None, 'model') if d_shardable else P(None, None),
        'v_proj/kernel': P(None, 'model') if d_shardable else P(None, None),
        'o_proj/kernel': P('model', None) if d_shardable else P(None, None),
        'gate_proj/kernel': P(None, 'model') if ff_shardable else P(None, None),
        'up_proj/kernel': P(None, 'model') if ff_shardable else P(None, None),
        'down_proj/kernel': P('model', None) if ff_shardable else P(None, None),
        'attn_norm/scale': P(None,),
        'ffn_norm/scale': P(None,),
        'norm/scale': P(None,),
        'lm_head/kernel': P('model', None) if vocab_shardable else P(None, None),
    }

    return rules


def shard_params_with_fastarray(params_dict: Dict[str, Any], mesh: Any, sharding_rules: Dict[str, Any]):
    """
    Shard parameters using FastArray compression and JAX sharding
    
    Args:
        params_dict: Dictionary of parameters (as FastArrays or numpy arrays)
        mesh: JAX mesh for sharding
        sharding_rules: Sharding rules dictionary
        
    Returns:
        Dictionary of sharded JAX arrays
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not available. Sharding requires JAX.")
    
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P, NamedSharding
    from jax import tree_util
    
    def get_param_spec(param_name: str, sharding_rules: Dict[str, Any]):
        for pattern, spec in sharding_rules.items():
            if pattern in param_name:
                return spec
        return P()  # Default: replicate across all devices

    sharded_params = {}
    for name, param in params_dict.items():
        if isinstance(param, FastArray):
            # First convert FastArray to JAX array
            jax_param = to_jax_array(param)
        else:
            # If it's already a numpy array or JAX array
            jax_param = jnp.array(param)
        
        # Determine sharding specification
        spec = get_param_spec(name, sharding_rules)
        sharding = NamedSharding(mesh, spec)
        
        # Apply sharding
        sharded_params[name] = jax.device_put(jax_param, sharding)
    
    return sharded_params


def fastarray_to_jax_state(fastarray_state: Dict[str, Any], mesh: Any, sharding_rules: Dict[str, Any]):
    """
    Convert a training state with FastArray parameters to JAX sharded state
    
    Args:
        fastarray_state: Training state with FastArray parameters
        mesh: JAX mesh for sharding
        sharding_rules: Sharding rules for parameters
        
    Returns:
        JAX training state with sharded parameters
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not available. State conversion requires JAX.")
        
    sharded_params = shard_params_with_fastarray(fastarray_state['params'], mesh, sharding_rules)
    
    return {
        'params': sharded_params,
        'apply_fn': fastarray_state['apply_fn'],
        'tx': fastarray_state['tx'],
        'opt_state': fastarray_state['opt_state'],
        'step': fastarray_state['step'],
        'dropout_rng': fastarray_state['dropout_rng']
    }


def create_jax_mesh(mesh_shape: Tuple[int, int]):
    """
    Create a JAX mesh for distributed training
    
    Args:
        mesh_shape: Shape of the device mesh (data_parallel, model_parallel)
        
    Returns:
        JAX Mesh object
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not available.")
    
    try:
        from jax.experimental import mesh_utils
        import jax
        mesh_devices = mesh_utils.create_device_mesh(mesh_shape)
        mesh = jax.sharding.Mesh(mesh_devices, axis_names=('data', 'model'))
        return mesh
    except Exception as e:
        print(f"Warning: Could not create JAX mesh: {e}")
        # Try to create a simple mesh with available devices
        import jax
        devices = jax.devices()
        if devices:
            # Reshape devices to match mesh_shape if possible
            try:
                reshaped_devices = np.array(devices).reshape(mesh_shape)
                mesh = jax.sharding.Mesh(reshaped_devices, axis_names=('data', 'model'))
                return mesh
            except:
                # Fallback to single device
                mesh = jax.sharding.Mesh(np.array(devices[:1]), axis_names=('data', 'model'))
                print("Using single device mesh as fallback")
                return mesh
        else:
            raise RuntimeError("No JAX devices available")


def convert_model_weights_for_tpu(fastarray_weights: Dict[str, Any], sharding_rules: Dict[str, Any], mesh: Any):
    """
    Convert FastArray model weights for TPU training with appropriate sharding
    
    Args:
        fastarray_weights: Dictionary of FastArray weights
        sharding_rules: Sharding rules for different parameter types
        mesh: JAX mesh for sharding
        
    Returns:
        Dictionary of sharded JAX arrays ready for TPU
    """
    return shard_params_with_fastarray(fastarray_weights, mesh, sharding_rules)


def fastarray_jax_train_step(state: Any, batch: Any, apply_fn: Any, loss_fn: Any):
    """
    A training step function that works with FastArray-converted JAX arrays
    
    Args:
        state: Training state with sharded parameters
        batch: Input batch (already converted to JAX)
        apply_fn: Model apply function
        loss_fn: Loss function
        
    Returns:
        Updated state and metrics
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not available. Training step requires JAX.")
        
    import jax
    from jax import tree_util
    
    def loss_and_metrics(params):
        logits = apply_fn({'params': params}, batch['input_ids'], training=True)
        loss, metrics = loss_fn(logits, batch['labels'])
        return loss, metrics

    grad_fn = jax.value_and_grad(loss_and_metrics, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)

    # Apply sharding to gradients
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    metrics = jax.lax.pmean(metrics, axis_name='batch')

    # Update state with new parameters
    new_state = state.apply_gradients(grads=grads)

    return new_state, {'loss': loss, **metrics}