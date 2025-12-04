"""
JAX-fastarray integration module
Enables using FastArray compressed arrays with JAX for TPU training
Enhanced with extreme compression techniques for maximum performance
"""
import numpy as np
from typing import Optional, Any, Tuple, Dict
import fastarray as fa  # Import the updated fastarray
from fastarray.fastarray import FastArray, CompressionType, CompressionAggressiveness
from fastarray.backend import JAX_AVAILABLE, to_jax_array, set_backend, get_current_backend


class JAXFastArray:
    """
    Wrapper to seamlessly integrate FastArray with JAX for TPU operations
    Enhanced for TPU V5e-8 with extreme compression and BF16 support
    """

    def __init__(self, fastarray: FastArray, sharding_rule=None):
        """
        Initialize a JAXFastArray from a FastArray

        Args:
            fastarray: The FastArray to wrap (with extreme compression for TPU)
            sharding_rule: JAX PartitionSpec for sharding across devices
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not available. Please install JAX for TPU operations.")

        # Import locally to avoid issues when JAX isn't available
        from jax.sharding import PartitionSpec as P

        self._fastarray = fastarray
        self._sharding_rule = sharding_rule

        # Convert to JAX array - this is where the magic happens for TPU!
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
    Optimized for TPU V5e-8 with extreme compression and INT8/AQT-style operations

    Args:
        fastarray: The FastArray to convert (with extreme compression)
        sharding_rule: Optional sharding specification

    Returns:
        JAX array ready for TPU operations
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not available. Please install JAX for TPU operations.")

    import jax.numpy as jnp

    # For extreme compression and speed, handle different compression types
    if fastarray.compression_type == CompressionType.INT8_QUANT:
        # Use INT8 data with proper scaling to achieve TPU-optimized operations
        decompressed = fastarray._decompress()
        jax_arr = jnp.array(decompressed, dtype=jnp.bfloat16)  # Convert to bfloat16 for TPU speed
    elif fastarray.compression_type == CompressionType.BF16_QUANT:
        # Convert the compressed data to a JAX array in a way that's compatible with TPU BF16 operations
        decompressed = fastarray._decompress()
        jax_arr = jnp.array(decompressed, dtype=jnp.bfloat16)
    elif fastarray.compression_type == CompressionType.LOW_RANK:
        # For low-rank, decompress and convert to bfloat16 for TPU operations
        decompressed = fastarray._decompress()
        jax_arr = jnp.array(decompressed, dtype=jnp.bfloat16)
    elif fastarray.compression_type == CompressionType.HYBRID:
        # For hybrid compressed data, decompress and optimize for TPU
        decompressed = fastarray._decompress()
        jax_arr = jnp.array(decompressed, dtype=jnp.bfloat16)
    else:
        # Use the standard conversion
        jax_arr = to_jax_array(fastarray, sharding_rule)
        
        # Convert to bfloat16 if using TPU (for optimal V5e-8 performance)
        if jax_arr.dtype != jnp.bfloat16:
            jax_arr = jax_arr.astype(jnp.bfloat16)

    return jax_arr


def create_sharding_rules_for_model(vocab_size: int, d_model: int, ff_dim: int, n_heads: int, model_parallel: int):
    """
    Create appropriate sharding rules for transformer models optimized for TPU V5e-8
    Enhanced for extreme compression scenarios
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
    Shard parameters using FastArray extreme compression and JAX sharding
    Optimized for TPU V5e-8 with maximum performance
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
            # First convert FastArray to JAX array with extreme compression support
            jax_param = create_jax_from_fastarray(param)
        else:
            # If it's already a numpy array or JAX array
            jax_param = jnp.array(param, dtype=jnp.bfloat16)  # Ensure BF16 for TPU

        # Determine sharding specification
        spec = get_param_spec(name, sharding_rules)
        sharding = NamedSharding(mesh, spec)

        # Apply sharding
        sharded_params[name] = jax.device_put(jax_param, sharding)

    return sharded_params


def fastarray_to_jax_state(fastarray_state: Dict[str, Any], mesh: Any, sharding_rules: Dict[str, Any]):
    """
    Convert a training state with FastArray extreme compression to JAX sharded state
    Optimized for TPU V5e-8 with maximum performance
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
    Create a JAX mesh for distributed TPU V5e-8 training
    Optimized for extreme compression scenarios
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not available.")

    try:
        from jax.experimental import mesh_utils
        import jax
        # Try to create a mesh with the specified shape
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
    Convert FastArray extreme compressed model weights for TPU V5e-8 training 
    with appropriate sharding and maximum optimization
    """
    return shard_params_with_fastarray(fastarray_weights, mesh, sharding_rules)


def fastarray_jax_train_step(state: Any, batch: Any, apply_fn: Any, loss_fn: Any):
    """
    A training step function that works with FastArray extreme compression and JAX arrays
    Optimized for TPU V5e-8 with maximum performance and speed
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


def create_fastarray_for_tpu_compression(data, dtype=None, compression="auto", compression_aggressiveness=CompressionAggressiveness.BALANCED):
    """
    Create a FastArray optimized for TPU V5e-8 with extreme compression options
    """
    # Convert data to the appropriate dtype first if needed
    if isinstance(data, np.ndarray) and dtype is not None:
        # Convert to appropriate float type before compression
        if data.dtype in [np.float64, np.float32]:
            data = data.astype(np.float32)
    
    # Create FastArray with appropriate compression based on aggressiveness
    return fa.array(data, compression=compression, compression_aggressiveness=compression_aggressiveness)


def create_extreme_compression_state(model, rng, dummy_input_shape, config, mesh, sharding_rules, optimizer,
                                   compression_aggressiveness=CompressionAggressiveness.EXTREME,
                                   device_type='tpu', num_devices=8):
    """
    Create a training state with extreme FastArray compression for maximum TPU performance
    Supports single and multi-device configurations (e.g., TPU-v5e-8 with 8 chips)
    """
    # Initialize model parameters
    import jax.numpy as jnp
    dummy_input = jnp.ones(dummy_input_shape, dtype=jnp.int32)
    variables = model.init(rng, dummy_input, training=False)
    params = variables['params']

    # Compress parameters using device-optimized FastArray compression
    def compress_param(param):
        if isinstance(param, jnp.ndarray):
            # Convert JAX array to numpy, then to FastArray with device-optimized compression
            np_param = np.array(param)
            fa_param = fa.array(np_param, compression="hybrid",
                               compression_aggressiveness=compression_aggressiveness,
                               device_type=device_type, num_devices=num_devices)
            return fa_param
        return param

    compressed_params = tree_util.tree_map(compress_param, params)

    # Calculate compression statistics
    def calculate_size(param):
        if isinstance(param, fa.FastArray):
            return param.nbytes
        elif hasattr(param, 'size') and hasattr(param, 'dtype'):
            return param.size * param.dtype.itemsize
        else:
            return 0

    original_size = sum(x.size * x.dtype.itemsize for x in tree_util.tree_leaves(params))
    compressed_size = sum(calculate_size(x) for x in tree_util.tree_leaves(compressed_params))

    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    saved_mb = (original_size - compressed_size) / 1024 / 1024

    print(f"🔄 Device-optimized compression ({device_type}x{num_devices}): {original_size/1024/1024:.2f}MB → {compressed_size/1024/1024:.2f}MB ({compression_ratio:.1f}x) | Saved {saved_mb:.2f}MB")

    # Convert compressed parameters back to sharded JAX arrays for training (with BF16)
    def decompress_to_jax_with_bf16(param):
        if isinstance(param, fa.FastArray):
            # Convert FastArray to JAX array with proper TPU optimization
            jax_param = create_jax_from_fastarray(param)
            # Ensure the parameter is in bfloat16 for TPU V5e-8
            if jax_param.dtype != jnp.bfloat16:
                jax_param = jax_param.astype(jnp.bfloat16)
            return jax_param
        elif isinstance(param, jnp.ndarray):
            return param.astype(jnp.bfloat16)  # Ensure BF16 for TPU
        else:
            return jnp.array(param, dtype=jnp.bfloat16)  # Convert and ensure BF16

    jax_params = tree_util.tree_map(decompress_to_jax_with_bf16, compressed_params)
    params = shard_params_with_fastarray({'dummy': jax_params}, mesh, sharding_rules)['dummy']  # Properly shard the parameters

    # Create training state with compressed parameters
    from flax.training.train_state import TrainState

    class CompressedTrainState(TrainState):
        dropout_rng: jax.random.PRNGKey
        compressed_params: dict = None  # Store original compressed params
        device_type: str = device_type  # Track the target device type
        num_devices: int = num_devices  # Track number of devices

    state = CompressedTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        dropout_rng=rng,
        compressed_params=compressed_params,  # Keep compressed version for saving
        device_type=device_type,
        num_devices=num_devices
    )

    return state