from typing import Union, NamedTuple, Mapping, Sequence, Callable

import jax.tree
import jax.numpy as jnp
from flax import linen as nn

from ol.utils import Array


class Inputs(NamedTuple):
  """Container for operator input data with all necessary fields and boundary conditions.
  
  Attributes:
      s: Geometry-related domain features [batch, 1, space, features]
      a: Full-domain function values [batch, 1, space, variables]
      q: Mapping of segment/boundary function values [batch, 1, space, variables] per boundary
      m: Mapping of binary masks for each boundary condition [batch, space] per boundary
      x_inp: Input spatial coordinates [space, dim]
      x_out: Output spatial coordinates for prediction [space, dim]
      t: Optional time coordinate [batch, 1]
      tau: Optional time difference/lead time parameter [batch, 1]
  """

  s: Array  # Geometry-related domain features
  a: Array  # Full-domain functions
  q: Mapping[str, Array]  # Segment function values with masks
  m: Mapping[str, Array]  # Binary function masks (not normalized)
  x_inp: Array  # Input coordinates
  x_out: Array  # Output coordinates
  t: Union[Array, float, None] = None  # Optional time coordinate
  tau: Union[Array, float, None] = None  # Optional lead time parameter

class AbstractOperator(nn.Module):
  """Base class for neural operators with input validation.
  
  Provides common interface and input validation for all operator architectures.
  Subclasses must implement the call() method for the actual operator logic.
  """

  def setup(self):
    """Initialize operator components. Override in subclasses."""
    raise NotImplementedError

  def _check_coordinates(self, x: Array) -> None:
    """Validate coordinate array format.
    
    Args:
        x: Coordinate array to validate [space, dim]
    
    Raises:
        AssertionError if coordinate constraints are violated
    """
    # Coordinates must be 2D (space points, spatial dimensions)
    assert x is not None
    assert x.ndim == 2
    # Support 1D, 2D, or 3D spatial dimensions
    assert x.shape[1] <= 3
    # Coordinates must be normalized to [-1, 1] range
    assert x.min() >= -1
    assert x.max() <= +1

  def _check_function(self, u: Array, x: Array) -> None:
    """Validate function array format.
    
    Args:
        u: Function array to validate [batch, time, space, vars]
        x: Coordinate array to match spatial dimension
    
    Raises:
        AssertionError if function constraints are violated
    """
    # Functions must be 4D (batch, time, space, variables)
    assert u is not None
    assert u.ndim == 4
    # Time dimension must be size 1 for static problems
    assert u.shape[1] == 1
    # Spatial dimension must match coordinates
    assert u.shape[2] == x.shape[2], f'u: {u.shape}, x: {x.shape}'

  def __call__(self, inputs: Inputs, **kwargs) -> Array:
    """Validate inputs and call the operator.
    
    Args:
        inputs: Inputs NamedTuple containing all operator input data
        **kwargs: Additional arguments to pass to call()
    
    Returns:
        Predicted function values [batch, time, space, vars]
    """
    # Validate input functions against their coordinate arrays
    self._check_function(inputs.s, x=inputs.x_inp)
    self._check_function(inputs.a, x=inputs.x_inp)
    # Validate all boundary condition functions
    jax.tree.map(lambda f: self._check_function(f, x=inputs.x_inp), inputs.q)
    # Validate all boundary masks (add dummy last dimension for checking)
    jax.tree.map(lambda f: self._check_function(f[..., None], x=inputs.x_inp), inputs.m)
    # Ensure masks are 3D boolean arrays (batch, space, boundary)
    assert jax.tree.all(jax.tree.map(lambda m: m.ndim == 3, inputs.m))
    assert jax.tree.all(jax.tree.map(lambda m: m.dtype == jnp.dtype(bool), inputs.m))

    # Delegate to subclass-specific implementation
    return self.call(inputs, **kwargs)

  def call(self, inputs: Inputs) -> Array:
    """Operator-specific forward pass. Override in subclasses.
    
    Args:
        inputs: Inputs NamedTuple
    
    Returns:
        Predicted function values
    """
    raise NotImplementedError

  @property
  def configs(self):
    """Get configuration parameters from class attributes.
    
    Returns:
        Dict mapping attribute names to their values (excluding 'parent')
    """
    # Extract all class annotations except the parent module reference
    configs = {
      attr: self.__getattr__(attr)
      for attr in self.__annotations__.keys() if attr != 'parent'
    }
    return configs

class LeadTimeConditionedNorm(nn.Module):
  """Learned normalization layer conditioned on an input parameter.
  
  Learns to shift and scale the distribution of normalized features based on
  a conditioning input (e.g., lead time tau). This allows the model to adapt
  feature normalization based on context.
  The correction_size must be either 1 (broadcast) or match one input dimension.
  
  Attributes:
      latent_size: Hidden dimension for MLPs computing scale and bias corrections
      correction_size: Output size of correction (1 for broadcast, or feature dim)
  """

  latent_size: Sequence[int]
  correction_size: int = 1

  def setup(self):
    # MLP for computing multiplicative correction (scale)
    self.mlp_scale = nn.Sequential(layers=[
      nn.Dense(self.latent_size, kernel_init=nn.initializers.normal(stddev=.01)),
      nn.sigmoid,
      nn.Dense(self.correction_size, kernel_init=nn.initializers.normal(stddev=.01)),
    ])
    # MLP for computing additive correction (bias)
    self.mlp_bias = nn.Sequential(layers=[
      nn.Dense(self.latent_size, kernel_init=nn.initializers.normal(stddev=.01)),
      nn.sigmoid,
      nn.Dense(self.correction_size, kernel_init=nn.initializers.normal(stddev=.01)),
    ])

  def __call__(self, c, x):
    """Apply learned correction to features based on conditioning input.
    
    Args:
        c: Conditioning input (e.g., lead time tau) [batch, *]
        x: Feature array to correct [batch, space, *]
    
    Returns:
        Corrected features with same shape as x
    """
    # Compute scale and bias corrections from conditioning input
    scale = 1 + c * self.mlp_scale(c)  # Additive correction centered at 1
    bias = c * self.mlp_bias(c)  # Additive bias
    # Reshape features to [batch, -1, features] for broadcasting
    shape = x.shape
    x = x.reshape(shape[0], -1, shape[-1])
    # Expand scale and bias to broadcast with feature dimension
    scale = jnp.expand_dims(scale, axis=1)
    bias = jnp.expand_dims(bias, axis=1)
    # Apply learned scaling and shifting
    x = x * scale + bias
    # Restore original shape
    x = x.reshape(*shape)
    return x

class FeedForward(nn.Module):
  """Multi-layer perceptron with optional normalization and dropout.
  
  Applies multiple dense or convolutional layers with non-linear activations.
  Optionally applies layer normalization and learned conditional normalization
  based on an input parameter. All arguments are concatenated and fed to the MLP.
  
  Attributes:
      layer_sizes: Sequence of output dimensions for each layer
      activation: Activation function applied after hidden layers
      use_layer_norm: Apply LayerNorm after final layer (default: False)
      use_conditional_norm: Apply learned normalization conditioned on input (default: False)
      cond_norm_hidden_size: Hidden dimension for conditional norm MLPs (default: 4)
      concatenate_axis: Axis along which to concatenate inputs (default: -1)
      conv: Use Conv layers instead of Dense layers (default: False)
      dropout: Dropout rate applied after each layer (default: None)
  """

  layer_sizes: Sequence[int]
  activation: Callable
  use_layer_norm: bool = False
  use_conditional_norm: bool = False
  cond_norm_hidden_size: int = 4
  concatenate_axis: int = -1
  conv: bool = False
  dropout: float = None

  def setup(self):
    # Set up layers (Dense or Conv)
    if not self.conv:
      self.layers = [nn.Dense(features) for features in self.layer_sizes]
    else:
      self.layers = [nn.Conv(features, kernel_size=1) for features in self.layer_sizes]
    # Set dropout layers (one for each layer)
    if self.dropout is not None:
      self.dropouts = [nn.Dropout(self.dropout) for _ in range(len(self.layers))]

    # Set up layer normalization applied after final layer
    self.layernorm = nn.LayerNorm(
      reduction_axes=-1,
      feature_axes=-1,
      use_scale=True,
      use_bias=True,
    ) if self.use_layer_norm else None

    # Set up learned conditional normalization layer
    self.correction = None
    if self.use_conditional_norm:
      self.correction = LeadTimeConditionedNorm(
        latent_size=self.cond_norm_hidden_size,
        correction_size=self.layer_sizes[-1],
      )

  def __call__(self, *args, c: Array = None, deterministic: bool = False, **kwargs):
    """Forward pass through the MLP.
    
    Args:
        *args: Positional arguments to concatenate
        c: Optional conditioning input for conditional normalization
        deterministic: If True, disable dropout
        **kwargs: Keyword arguments to concatenate
    
    Returns:
        Output feature tensor
    """
    # Concatenate all inputs
    x = concatenate_args(args=args, kwargs=kwargs, axis=self.concatenate_axis)
    # Apply hidden layers with activation and dropout
    for i, layer in enumerate(self.layers[:-1]):
      x = layer(x)
      x = self.activation(x)
      if self.dropout is not None:
        x = self.dropouts[i](x, deterministic=deterministic)
    # Apply final layer without activation
    x = self.layers[-1](x)
    if self.dropout is not None:
      x = self.dropouts[-1](x, deterministic=deterministic)
    # Apply optional layer normalization
    if self.layernorm:
      x = self.layernorm(x)
    # Apply optional learned conditional normalization
    if self.correction:
      assert c is not None, "Conditioning input required for conditional normalization"
      x = self.correction(c=c, x=x)
    return x

def concatenate_args(args, kwargs, axis: int = -1):
  """Concatenate positional and keyword arguments along given axis.
  
  Args:
      args: Tuple of positional arguments
      kwargs: Dict of keyword arguments
      axis: Axis along which to concatenate (default: -1)
  
  Returns:
      Concatenated array
  """
  # Flatten all arguments and keyword arguments into a single list
  combined_args = jax.tree.flatten(args)[0] + jax.tree.flatten(kwargs)[0]
  # Concatenate all tensors along specified axis
  concat_args = jnp.concatenate(combined_args, axis=axis)
  return concat_args

def segment_mean(arr, idx, num_segments: int, indices_are_sorted: bool = False):
  """Compute mean of array elements grouped by segment indices.
  
  Args:
      arr: Array to segment [N, *]
      idx: Segment index for each element [N, *]
      num_segments: Number of unique segment indices
      indices_are_sorted: If True, assumes indices are pre-sorted for efficiency (default: False)
  
  Returns:
      Segment means [num_segments, *]
  """
  # Sum array values within each segment
  sums = jax.ops.segment_sum(arr, idx, num_segments=num_segments, indices_are_sorted=indices_are_sorted)
  # Count elements in each segment
  counts = jax.ops.segment_sum(jnp.ones_like(arr), idx, num_segments=num_segments, indices_are_sorted=indices_are_sorted)
  # Avoid division by zero for empty segments
  counts = jnp.maximum(counts, 1.0)
  # Compute mean by dividing sum by count
  return sums / counts

def segment_softmax(arr, segment_ids, num_segments: int, indices_are_sorted: bool = False):
  """Compute softmax normalization grouped by segment (for attention).
  
  Computes exp(arr) / sum(exp(arr)) where the sum is taken per segment.
  This is useful for normalizing attention weights separately for each receiver node.
  
  Args:
      arr: Values to normalize [N, *]
      segment_ids: Segment index for each element [N, *]
      num_segments: Number of unique segment indices
      indices_are_sorted: If True, assumes indices are pre-sorted (default: False)
  
  Returns:
      Softmax-normalized values [N, *] (sum to 1 per segment)
  """
  # Find maximum per segment for numerical stability
  max_per_segment = jax.ops.segment_max(arr, segment_ids, num_segments=num_segments, indices_are_sorted=indices_are_sorted)
  # Expand max back to original shape for broadcasting
  max_expanded = max_per_segment[segment_ids]
  # Compute exp(arr - max) for numerical stability
  arr_exp = jnp.exp(arr - max_expanded)
  # Sum exp values per segment
  sum_per_segment = jax.ops.segment_sum(arr_exp, segment_ids, num_segments=num_segments, indices_are_sorted=indices_are_sorted)
  # Expand sum back to original shape for broadcasting
  sum_expanded = sum_per_segment[segment_ids]
  # Normalize by segment sum
  return arr_exp / sum_expanded

def segment_attention(q, k, v, segment_ids, num_segments: int = None, sorted: bool = False):
  """Compute segment-wise scaled dot-product attention.
  
  Computes attention as: Attention(Q, K, V) = softmax(QK^T/sqrt(d))V,
  with softmax and aggregation performed separately per segment.
  
  Args:
      q: Query array [N, H, D] where H=heads, D=head_dim
      k: Key array [N, H, D]
      v: Value array [N, H, D]
      segment_ids: Segment index for each element [N]
      num_segments: Number of unique segments (default: None)
      sorted: If True, assumes indices are pre-sorted (default: False)
  
  Returns:
      Attention output aggregated per segment [num_segments, H, D]
  """
  # Get head dimension for scaled attention
  head_dim = q.shape[-1]
  # Compute scaled similarity scores: QK^T / sqrt(d)
  sim = jnp.sum(q * k, axis=-1) * (head_dim ** -0.5)
  # Apply segment-wise softmax to normalize attention weights per segment
  attn = segment_softmax(arr=sim, segment_ids=segment_ids, num_segments=num_segments, indices_are_sorted=sorted)
  # Scale values by attention weights and aggregate per segment
  v_scaled = attn[..., None] * v
  # Sum scaled values within each segment
  out = jax.ops.segment_sum(v_scaled, segment_ids=segment_ids, num_segments=num_segments, indices_are_sorted=sorted)

  return out
