"""General utilities for logging, array operations, and profiling.

This module provides helper functions used throughout the codebase.
"""

from datetime import datetime
from time import time
from typing import Union, Sequence, Callable, Any

import flax.typing
import jax
import jax.numpy as jnp
import numpy as np


Array = Union[jnp.ndarray, np.ndarray]
ScalarArray = Union[jnp.ndarray, np.ndarray]
Tree = Any
_LOG_START_TIME = time()

def _format_runtime(seconds_elapsed: float) -> str:
  """Format elapsed seconds as human-readable duration string.
  
  Args:
      seconds_elapsed: Total seconds elapsed as float
  
  Returns:
      Formatted string as 'dd-hh:mm:ss'
  """
  # Convert to integer and decompose into days, hours, minutes, seconds
  total_seconds = int(seconds_elapsed)
  days, rem = divmod(total_seconds, 24 * 60 * 60)
  hours, rem = divmod(rem, 60 * 60)
  minutes, seconds = divmod(rem, 60)
  return f'{days:02d}-{hours:02d}:{minutes:02d}:{seconds:02d}'

def log(message: Any, flush: bool = True):
  """Print timestamped log message with runtime duration.
  
  Args:
      message: Message to log (can be any type, will be converted to string)
      flush: Whether to flush output buffer immediately (default: True)
  """
  # Get current datetime and elapsed runtime
  daytime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  runtime = _format_runtime(time() - _LOG_START_TIME)
  # Print formatted message with timestamp and elapsed time
  print(f'{daytime} | {runtime} | {message}', flush=flush)

def profile(f: Callable, kwargs: dict, repeats: int = 1, block_until_ready: bool = False):
  """Measure average execution time of a function.
  
  Args:
      f: Callable function to profile
      kwargs: Dict of keyword arguments to pass to f
      repeats: Number of times to execute function (default: 1)
      block_until_ready: If True, wait for JAX computation to complete before measuring (default: False)
  
  Returns:
      Average execution time in seconds
  """
  # Record start time and execute function 'repeats' times
  t_0 = time()
  for _ in range(repeats):
    u = f(**kwargs)
  # Optionally wait for JAX GPU/TPU operations to complete
  if block_until_ready:
    u = u.block_until_ready()
  # Return average time per execution
  return ((time() - t_0) / repeats)

def shuffle_arrays(rngkey: flax.typing.PRNGKey, arrays: Sequence[Array], axis: int = 0) -> Sequence[Array]:
  """Shuffle multiple arrays with the same random permutation.
  
  Args:
      rngkey: JAX random key for reproducibility
      arrays: Sequence of arrays to shuffle together
      axis: Axis along which to apply permutation (default: 0)
  
  Returns:
      Tuple of shuffled arrays in same order
  """
  # Move the desired axis to leading position for easier indexing
  arrays = jax.tree.map(lambda v: jnp.moveaxis(v, axis, 0), arrays)

  # Generate a random permutation matching the size of the axis
  length = arrays[0].shape[0]
  assert all(jax.tree.map(lambda v: v.shape[0] == length, arrays))
  permutation = jax.random.permutation(rngkey, length)

  # Apply the same permutation to all arrays
  arrays = jax.tree.map(lambda v: v[permutation], arrays)
  # Move axis back to original position
  arrays = jax.tree.map(lambda v: jnp.moveaxis(v, 0, axis), arrays)

  return arrays

def normalize(arr: Array, shift: Array, scale: Array):
  """Normalize array using standardization: (arr - shift) / scale.
  
  Args:
      arr: Array to normalize
      shift: Mean/offset to subtract
      scale: Standard deviation or scaling factor
  
  Returns:
      Normalized array with same shape as input
  """
  # Avoid division by zero by replacing zero scales with 1
  scale = jnp.where(scale == 0., 1., scale)
  # Apply standardization formula
  arr = (arr - shift) / scale
  return arr

def unnormalize(arr: Array, mean: Array, std: Array):
  """Reverse standardization to original scale: std * arr + mean.
  
  Args:
      arr: Normalized array
      mean: Original mean to add back
      std: Original standard deviation to scale by
  
  Returns:
      Denormalized array in original value range
  """
  # Reverse the normalization formula
  arr = std * arr + mean
  return arr

def segment_mean(arr: Array, chunks: Sequence, axis: int = 0):
  """Compute mean of array segments identified by chunk indices.
  
  Uses np.add.reduceat for efficient grouped reduction. Elements with the same
  chunk index are grouped together and their mean is computed.
  
  Args:
      arr: Array to segment
      chunks: Integer array of chunk assignments (one per element along axis)
      axis: Axis along which to apply segmentation (default: 0)
  
  Returns:
      Array with shape [..., num_chunks, ...] containing mean of each chunk
  """
  # Convert to numpy for efficient reduction operations
  arr = np.array(arr)
  chunks = np.array(chunks)
  assert len(chunks.shape) == 1
  assert chunks.shape[0] == arr.shape[axis]

  # Swap target axis to front for processing
  arr = arr.swapaxes(0, axis)
  # Sort array and chunks by chunk indices to group same chunks together
  argsort = np.argsort(chunks)
  chunks = chunks[argsort]
  arr = arr[argsort]

  # Find indices where chunk ID changes to identify segment boundaries
  steps = np.where(chunks[1:] - chunks[:-1])[0] + 1
  steps = np.concatenate([[0], steps])  # Include start of first segment

  # Use reduceat to efficiently sum segments and count elements
  reduced = np.add.reduceat(arr, indices=steps, axis=0)
  sizes = np.add.reduceat(np.ones_like(arr), indices=steps, axis=0)
  # Compute mean by dividing sum by count
  out = reduced / sizes

  # Restore original axis ordering
  out = out.swapaxes(0, axis)

  return out
