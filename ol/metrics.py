from copy import copy
from dataclasses import dataclass
from typing import Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from ol.utils import Array, ScalarArray


EPSILON = 1e-10

@dataclass
class BatchMetrics:
    """Container for per-sample, per-variable metrics.
    
    Attributes:
        mse: Mean squared error [batch, var]
        l1: L1 error [batch, var]
        l2: L2 error [batch, var]
        recall: Recall score [batch, var]
        chamfer: Chamfer distance [batch, var]
    """
    mse: Array = None
    l1: Array = None
    l2: Array = None
    recall: Array = None
    chamfer: Array = None

    def map(self, f):
        for key in self.__dict__.keys():
            self.__setattr__(key, f(self.__getattribute__(key)))

    def reshape(self, shape):
        self.map(lambda m: m.reshape(shape))

    def __add__(self, obj):
      out = copy(self)
      for key in self.__dict__.keys():
        out.__setattr__(key, self.__getattribute__(key) + obj.__getattribute__(key))
      return out

@dataclass
class Metrics:
    """Container for scalar summary metrics.
    
    Attributes:
        mse: Mean squared error scalar
        l1: L1 error scalar
        l2: L2 error scalar
        recall: Recall score scalar
        chamfer: Chamfer distance scalar
    """
    mse: float = None
    l1: float = None
    l2: float = None
    recall: float = None
    chamfer: float = None

@dataclass
class EvalMetrics:
  """Container for evaluation metric statistics across samples.
  
  Attributes:
      median: Median values of all metrics across batch
      std: Standard deviation of metrics across batch
      maximum: Maximum values of metrics across batch
  """
  median: Metrics = None
  std: Metrics = None
  maximum: Metrics = None

  def to_dict(self):
      return {key: val.__dict__ for key, val in self.__dict__.items()}

def lp_norm(arr: Array, p: int = 2, chunks: Union[None, Sequence[int]] = None, num_chunks: int = None) -> Array:
    """Compute Bochner Lp-norm of array values over time and space.

    Args:
        arr: Point-wise values on a uniform grid [batch, time, space, var]
        p: Order of the norm (default: 2 for L2 norm). Common values: 1 (L1), 2 (L2), inf (max)
        chunks: Variable grouping indices for vector-valued functions.
            If None, each variable treated independently. Default: None
        num_chunks: Number of unique chunks. Required if chunks is provided.

    Returns:
        Lp-norm per sample and (optionally) per chunk [batch, num_chunks] or [batch]
    """

    # Set the default chunks
    if chunks is None:
        chunks = jnp.arange(arr.shape[-1])
        num_chunks = arr.shape[-1]
        keep_var_dim = False
    else:
        keep_var_dim = True

    # Compute power of absolute value
    pow_abs = jnp.power(jnp.abs(arr), p)
    # Sum on timespace (quadrature)
    abs_pow_sum_vars = jnp.sum(pow_abs, axis=(1, 2))
    # Sum on variable chunks
    abs_pow_sum = jax.vmap(jax.ops.segment_sum, in_axes=(0, None, None))(abs_pow_sum_vars, chunks, num_chunks)
    # Take the p-th root
    pth_root = jnp.power(abs_pow_sum, (1/p))
    # Squeeze variable axis
    if not keep_var_dim:
        pth_root = jnp.squeeze(pth_root, axis=-1)

    return pth_root

def rel_lp_error(gtr: Array, prd: Array, p: int = 2, chunks: Union[None, Sequence[int]] = None, num_chunks: int = None, exclude_percentile: float = 0.0) -> Array:
    """Compute relative Bochner Lp-error with optional outlier exclusion.

    Args:
        gtr: Ground truth function values [batch, time, space, var]
        prd: Predicted function values [batch, time, space, var]
        p: Order of the norm (default: 2). Common values: 1 (L1), 2 (L2), inf (max)
        chunks: Variable grouping indices for vector-valued functions. Default: None
        num_chunks: Number of unique chunks. Required if chunks is provided.
        exclude_percentile: Percentage of largest values (by magnitude) to exclude
            from error computation per sample per component (0-100). Default: 0.0

    Returns:
        Relative error per sample and variable [batch, var]
    """

    if chunks is None:
        chunks = jnp.arange(gtr.shape[-1])
        num_chunks = gtr.shape[-1]
    else:
        chunks = jnp.array(chunks)

    # Calculate the error
    err = (prd - gtr)
    # Exclude the largest values (in gtr) from the computations
    percentiles = jax.vmap(lambda arr: jax.vmap(lambda a: jnp.percentile(a, q=(100-exclude_percentile)), in_axes=-1)(arr), in_axes=0)(jnp.abs(gtr))
    err_corrected = jnp.where(jnp.abs(gtr) > percentiles[:, None, None, :], 0.0, err)
    gtr_corrected = jnp.where(jnp.abs(gtr) > percentiles[:, None, None, :], 0.0, gtr)
    # Calculate the norms
    err_norm = lp_norm(err_corrected, p=p, chunks=chunks, num_chunks=num_chunks)
    gtr_norm = lp_norm(gtr_corrected, p=p, chunks=chunks, num_chunks=num_chunks)

    return (err_norm / (gtr_norm + EPSILON))

def rel_lp_error_norm(gtr: Array, prd: Array, p: int = 2, chunks: Union[None, Sequence[int]] = None, num_chunks: int = None, exclude_percentile: float = 0.0) -> Array:
    """Compute vector norm of relative Lp-errors across variables.

    Args:
        gtr: Ground truth function values [batch, time, space, var]
        prd: Predicted function values [batch, time, space, var]
        p: Order of the norm (default: 2). Common values: 1 (L1), 2 (L2), inf (max)
        chunks: Variable grouping indices for vector-valued functions. Default: None
        num_chunks: Number of unique chunks. Required if chunks is provided.
        exclude_percentile: Percentage of largest values to exclude (0-100). Default: 0.0

    Returns:
        Vector norm of error across all variables per sample [batch]
    """

    err_per_var = rel_lp_error(gtr, prd, p=p, chunks=chunks, num_chunks=num_chunks, exclude_percentile=exclude_percentile)
    err_agg = jnp.linalg.norm(err_per_var, ord=p, axis=1)
    return err_agg

def rel_lp_error_mean(gtr: Array, prd: Array, p: int = 2, chunks: Union[None, Sequence[int]] = None, num_chunks: int = None, exclude_percentile: float = 0.0) -> Array:
    """Compute mean relative Lp-error across variables per sample.

    Args:
        gtr: Ground truth function values [batch, time, space, var]
        prd: Predicted function values [batch, time, space, var]
        p: Order of the norm (default: 2). Common values: 1 (L1), 2 (L2), inf (max)
        chunks: Variable grouping indices for vector-valued functions. Default: None
        num_chunks: Number of unique chunks. Required if chunks is provided.
        exclude_percentile: Percentage of largest values to exclude (0-100). Default: 0.0

    Returns:
        Mean relative error across variables per sample [batch]
    """

    err_per_var = rel_lp_error(gtr, prd, p=p, chunks=chunks, num_chunks=num_chunks, exclude_percentile=exclude_percentile)
    err_agg = jnp.mean(err_per_var, axis=1)
    return err_agg

def rel_lp_loss(gtr: Array, prd: Array, p: int = 2, q: float = 0.0) -> ScalarArray:
    """Compute scalar relative Lp-norm loss over entire batch.

    Args:
        gtr: Ground truth function values [batch, time, space, var]
        prd: Predicted function values [batch, time, space, var]
        p: Order of the norm (default: 2). Common values: 1 (L1), 2 (L2), inf (max)
        q: Percentage of largest values to exclude per sample per component (0-100). Default: 0.0

    Returns:
        Scalar loss: mean relative Lp-norm over entire batch
    """

    return jnp.mean(rel_lp_error_norm(gtr, prd, p=p, exclude_percentile=q))

def mse_error(gtr: Array, prd: Array) -> Array:
    """Compute mean squared error per variable.
    
    Args:
        gtr: Ground truth function values [batch, time, space, var]
        prd: Predicted function values [batch, time, space, var]
    
    Returns:
        MSE per sample [batch]
    """
    return jnp.mean(jnp.power(prd - gtr, 2), axis=(1, 2, 3))

def mse_loss(gtr: Array, prd: Array) -> ScalarArray:
    """Compute scalar mean squared error loss.
    
    Args:
        gtr: Ground truth function values [batch, time, space, var]
        prd: Predicted function values [batch, time, space, var]
    
    Returns:
        Scalar MSE loss (mean over all dimensions)
    """
    return jnp.mean(jnp.power(prd - gtr, 2))

def get_critical_values_mask(arr: Array, q: float) -> Array:
  """Identify locations of top q% values in magnitude.
  
  Args:
      arr: Input array [batch, time, space, var]
      q: Percentile threshold (0-100); top q% values are marked as critical
  
  Returns:
      Boolean mask [batch, time, space, var] indicating critical value locations
  """
  percentiles = jax.vmap(lambda arr: jax.vmap(lambda a: jnp.percentile(a, q=(100-q)), in_axes=-1)(arr), in_axes=0)(jnp.abs(arr))
  mask = jnp.where(jnp.abs(arr) > percentiles[:, None, None, :], True, False)
  return mask

def recall(gtr: Array, prd: Array, q: float) -> Array:
  """Compute recall of critical points between ground truth and prediction.
  
  Identifies top q% critical values in both arrays and computes the fraction
  of ground truth critical points that are also identified in the prediction.
  
  Args:
      gtr: Ground truth function values [batch, time, space, var]
      prd: Predicted function values [batch, time, space, var]
      q: Percentile threshold (0-100) for identifying critical values
  
  Returns:
      Recall score per variable [batch, var], values in [0, 1]
  """

  mask_gtr = get_critical_values_mask(gtr, q=q)
  mask_prd = get_critical_values_mask(prd, q=q)
  true_positive = (mask_gtr & mask_prd)
  score = true_positive.sum(axis=(1, 2)) / mask_gtr.sum(axis=(1, 2))
  return score

def recall_mean(gtr: Array, prd: Array, q: float) -> Array:
  """Mean recall score per sample (averaged over variables).
  
  Args:
      gtr: Ground truth function values [batch, time, space, var]
      prd: Predicted function values [batch, time, space, var]
      q: Percentile threshold (0-100) for identifying critical values
  
  Returns:
      Mean recall per sample [batch]
  """
  score_per_var = recall(gtr, prd, q=q)
  score_agg = jnp.mean(score_per_var, axis=1)
  return score_agg

def recall_loss(gtr: Array, prd: Array, q: float = 0.1) -> ScalarArray:
  """Scalar recall loss: 1 - mean recall.
  
  Args:
      gtr: Ground truth function values [batch, time, space, var]
      prd: Predicted function values [batch, time, space, var]
      q: Percentile threshold (0-100) for identifying critical values
  
  Returns:
      Scalar loss value
  """
  return 1 - jnp.mean(recall(gtr, prd, q=q))

def iou(gtr: Array, prd: Array, q: float) -> Array:
  """Compute Intersection over Union (IoU) of critical points.
  
  Args:
      gtr: Ground truth function values [batch, time, space, var]
      prd: Predicted function values [batch, time, space, var]
      q: Percentile threshold (0-100) for identifying critical values
  
  Returns:
      IoU score per variable [batch, var], values in [0, 1]
  """

  mask_gtr = get_critical_values_mask(gtr, q=q)
  mask_prd = get_critical_values_mask(prd, q=q)
  intersection = (mask_gtr & mask_prd)
  union = (mask_gtr | mask_prd)
  score = intersection.sum(axis=(1, 2)) / union.sum(axis=(1, 2))
  return score

def _chamfer_distance_single_instance(x, m_gtr, m_prd, size) -> ScalarArray:
  """Compute one-directional Chamfer distance between masked regions.
  
  Measures average distance from ground truth critical points to nearest
  predicted critical point (one-directional, accounts for false negatives).
  
  Args:
      x: Space coordinates [space, dim]
      m_gtr: Ground truth critical value mask [space,]
      m_prd: Predicted critical value mask [space,]
      size: Expected size of masked region (for extracting fixed-size sets)
  
  Returns:
      Scalar Chamfer distance value
  """

  x_gtr = x[jnp.where(m_gtr, size=size)]
  x_prd = x[jnp.where(m_prd, size=size)]
  d = jnp.linalg.norm(x_gtr[:, None, :] - x_prd[None, :, :], axis=-1)
  return jnp.min(d, axis=1).mean()

def chamfer(x, u_gtr, u_prd, q):
  """Compute Chamfer distance between ground-truth and prediction critical points.
  
  Args:
      x: Spatial coordinates [batch, space, dim]
      u_gtr: Ground truth function values [batch, time, space, var]
      u_prd: Predicted function values [batch, time, space, var]
      q: Percentile threshold (0-100) for identifying critical values
  
  Returns:
      Chamfer distance per sample and variable [batch, var]
  """
  mask_gtr = get_critical_values_mask(u_gtr, q=q)
  mask_prd = get_critical_values_mask(u_prd, q=q)
  size = np.ceil(q / 100 * u_gtr.shape[2]).astype(int)
  _f_per_var = jax.vmap(_chamfer_distance_single_instance, in_axes=(None, 1, 1, None))
  _f_per_time_var = jax.vmap(_f_per_var, in_axes=(0, 0, 0, None))
  _f_per_sample_time_var = jax.vmap(_f_per_time_var, in_axes=(0, 0, 0, None))
  score = _f_per_sample_time_var(x, mask_gtr, mask_prd, size)

  return score[:, 0, :]

def chamfer_mean(x, u_gtr, u_prd, q):
  """Mean Chamfer distance per sample (averaged over variables).
  
  Args:
      x: Spatial coordinates [batch, space, dim]
      u_gtr: Ground truth function values [batch, time, space, var]
      u_prd: Predicted function values [batch, time, space, var]
      q: Percentile threshold (0-100) for identifying critical values
  
  Returns:
      Mean Chamfer distance per sample [batch]
  """
  return jnp.mean(chamfer(x, u_gtr, u_prd, q), axis=-1)

def chamfer_loss(x, u_gtr, u_prd, q):
   """Scalar Chamfer loss (mean over batch and variables).
   
   Args:
       x: Spatial coordinates [batch, space, dim]
       u_gtr: Ground truth function values [batch, time, space, var]
       u_prd: Predicted function values [batch, time, space, var]
       q: Percentile threshold (0-100) for identifying critical values
   
   Returns:
       Scalar loss value
   """
   return jnp.mean(chamfer(x, u_gtr, u_prd, q))

def _recall_tol_single_instance(x, m_gtr, m_prd, tol, size) -> ScalarArray:
  """Compute recall at tolerance threshold for masked regions.
  
  Counts how many ground truth critical points have a predicted critical point
  within distance tolerance.
  
  Args:
      x: Space coordinates [space, dim]
      m_gtr: Ground truth critical value mask [space,]
      m_prd: Predicted critical value mask [space,]
      tol: Distance tolerance threshold
      size: Expected size of masked region
  
  Returns:
      Scalar recall score at tolerance
  """

  x_gtr = x[jnp.where(m_gtr, size=size)]
  x_prd = x[jnp.where(m_prd, size=size)]
  d = jnp.linalg.norm(x_gtr[:, None, :] - x_prd[None, :, :], axis=-1)
  minimum_positive_distance = jnp.min(d, axis=1)
  true_positive = minimum_positive_distance <= tol
  score = true_positive.sum(axis=0) / size
  return score

def recall_tol(x, u_gtr, u_prd, q, tol):
  """Compute recall at tolerance between ground truth and prediction critical points.
  
  Args:
      x: Spatial coordinates [batch, space, dim]
      u_gtr: Ground truth function values [batch, time, space, var]
      u_prd: Predicted function values [batch, time, space, var]
      q: Percentile threshold (0-100) for identifying critical values
      tol: Distance tolerance for matching critical points
  
  Returns:
      Recall at tolerance per sample and variable [batch, var]
  """
  mask_gtr = get_critical_values_mask(u_gtr, q=q)
  mask_prd = get_critical_values_mask(u_prd, q=q)
  size = np.ceil(q / 100 * u_gtr.shape[2]).astype(int)
  _f_per_var = jax.vmap(_recall_tol_single_instance, in_axes=(None, 1, 1, None, None))
  _f_per_time_var = jax.vmap(_f_per_var, in_axes=(0, 0, 0, None, None))
  _f_per_sample_time_var = jax.vmap(_f_per_time_var, in_axes=(0, 0, 0, None, None))
  score = _f_per_sample_time_var(x, mask_gtr, mask_prd, tol, size)

  return score[:, 0, :]

def recall_tol_mean(x, u_gtr, u_prd, q, tol):
  """Mean recall at tolerance per sample (averaged over variables).
  
  Args:
      x: Spatial coordinates [batch, space, dim]
      u_gtr: Ground truth function values [batch, time, space, var]
      u_prd: Predicted function values [batch, time, space, var]
      q: Percentile threshold (0-100) for identifying critical values
      tol: Distance tolerance for matching critical points
  
  Returns:
      Mean recall at tolerance per sample [batch]
  """
  return jnp.mean(recall_tol(x, u_gtr, u_prd, q, tol), axis=-1)

def recall_tol_loss(x, u_gtr, u_prd, q, tol):
   """Scalar recall loss at tolerance (mean over batch and variables).
   
   Args:
       x: Spatial coordinates [batch, space, dim]
       u_gtr: Ground truth function values [batch, time, space, var]
       u_prd: Predicted function values [batch, time, space, var]
       q: Percentile threshold (0-100) for identifying critical values
       tol: Distance tolerance for matching critical points
   
   Returns:
       Scalar loss value
   """
   return jnp.mean(recall_tol(x, u_gtr, u_prd, q, tol))
