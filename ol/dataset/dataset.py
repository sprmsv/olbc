"""Classes for loading, processing, and manipulating the datasets."""

import h5py
from pathlib import Path
from typing import Union, Sequence, NamedTuple, Mapping, Tuple
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import PRNGKey

from ol.dataset.metadata import DATASET_METADATA
from ol.graph.graphbuilder import GraphMetadata, GraphBuilder
from ol.utils import Array


class DiscretizedFunction(NamedTuple):
  """Represents a discretized function on a spatial domain.
  
  Attributes:
      mask: Binary mask indicating valid/available values [batch, time, space]
      values: Function values on spatial domain [batch, time, space, variables]
  """
  mask: Array
  values: Array

  def __repr__(self):
    return f'{self.__class__.__name__}({self.values.shape})'

class Batch(NamedTuple):
  """A batch of samples containing coordinates, time, graphs, and function values.
  
  Attributes:
      x: Spatial coordinates [batch, time, space, dim]
      t: Time coordinates [batch, time, 1, 1] or None if time-independent
      g: GraphMetadata containing node/edge indices and spatial coordinates
      functions: Mapping of function names to DiscretizedFunction (mask + values)
  """
  x: Array
  t: Union[None, Array]
  g: GraphMetadata
  functions: Mapping[str, DiscretizedFunction]

  @property
  def shape(self) -> tuple:
    """Get batch shape [batch_size, time_steps, space_points, ...]"""
    return self.x.shape

  def shuffled(self, rngkey):
    """Returns a shuffled (space axis) version of the same batch.
    
    Randomly permutes spatial axis while updating all graph indices accordingly.
    
    Args:
        rngkey: JAX random key for permutation
    
    Returns:
        Batch with permuted spatial axis and updated graph indices
    """

    permutation = jax.random.permutation(rngkey, self.shape[2])
    permutation_g = jnp.concatenate([permutation, jnp.array([self.shape[2]])])  # dummy graph node included
    argsort_g = jnp.argsort(permutation_g)
    batch = Batch(
      x=self.x[:, :, permutation],
      t=self.t,
      g=GraphMetadata(
        x_pnodes_inp=self.g.x_pnodes_inp[:, permutation_g],
        x_pnodes_out=self.g.x_pnodes_out[:, permutation_g],
        x_rnodes=self.g.x_rnodes,
        r_rnodes=self.g.r_rnodes,
        p2r_edge_indices=jnp.stack([argsort_g[self.g.p2r_edge_indices[:, :, 0]], self.g.p2r_edge_indices[:, :, 1]], axis=-1),
        r2p_edge_indices=(jnp.stack([self.g.r2p_edge_indices[:, :, 0], argsort_g[self.g.r2p_edge_indices[:, :, 1]]], axis=-1) if (self.g.r2p_edge_indices is not None) else None),
        r2r_edge_indices=self.g.r2r_edge_indices,
        r2r_edge_domains=self.g.r2r_edge_domains,
      ),
      functions=jax.tree.map(lambda arr: arr[:, :, permutation], self.functions),
    )
    return batch

  def __len__(self) -> int:
    return self.shape[0]

class Stats(NamedTuple):
  """Statistics for normalization and denormalization of array data.
  
  Attributes:
      mean: Mean value for normalization [1, 1, 1, features]
      std: Standard deviation for normalization [1, 1, 1, features]
      min: Minimum value in domain [1, 1, 1, features]
      max: Maximum value in domain [1, 1, 1, features]
  """
  mean: Array = None
  std: Array = None
  min: Array = None
  max: Array = None

StatsCollectionType = Mapping[str, Union[Stats, Mapping[str, Stats]]]

class Dataset:
  """Loads and processes PDE datasets stored in HDF5 format.
  
  Manages dataset I/O, preprocessing (resampling, downsampling), normalization
  statistics, and graph metadata for neural operator training. Supports time-
  dependent and time-independent problems, with flexible function masking for
  boundary conditions.
  
  Attributes:
      metadata: Dataset metadata (shapes, functions, bounding boxes)
      file: HDF5 file handle for data access
      rigs: Pre-computed graph metadata for all samples (optional)
      stats: Normalization statistics (mean, std, min, max) for each function
      replace_nan: Strategy for NaN handling ('mean', 'maxabs', 'zero')
      space_downsample_factor: Spatial resampling factor
      time_downsample_factor: Temporal resampling factor (>=1)
      boundary_downsample_factor: Boundary condition masking factor
  """

  def __init__(self,
    name: str,
    dir: str,
    file: str,
    splits: Sequence[Tuple[int, int]] = None,
    time_cutoff_idx: int = None,
    time_downsample_factor: int = 1,
    space_downsample_factor: float = 1.,
    boundary_downsample_factor: float = 1.,
    preload: bool = False,
    replace_nan: str = 'mean',
    rngkey: PRNGKey = None,
  ):
    """Initialize a PDE dataset from HDF5 file.
    
    Args:
        name: Dataset name (e.g., 'poisson-circle-bc1')
        dir: Root directory containing dataset
        file: HDF5 filename within dataset directory
        splits: List of (start, end) indices for train/val/test splits
        time_cutoff_idx: Maximum time step index (truncates temporal axis)
        time_downsample_factor: Downsample temporal dimension by this factor
        space_downsample_factor: Downsample spatial dimension by this factor
        boundary_downsample_factor: Mask boundary values with this probability
        preload: If True, load entire dataset into memory (default: False)
        replace_nan: How to handle NaN values ('mean', 'maxabs', 'zero')
        rngkey: JAX random key for sampling operations
    """

    # Set attributes
    self.replace_nan = replace_nan
    self.rngkey = rngkey if (rngkey is not None) else jax.random.key(0)
    self.metadata = deepcopy(DATASET_METADATA[name])
    self.time_cutoff_idx = time_cutoff_idx
    self.time_downsample_factor = time_downsample_factor
    self.space_downsample_factor = space_downsample_factor
    self.boundary_downsample_factor = boundary_downsample_factor
    self.splits = splits

    # Set data attributes
    self.rigs: GraphMetadata = None
    self.file = h5py.File(Path(dir) / name / file, 'r')
    if self.splits == None: self.splits = [(0, self.file.attrs['count'][()].item())]
    if preload:
      self.reader = flatten_nested_dictionaries(load_h5py_group_as_dictionary(self.file, splits=self.splits))
      if self.metadata.fix: self.reader[self.metadata.x.path] = self.file[self.metadata.x.path]
      len_splits = [(self.splits[i][1] - self.splits[i][0]) for i in range(len(self.splits))]
      split_borders = ([0] + np.cumsum(len_splits).tolist())
      self.splits = [(split_borders[i], split_borders[i+1]) for i in range(len(self.splits))]
    else:
      self.reader = self.file

    # Load a sample
    self.sample = self._get_batch(idx=[0])

    # Change metadata based on the file
    self.metadata.shape = (self.file.attrs['count'], *self.metadata.shape[1:])

    if self.time_dependent:
      raise NotImplementedError

    # Instantiate the dataset stats
    self.stats = {}
    self.stats['x'] = Stats(
      min=np.array(self.metadata.bbox_x[0]).reshape(1, 1, 1, -1),
      max=np.array(self.metadata.bbox_x[1]).reshape(1, 1, 1, -1),
    )
    self.stats['t'] = Stats(
      min=np.array(self.metadata.bbox_t[0]).reshape(1, 1, 1, 1),
      max=np.array(self.metadata.bbox_t[1]).reshape(1, 1, 1, 1),
    ) if self.time_dependent else Stats()

  @property
  def time_dependent(self):
    """Check if dataset includes time-dependent functions."""
    return self.metadata.bbox_t is not None

  def compute_stats(self, split: int = 0, batch_size: int = 1, exclude_percentile: float = 0.0) -> None:
    """Compute and store normalization statistics (mean, std) for all functions.
    
    Args:
        split: Dataset split index to compute statistics for
        batch_size: Batch size for processing
        exclude_percentile: Optional percentile for outlier exclusion (0.0 = none)
    """

    # Check inputs
    if self.time_dependent:
      raise NotImplementedError

    # Compute statistics of functions
    stats = {key: {'sum': 0.0, 'sum_of_square': 0.0, 'count': 0} for key in self.metadata.functions.keys()}
    for batch in self.batches(split=split, batch_size=batch_size, get_graphs=False):
      for key in self.metadata.functions.keys():
        mask = batch.functions[key].mask
        values = batch.functions[key].values
        mask = np.tile(mask[..., None], reps=(1, 1, 1, values.shape[-1]))
        if exclude_percentile > 0:
          percentiles = jax.vmap(lambda msk, arr: jax.vmap(lambda m, a: jnp.percentile(a, q=(100-exclude_percentile*(m.sum()/m.shape[-1]))), in_axes=-1)(msk, arr), in_axes=0)(mask, jnp.abs(values))
          mask_val = mask & np.where(np.abs(values) <= percentiles[:, None, None, :], True, False)
        else:
          mask_val = mask
        stats[key]['sum'] += np.concatenate([np.sum(values[..., [idx_var]][np.where(mask_val[..., idx_var])], axis=0) for idx_var in range(values.shape[-1])]).reshape(1, 1, 1, -1)
        stats[key]['sum_of_square'] += np.concatenate([np.sum(values[..., [idx_var]][np.where(mask_val[..., idx_var])]**2, axis=0) for idx_var in range(values.shape[-1])]).reshape(1, 1, 1, -1)
        stats[key]['count'] += np.array([np.where(mask_val[..., idx_var])[0].shape[0] for idx_var in range(values.shape[-1])]).reshape(1, 1, 1, -1)

    # Aggregate the statistics
    for key in self.metadata.functions.keys():
      mean = stats[key]['sum'] / stats[key]['count']
      mean_of_square = stats[key]['sum_of_square'] / stats[key]['count']
      var = mean_of_square - mean**2
      var[var<0] = 0
      std = np.sqrt(var)
      self.stats[key] = Stats(mean=mean, std=std)

    # Write the statistics in the right structure
    self.stats = {
      'x': self.stats['x'],
      't': self.stats['t'],
      'geo': Stats(
        mean=jnp.concatenate([self.stats[key].mean for key in self.metadata.geo], axis=-1),
        std=jnp.concatenate([self.stats[key].std for key in self.metadata.geo], axis=-1),
      ),
      'dom': Stats(
        mean=jnp.concatenate([self.stats[key].mean for key in self.metadata.dom], axis=-1),
        std=jnp.concatenate([self.stats[key].std for key in self.metadata.dom], axis=-1),
      ),
      'seg': {key: self.stats[key] for key in self.metadata.seg},
      'out': Stats(
        mean=jnp.concatenate([self.stats[key].mean for key in self.metadata.out], axis=-1),
        std=jnp.concatenate([self.stats[key].std for key in self.metadata.out], axis=-1),
      ),
    }

  def build_graphs(self, builder: GraphBuilder, rmesh_correction_dsf: float = 1.0, batch_size: int = 1, rngkey: PRNGKey = None) -> None:
    """Builds and stores regional mesh graph metadata for all dataset samples.
    
    Constructs p2r, r2r, r2p edge sets and regional node positions for the
    entire dataset, padding all edge sets to maximum size.
    
    Args:
        builder: GraphBuilder instance for constructing mesh hierarchy
        rmesh_correction_dsf: Regional mesh correction downsampling factor
        batch_size: Number of samples to process per batch
        rngkey: JAX random key for stochastic graph construction
    """

    if rngkey is None:
      rngkey = jax.random.key(0)

    # Build graph metadata with potentially different number of edges
    metadata = []
    num_p2r_edges = 0
    num_r2r_edges = 0
    num_r2p_edges = 0
    if self.rigs is not None:
      # NOTE: Use the old number of edges in order to avoid re-compilation
      num_p2r_edges = self.rigs.p2r_edge_indices.shape[1]
      num_r2r_edges = self.rigs.r2r_edge_indices.shape[1]
      if self.rigs.r2p_edge_indices is not None:
        num_r2p_edges = self.rigs.r2p_edge_indices.shape[1]
    for split, split_ends in enumerate(self.splits):
      if not (split_ends[1] - split_ends[0]) > 0: continue
      for batch in self.batches(split=split, batch_size=batch_size, get_graphs=False):
        # Loop over all coordinates in the batch
        # NOTE: Assuming constant x in time
        for s in range(batch.shape[0]):
          x = batch.x[s, 0]
          z = batch.functions['sdfgrad'].values[s, 0]
          mask = np.any(np.stack([batch.functions[key].mask[s, 0] for key in self.metadata.seg]), axis=0)
          rngkey, subrngkey = jax.random.split(rngkey)
          m = builder.build_metadata(x_inp=x, x_out=x, x_bnd=x[mask], z_bnd=z[mask], bbox=np.array(self.metadata.bbox_x), rmesh_correction_dsf=rmesh_correction_dsf, rngkey=subrngkey)
          metadata.append(m)
          # Store the maximum number of edges
          if self.rigs is None:
            num_p2r_edges = max(num_p2r_edges, m.p2r_edge_indices.shape[1])
            num_r2r_edges = max(num_r2r_edges, m.r2r_edge_indices.shape[1])
            if m.r2p_edge_indices is not None:
              num_r2p_edges = max(num_r2p_edges, m.r2p_edge_indices.shape[1])
          # Break the loop if the coordinates are fixed on the batch axis
          if self.metadata.fix:
            break
        # Break the loop if the coordinates are fixed on the batch axis
        if self.metadata.fix:
          break
      if self.metadata.fix:
        break

    # Pad the edge sets using dummy nodes
    # NOTE: Exploiting jax' behavior for out-of-dimension indexing
    for i, m in enumerate(metadata):
      m: GraphMetadata
      metadata[i] = GraphMetadata(
        x_pnodes_inp=m.x_pnodes_inp,
        x_pnodes_out=m.x_pnodes_out,
        x_rnodes=m.x_rnodes,
        r_rnodes=m.r_rnodes,
        p2r_edge_indices=m.p2r_edge_indices[:, jnp.arange(num_p2r_edges), :],
        r2r_edge_indices=m.r2r_edge_indices[:, jnp.arange(num_r2r_edges), :],
        r2r_edge_domains=m.r2r_edge_domains[:, jnp.arange(num_r2r_edges), :],
        r2p_edge_indices=m.r2p_edge_indices[:, jnp.arange(num_r2p_edges), :] if (m.r2p_edge_indices is not None) else None,
      )

    # Concatenate all padded graph sets and store them
    self.rigs = jax.tree.map(lambda *v: jnp.concatenate(v), *metadata)

  def _get_sample(self, idx: int) -> Tuple[Array, Array, Mapping[str, DiscretizedFunction]]:
    """Fetches a single sample from the dataset and applies preprocessing.
    
    Loads coordinates and function values, applies spatial/temporal resampling,
    handles boundary masking, and processes NaN values.
    
    Args:
        idx: Global sample index
    
    Returns:
        Tuple of (coordinates, time, functions) where:
            - coordinates: [1, 1, space, dim] spatial coordinates
            - time: [1, time, 1, 1] or None if time-independent
            - functions: Mapping of name to DiscretizedFunction (mask + values)
    """

    # Load the coordinates
    x = np.stack([self.reader[self.metadata.x.path][idx, 0, _d] for _d in range(self.reader[self.metadata.x.path].shape[2])])
    x = x[None, None, :, :]
    # NOTE: Assuming the same coordinates at all times
    assert x.shape[1] == 1
    x = np.swapaxes(x, 2, 3)

    # Load the times
    if self.time_dependent:
      if self.metadata.t is not None:
        t = self.reader[self.metadata.t.path][idx]
        t = t[None, :, None, None]
      else:
        raise NotImplementedError
    else:
      t = None

    # Set the subsampling permutation (same permutation for all samples)
    permutation = jax.random.permutation(self.rngkey, x.shape[2])
    _x_size_original = x.shape[2]
    _x_size_after = int(self.metadata.shape[2] / self.space_downsample_factor)
    # Permute (and sub-sample) the coordinates
    x = subsample_array(x, permutation, size=_x_size_after, ax=2)
    # Set the kept boundary function values (same for all samples)
    boundary_keep = jax.random.uniform(self.rngkey, shape=x.shape[:3]) < (1 / self.boundary_downsample_factor)

    # Downsample the time axis
    if self.time_dependent:
      if self.time_cutoff_idx:
        t = t[:, :self.time_cutoff_idx]
      if self.time_downsample_factor > 1:
        t = t[:, ::self.time_downsample_factor]

    # Load the registered variables
    functions = {name: None for name in self.metadata.functions.keys()}
    for name, group in self.metadata.functions.items():
      # Get each array and index it if necessary
      arrays = []
      for arr in group.arrays:
        # NOTE: A 4-dimensional array [sample, time, channels, position] is expected
        array: Array = np.stack([np.stack([self.reader[arr.path][idx, _t, _c] for _c in arr.indices]) for _t in range(self.metadata.shape[1])])
        # Replace nan values
        if np.any(np.isnan(array)):
          if self.replace_nan == 'mean':
            fill_value = np.nanmean(array)
          elif self.replace_nan == 'maxabs':
            fill_value = np.nanmax(np.abs(array))
          elif self.replace_nan == 'zero':
            fill_value = 0.0
          array = np.nan_to_num(array, nan=fill_value)
        if len(array.shape) == 2:
          # NOTE: Handle arrays of arrays with variable sizes in space
          array = np.concatenate(array.flatten()).reshape(*array.shape, -1)
        array = array.swapaxes(-2, -1)
        arrays.append(array)
      # Concatenate channels together
      arrays = np.concatenate(arrays, axis=-1)
      # Load the x indices and create a mask accordingly
      mask = np.zeros(shape=(1, 1, _x_size_original), dtype=bool)
      values = np.zeros(shape=(1, 1, _x_size_original, arrays.shape[-1]), dtype=arrays.dtype)
      if group.x_indices is not None:
        x_indices = self.reader[group.x_indices][idx]
        # NOTE: Assuming the same coordinate indices at all times
        assert x_indices.shape[0] == 1
        assert x_indices.shape[1] == 1
        x_indices = x_indices[0, 0]
      else:
        x_indices = np.arange(_x_size_original)
      assert x_indices.shape[0] == arrays.shape[1], f'{x_indices.shape} and {arrays.shape}'
      mask[:, :, x_indices] = True
      values[:, :, x_indices, :] = arrays
      # Permute and subsample the coordinates
      mask = subsample_array(mask, permutation, size=_x_size_after, ax=2)
      boundary_keep = subsample_array(boundary_keep, permutation, size=_x_size_after, ax=2)  # redundant but kept for consistency
      values = subsample_array(values, permutation, size=_x_size_after, ax=2)
      # Mask the boundary functions
      if name in self.metadata.seg:
        mask = np.all(np.stack([mask, boundary_keep]), axis=0)
        values = values * np.tile(boundary_keep[..., None], reps=(1, 1, 1, values.shape[-1])).astype(float)
      # Downsample and cut the time axis
      if self.time_dependent:
        if self.time_cutoff_idx:
          mask = mask[:, :self.time_cutoff_idx]
          values = values[:, :self.time_cutoff_idx]
        if self.time_downsample_factor > 1:
          mask = mask[:, ::self.time_downsample_factor]
          values = values[:, ::self.time_downsample_factor]

      # Add the variable group
      functions[name] = DiscretizedFunction(mask=mask, values=values)

    return x, t, functions

  def _get_batch(self, idx: Sequence[int], get_graphs: bool = True) -> Batch:
    """Fetches multiple samples and returns as a single batch.
    
    Args:
        idx: Sequence of sample indices to fetch
        get_graphs: If True, include pre-computed graph metadata (default: True)
    
    Returns:
        Batch containing stacked samples and optionally graph metadata
    """

    # Instantiate the containers
    x = []
    t = [] if self.time_dependent else None
    functions: Mapping[str, Sequence[DiscretizedFunction]] = {name: [] for name in self.metadata.functions.keys()}
    # Get samples one by one
    for _idx in idx:
      _x, _t, _variables = self._get_sample(_idx)
      x.append(_x)
      if self.time_dependent: t.append(_t)
      for name in functions.keys():
        functions[name].append(_variables[name])
    for name in functions.keys():
      functions[name] = DiscretizedFunction(
        mask=np.concatenate([f.mask for f in functions[name]], axis=0),
        values=np.concatenate([f.values for f in functions[name]], axis=0),
      )

    # Stack all arrays
    x = np.concatenate(x, axis=0)
    if self.time_dependent: t = np.concatenate(t, axis=0)

    # Get graphs
    if (self.rigs is not None) and get_graphs:
      g = jax.tree.map(lambda v: v[idx], self.rigs)
    else:
      g = None

    batch = Batch(x=x, t=t, g=g, functions=functions)

    return batch

  def batches(self, split: int, batch_size: int, get_graphs: bool = True, rngkey: PRNGKey = None):
    """Generator yielding batches of samples from specified split.
    
    Args:
        split: Split index (0=train, 1=val, 2=test, etc.)
        batch_size: Number of samples per batch
        get_graphs: If True, include pre-computed graph metadata
        rngkey: Optional JAX random key for shuffling
    
    Yields:
        Batch objects of specified batch_size
    """
    split_length = self.splits[split][1] - self.splits[split][0]
    assert batch_size > 0
    assert batch_size <= split_length

    _idx_mode = self.splits[split][0] + np.arange(split_length)
    if rngkey is not None:
      _idx_mode = jax.random.permutation(rngkey, _idx_mode)

    len_dividable = split_length - (split_length % batch_size)
    for idx in np.split(_idx_mode[:len_dividable], len_dividable // batch_size):
      batch = self._get_batch(idx, get_graphs=get_graphs)
      yield batch

    if (split_length % batch_size):
      idx = _idx_mode[len_dividable:]
      batch = self._get_batch(idx, get_graphs=get_graphs)
      yield batch

  def __len__(self):
    return self.metadata.shape[0]

def subsample_array(arr, permutation, size, ax):
  """Subsample array along specified axis using permutation.
  
  Args:
      arr: Input array
      permutation: Permutation indices
      size: Number of elements to keep
      ax: Axis along which to subsample
  
  Returns:
      Subsampled array with 'size' elements along axis 'ax'
  """
  arr = np.swapaxes(arr, 0, ax)
  arr = arr[permutation]
  arr = arr[:size]
  arr = np.swapaxes(arr, 0, ax)
  return arr

def load_h5py_group_as_dictionary(group, splits):
  """Load HDF5 group data as nested dictionary with split slicing.
  
  Args:
      group: HDF5 group to load
      splits: List of (start, end) index tuples for slicing
  
  Returns:
      Nested dictionary with data loaded from splits
  """
  out = {
    key: (
      np.concatenate([group[key][slice(*split)] for split in splits], axis=0)
      if len(group[key].shape) > 0 else group[key]
    )
    if isinstance(group[key], h5py.Dataset)
    else load_h5py_group_as_dictionary(group[key], splits=splits)
    for key in group.keys()
  }
  return out

def flatten_nested_dictionaries(d: dict) -> dict:
  """Flatten a nested dictionary into a single-level dictionary with path keys.
  
  Args:
      d: Nested dictionary structure
  
  Returns:
      Flattened dictionary with keys as paths (e.g., 'group/subgroup/key')
  """
  out = {}
  for key, val in d.items():
    if isinstance(val, dict):
      for subkey, subval in flatten_nested_dictionaries(val).items():
        out['/'.join([key, subkey])] = subval
    else:
      out[key] = val

  return out
