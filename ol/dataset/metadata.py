"""Metadata of the datasets."""

from dataclasses import dataclass
from typing import Sequence, Tuple, Mapping


@dataclass
class ArrayMetadata:
  """Metadata for a single array in the HDF5 dataset.
  
  Attributes:
      path: Path to array in HDF5 file
      indices: Indices of array channels to use (None = use all)
      names: Display names for each channel
      signed: Whether each channel is signed (can be negative)
  """
  path: str
  indices: Sequence[int] = None
  names: Sequence[str] = None
  signed: Sequence[str] = None

@dataclass
class ArrayGroup:
  """Group of related arrays forming a single dataset variable.
  
  Attributes:
      arrays: List of ArrayMetadata objects in this group
      x_indices: Path to spatial indices array (for irregular point sets)
  """
  arrays: Sequence[ArrayMetadata]
  x_indices: str = None

  @property
  def signed(self):
    """Concatenated signed flags from all arrays in group."""
    out = []
    for arr in self.arrays:
      for idx in arr.indices:
        out.append(arr.signed[idx])
    return out

  @property
  def names(self):
    """Concatenated channel names from all arrays in group."""
    out = []
    for arr in self.arrays:
      for idx in arr.indices:
        out.append(arr.names[idx])
    return out

@dataclass
class Metadata:
  """Complete metadata for a PDE dataset.
  
  Attributes:
      periodic: Whether domain has periodic boundary conditions
      fix: Whether spatial coordinates are fixed (same for all samples)
      shape: Data shape [num_samples, time_steps, space_points, variables]
      boundary_size: Number of boundary nodes
      x: Spatial coordinate array metadata
      t: Time coordinate array metadata (None if time-independent)
      bbox_x: Bounding box for spatial domain ((xmin, ymin), (xmax, ymax))
      bbox_t: Bounding box for time domain (tmin, tmax)
      functions: Mapping of function names to ArrayGroup definitions
      geo: List of geometric function names (e.g., 'sdf', 'sdfgrad')
      dom: List of domain function names
      ext: List of extension function names
      seg: List of boundary segment/condition names
      out: List of output variable names
  """

  periodic: bool
  fix: bool
  shape: Tuple[int, int, int, int]
  boundary_size: int
  x: ArrayMetadata
  t: ArrayMetadata
  bbox_x: Tuple[Tuple[float, float], Tuple[float, float]]
  bbox_t: Tuple[float, float]
  functions: Mapping[str, ArrayGroup]
  geo: Sequence[str]
  dom: Sequence[str]
  ext: Sequence[str]
  seg: Sequence[str]
  out: Sequence[str]

DATASET_METADATA = {
  'poisson-boomerang-bc1': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 15187, -1),
    boundary_size=1160,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'solution': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'source': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=['source'],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g'],
    seg=['bc-0-dir'],
    out=['solution'],
  ),
  'poisson-boomerang-bc4': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 15187, -1),
    boundary_size=1160,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'solution': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'source': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-0-rob': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/robin/g', indices=[0], names=['$g_R$'], signed=[True]),
          ArrayMetadata(path='boundaries/0/robin/alpha', indices=[0], names=['$\\alpha_R$'], signed=[True]),
        ],
        x_indices='boundaries/0/robin/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=['source'],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-0-rob'],
    out=['solution'],
  ),
  'poisson-boomerang-bc5': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 15187, -1),
    boundary_size=1160,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'solution': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'source': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-0-rob': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/robin/g', indices=[0], names=['$g_R$'], signed=[True]),
          ArrayMetadata(path='boundaries/0/robin/alpha', indices=[0], names=['$\\alpha_R$'], signed=[True]),
        ],
        x_indices='boundaries/0/robin/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=['source'],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-0-rob'],
    out=['solution'],
  ),
  'poisson-circle-bc1': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 16859, -1),
    boundary_size=1004,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'solution': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'source': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=['source'],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g'],
    seg=['bc-0-dir'],
    out=['solution'],
  ),
  'poisson-circle-bc4': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 16859, -1),
    boundary_size=1004,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'solution': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'source': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-0-rob': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/robin/g', indices=[0], names=['$g_R$'], signed=[True]),
          ArrayMetadata(path='boundaries/0/robin/alpha', indices=[0], names=['$\\alpha_R$'], signed=[True]),
        ],
        x_indices='boundaries/0/robin/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=['source'],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-0-rob'],
    out=['solution'],
  ),
  'poisson-circle-bc5': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 16859, -1),
    boundary_size=1004,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'solution': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'source': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-0-rob': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/robin/g', indices=[0], names=['$g_R$'], signed=[True]),
          ArrayMetadata(path='boundaries/0/robin/alpha', indices=[0], names=['$\\alpha_R$'], signed=[True]),
        ],
        x_indices='boundaries/0/robin/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=['source'],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-0-rob'],
    out=['solution'],
  ),
  'poisson-square-bc1': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 16598, -1),
    boundary_size=1120,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'solution': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'source': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=['source'],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g'],
    seg=['bc-0-dir'],
    out=['solution'],
  ),
  'poisson-square-bc4': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 16598, -1),
    boundary_size=1120,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'solution': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'source': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-0-rob': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/robin/g', indices=[0], names=['$g_R$'], signed=[True]),
          ArrayMetadata(path='boundaries/0/robin/alpha', indices=[0], names=['$\\alpha_R$'], signed=[True]),
        ],
        x_indices='boundaries/0/robin/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=['source'],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-0-rob'],
    out=['solution'],
  ),
  'poisson-square-bc5': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 16598, -1),
    boundary_size=1120,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'solution': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'source': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-0-rob': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/robin/g', indices=[0], names=['$g_R$'], signed=[True]),
          ArrayMetadata(path='boundaries/0/robin/alpha', indices=[0], names=['$\\alpha_R$'], signed=[True]),
        ],
        x_indices='boundaries/0/robin/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=['source'],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-0-rob'],
    out=['solution'],
  ),
  'poisson-squareboom-bc1': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 20798, -1),
    boundary_size=1824,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'solution': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'source': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=['source'],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g'],
    seg=['bc-0-dir'],
    out=['solution'],
  ),
  'poisson-squareboom-bc4': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 20798, -1),
    boundary_size=1824,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'solution': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'source': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-0-rob': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/robin/g', indices=[0], names=['$g_R$'], signed=[True]),
          ArrayMetadata(path='boundaries/0/robin/alpha', indices=[0], names=['$\\alpha_R$'], signed=[True]),
        ],
        x_indices='boundaries/0/robin/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=['source'],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-0-rob'],
    out=['solution'],
  ),
  'poisson-squareboom-bc5': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 20798, -1),
    boundary_size=1824,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'solution': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'source': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-0-rob': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/robin/g', indices=[0], names=['$g_R$'], signed=[True]),
          ArrayMetadata(path='boundaries/0/robin/alpha', indices=[0], names=['$\\alpha_R$'], signed=[True]),
        ],
        x_indices='boundaries/0/robin/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=['source'],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-0-rob'],
    out=['solution'],
  ),
  'elasticity-squarehollow-m1': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 15074, -1),
    boundary_size=1614,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'displacement': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0, 1], names=['$u_x$', '$u_y$'], signed=[True, True])],
        x_indices=None,
      ),
      'stress': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/cauchystress', indices=[0, 1, 3], names=['$\\sigma_{xx}$', '$\\sigma_{xy}$', '$\\sigma_{yx}$', '$\\sigma_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'strain': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/strain', indices=[0, 1, 3], names=['$E_{xx}$', '$E_{xy}$', '$E_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/alpha', indices=[0], names=['$\\alpha_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/beta', indices=[0], names=['$\\beta_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/g', indices=[0], names=['$\\g_1$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-1-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/1/dirichlet/indices',
      ),
      'bc-1-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/1/neumann/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=[],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g', 'ext-1-alpha', 'ext-1-beta', 'ext-1-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-1-dir', 'bc-1-neu'],
    out=['displacement', 'strain', 'stress'],
  ),
  'elasticity-squarehollow-m2': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 15074, -1),
    boundary_size=1614,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'displacement': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0, 1], names=['$u_x$', '$u_y$'], signed=[True, True])],
        x_indices=None,
      ),
      'stress': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/cauchystress', indices=[0, 1, 3], names=['$\\sigma_{xx}$', '$\\sigma_{xy}$', '$\\sigma_{yx}$', '$\\sigma_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'strain': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/strain', indices=[0, 1, 3], names=['$E_{xx}$', '$E_{xy}$', '$E_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/alpha', indices=[0], names=['$\\alpha_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/beta', indices=[0], names=['$\\beta_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/g', indices=[0], names=['$\\g_1$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-1-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/1/dirichlet/indices',
      ),
      'bc-1-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/1/neumann/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=[],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g', 'ext-1-alpha', 'ext-1-beta', 'ext-1-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-1-dir', 'bc-1-neu'],
    out=['displacement', 'strain', 'stress'],
  ),
  'elasticity-squarehollow-m3': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 15074, -1),
    boundary_size=1614,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'displacement': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0, 1], names=['$u_x$', '$u_y$'], signed=[True, True])],
        x_indices=None,
      ),
      'stress': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/cauchystress', indices=[0, 1, 3], names=['$\\sigma_{xx}$', '$\\sigma_{xy}$', '$\\sigma_{yx}$', '$\\sigma_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'strain': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/strain', indices=[0, 1, 3], names=['$E_{xx}$', '$E_{xy}$', '$E_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/alpha', indices=[0], names=['$\\alpha_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/beta', indices=[0], names=['$\\beta_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/g', indices=[0], names=['$\\g_1$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-1-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/1/dirichlet/indices',
      ),
      'bc-1-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/1/neumann/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=[],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g', 'ext-1-alpha', 'ext-1-beta', 'ext-1-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-1-dir', 'bc-1-neu'],
    out=['displacement', 'strain', 'stress'],
  ),
  'elasticity-circlehollow-m1': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 11786, -1),
    boundary_size=1310,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'displacement': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0, 1], names=['$u_x$', '$u_y$'], signed=[True, True])],
        x_indices=None,
      ),
      'stress': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/cauchystress', indices=[0, 1, 3], names=['$\\sigma_{xx}$', '$\\sigma_{xy}$', '$\\sigma_{yx}$', '$\\sigma_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'strain': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/strain', indices=[0, 1, 3], names=['$E_{xx}$', '$E_{xy}$', '$E_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/alpha', indices=[0], names=['$\\alpha_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/beta', indices=[0], names=['$\\beta_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/g', indices=[0], names=['$\\g_1$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-1-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/1/dirichlet/indices',
      ),
      'bc-1-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/1/neumann/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=[],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g', 'ext-1-alpha', 'ext-1-beta', 'ext-1-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-1-dir', 'bc-1-neu'],
    out=['displacement', 'strain', 'stress'],
  ),
  'elasticity-circlehollow-m2': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 11786, -1),
    boundary_size=1310,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'displacement': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0, 1], names=['$u_x$', '$u_y$'], signed=[True, True])],
        x_indices=None,
      ),
      'stress': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/cauchystress', indices=[0, 1, 3], names=['$\\sigma_{xx}$', '$\\sigma_{xy}$', '$\\sigma_{yx}$', '$\\sigma_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'strain': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/strain', indices=[0, 1, 3], names=['$E_{xx}$', '$E_{xy}$', '$E_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/alpha', indices=[0], names=['$\\alpha_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/beta', indices=[0], names=['$\\beta_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/g', indices=[0], names=['$\\g_1$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-1-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/1/dirichlet/indices',
      ),
      'bc-1-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/1/neumann/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=[],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g', 'ext-1-alpha', 'ext-1-beta', 'ext-1-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-1-dir', 'bc-1-neu'],
    out=['displacement', 'strain', 'stress'],
  ),
  'elasticity-circlehollow-m3': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 11786, -1),
    boundary_size=1310,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'displacement': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0, 1], names=['$u_x$', '$u_y$'], signed=[True, True])],
        x_indices=None,
      ),
      'stress': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/cauchystress', indices=[0, 1, 3], names=['$\\sigma_{xx}$', '$\\sigma_{xy}$', '$\\sigma_{yx}$', '$\\sigma_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'strain': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/strain', indices=[0, 1, 3], names=['$E_{xx}$', '$E_{xy}$', '$E_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/alpha', indices=[0], names=['$\\alpha_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/beta', indices=[0], names=['$\\beta_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/g', indices=[0], names=['$\\g_1$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-1-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/1/dirichlet/indices',
      ),
      'bc-1-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/1/neumann/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=[],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g', 'ext-1-alpha', 'ext-1-beta', 'ext-1-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-1-dir', 'bc-1-neu'],
    out=['displacement', 'strain', 'stress'],
  ),
  'elasticity-boomcircletri-m1': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 17662, -1),
    boundary_size=1652,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'displacement': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0, 1], names=['$u_x$', '$u_y$'], signed=[True, True])],
        x_indices=None,
      ),
      'stress': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/cauchystress', indices=[0, 1, 3], names=['$\\sigma_{xx}$', '$\\sigma_{xy}$', '$\\sigma_{yx}$', '$\\sigma_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'strain': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/strain', indices=[0, 1, 3], names=['$E_{xx}$', '$E_{xy}$', '$E_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/alpha', indices=[0], names=['$\\alpha_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/beta', indices=[0], names=['$\\beta_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/g', indices=[0], names=['$\\g_1$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-1-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/1/dirichlet/indices',
      ),
      'bc-1-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/1/neumann/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=[],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g', 'ext-1-alpha', 'ext-1-beta', 'ext-1-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-1-dir', 'bc-1-neu'],
    out=['displacement', 'strain', 'stress'],
  ),
  'elasticity-boomcircletri-m2': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 17662, -1),
    boundary_size=1652,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'displacement': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0, 1], names=['$u_x$', '$u_y$'], signed=[True, True])],
        x_indices=None,
      ),
      'stress': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/cauchystress', indices=[0, 1, 3], names=['$\\sigma_{xx}$', '$\\sigma_{xy}$', '$\\sigma_{yx}$', '$\\sigma_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'strain': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/strain', indices=[0, 1, 3], names=['$E_{xx}$', '$E_{xy}$', '$E_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/alpha', indices=[0], names=['$\\alpha_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/beta', indices=[0], names=['$\\beta_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/g', indices=[0], names=['$\\g_1$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-1-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/1/dirichlet/indices',
      ),
      'bc-1-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/1/neumann/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=[],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g', 'ext-1-alpha', 'ext-1-beta', 'ext-1-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-1-dir', 'bc-1-neu'],
    out=['displacement', 'strain', 'stress'],
  ),
  'elasticity-boomcircletri-m3': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 17662, -1),
    boundary_size=1652,
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'sdf': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[False])],
        x_indices=None,
      ),
      'sdfgrad': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/sdf_grad', indices=[0, 1], names=['$\\nabla$ SDF', '$\\nabla$ SDF'], signed=[True, True])],
        x_indices=None,
      ),
      'displacement': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0, 1], names=['$u_x$', '$u_y$'], signed=[True, True])],
        x_indices=None,
      ),
      'stress': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/cauchystress', indices=[0, 1, 3], names=['$\\sigma_{xx}$', '$\\sigma_{xy}$', '$\\sigma_{yx}$', '$\\sigma_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'strain': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/strain', indices=[0, 1, 3], names=['$E_{xx}$', '$E_{xy}$', '$E_{yy}$'], signed=[True, True, True, True])],
        x_indices=None,
      ),
      'ext-0-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/alpha', indices=[0], names=['$\\alpha_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/beta', indices=[0], names=['$\\beta_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-0-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/0/g', indices=[0], names=['$\\g_0$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-alpha': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/alpha', indices=[0], names=['$\\alpha_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-beta': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/beta', indices=[0], names=['$\\beta_1$'], signed=[False])],
        x_indices=None,
      ),
      'ext-1-g': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/extensions/1/g', indices=[0], names=['$\\g_1$'], signed=[False])],
        x_indices=None,
      ),
      'bc-0-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/0/dirichlet/indices',
      ),
      'bc-0-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/0/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/0/neumann/indices',
      ),
      'bc-1-dir': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/1/dirichlet/indices',
      ),
      'bc-1-neu': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/1/neumann/g', indices=[0], names=['$g_N$'], signed=[True]),
        ],
        x_indices='boundaries/1/neumann/indices',
      ),
    },
    geo=['sdf', 'sdfgrad'],
    dom=[],
    ext=['ext-0-alpha', 'ext-0-beta', 'ext-0-g', 'ext-1-alpha', 'ext-1-beta', 'ext-1-g'],
    seg=['bc-0-dir', 'bc-0-neu', 'bc-1-dir', 'bc-1-neu'],
    out=['displacement', 'strain', 'stress'],
  ),
  }
