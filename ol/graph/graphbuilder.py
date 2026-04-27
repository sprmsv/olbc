"""Graph construction for multi-scale operator learning.

This module builds typed graphs connecting physical nodes (pnodes), regional
nodes (rnodes), and their interactions (p2r, r2p, r2r edges). It handles
triangulation, support radius computation, and feature extraction.
"""

from typing import Tuple, Union, NamedTuple

import flax.typing
import jax.numpy as jnp
import jax.random
import numpy as np
from scipy.spatial import Delaunay

from ol.graph.entities import TypedGraph, EdgeSet, EdgeSetKey, EdgesIndices, NodeSet, Context
from ol.utils import Array, shuffle_arrays


class GraphSet(NamedTuple):
  p2r: TypedGraph
  r2r: TypedGraph
  r2p: TypedGraph

  def __len__(self) -> int:
    return self.p2r.nodes['pnodes'].n_node.shape[0]

class GraphMetadata(NamedTuple):
  """Light-weight class for storing graph metadata."""

  x_pnodes_inp: Array
  x_pnodes_out: Array
  x_rnodes: Array
  r_rnodes: Array
  p2r_edge_indices: Array
  r2r_edge_indices: Array
  r2r_edge_domains: Array
  r2p_edge_indices: Array

  def __len__(self) -> int:
    return self.x_pnodes_inp.shape[0]

class GraphBuilder:

  def __init__(self,
    pmesh_subsample_factor: float,
    overlap_factor_p2r: float,
    overlap_factor_r2p: float,
    rmesh_levels: int,
    rmesh_subsample_factor: float,
    periodic: bool,
    node_coordinate_freqs: int,
    gridres: Tuple = None,
  ):
    # Set attributes
    self.pmesh_subsample_factor = pmesh_subsample_factor
    self.overlap_factor_p2r = overlap_factor_p2r
    self.overlap_factor_r2p = overlap_factor_r2p
    self.rmesh_levels = rmesh_levels
    self.rmesh_subsample_factor = rmesh_subsample_factor
    self.periodic = periodic
    self.node_coordinate_freqs = node_coordinate_freqs
    self.gridres = gridres

    # Domain shifts for periodic BC
    self._domain_shifts = jnp.concatenate([
      jnp.array([[0., 0.]]),  # C
      jnp.array([[-2, 0.]]),  # W
      jnp.array([[-2, +2]]),  # NW
      jnp.array([[0., +2]]),  # N
      jnp.array([[+2, +2]]),  # NE
      jnp.array([[+2, 0.]]),  # E
      jnp.array([[+2, -2]]),  # SE
      jnp.array([[0., -2]]),  # S
      jnp.array([[-2, -2]]),  # SW
    ], axis=0)

  def _compute_minimum_support_radius(self, x: Array, x_bnd: Array, z_bnd: Array) -> Array:
    if self.periodic:
      x_extended = (x[None, :, :] + self._domain_shifts[:, None, :]).reshape(-1, 2)
      x_bnd_ext = (x_bnd[None, :, :] + self._domain_shifts[:, None, :]).reshape(-1, 2)
      z_bnd_ext = np.tile(z_bnd, reps=(self._domain_shifts.shape[0], 1))
      points, simplices, _ = construct_triangulation(x=np.concatenate([x_extended, x_bnd_ext]), x_bnd=x_bnd_ext, z_bnd=z_bnd_ext)
    else:
      points, simplices, _ = construct_triangulation(x=np.concatenate([x, x_bnd]), x_bnd=x_bnd, z_bnd=z_bnd)

    radii = np.zeros(shape=(x.shape[0],))
    medians = _compute_triangulation_medians(points, simplices)
    boundary_edge_lengths = _compute_triangulation_boundary_edge_lengths(points, simplices, x.shape[0])
    mask = simplices < x.shape[0] # [N, 3]
    indices = simplices[mask]
    values = np.stack([medians[mask], boundary_edge_lengths[mask]]).max(axis=0)
    sorted_idx = np.argsort(indices)
    sorted_indices = indices[sorted_idx]
    sorted_values = values[sorted_idx]
    unique_indices, idx_start = np.unique(sorted_indices, return_index=True)
    radii[unique_indices] = np.maximum.reduceat(sorted_values, idx_start)

    return radii

  def _get_supported_pnodes_by_rnodes(self,
    centers: Array,
    points: Array,
    radii: Array,
    ord_distance: int = 2,
  ) -> Array:
    """ord_distance can be 1, 2, or np.inf"""

    # Get relative coordinates
    rel = points[:, None] - centers
    # Mirror relative positions because of periodic boudnary conditions
    if self.periodic:
      rel = jnp.where(rel >= 1., (rel - 2.), rel)
      rel = jnp.where(rel < -1., (rel + 2.), rel)

    # Compute distance
    # NOTE: Order of the norm determines the shape of the sub-regions
    distance = jnp.linalg.norm(rel, ord=ord_distance, axis=-1)

    # Get indices
    # -> [idx_point, idx_center]
    idx_nodes = jnp.stack(jnp.where(distance <= radii), axis=-1)

    return idx_nodes

  def _get_r2r_edges(self, x_rnodes: Array, x_bnd: Array, z_bnd: Array) -> Tuple[Array, Array]:
    """Constructrs the processor graph (rmesh to rmesh)"""

    # Define edges and their corresponding -extended- domain
    edges = []
    domains = []
    x_bnd_ext = (x_bnd[None, :, :] + self._domain_shifts[:, None, :]).reshape(-1, 2)
    z_bnd_ext = np.tile(z_bnd, reps=(self._domain_shifts.shape[0], 1))
    for level in range(self.rmesh_levels):
      # Sub-sample the rmesh
      _rmesh_size = int(x_rnodes.shape[0] / (self.rmesh_subsample_factor ** level))
      if _rmesh_size < 4:
        continue
      _x_rmesh = x_rnodes[:_rmesh_size]
      if self.periodic:
        # Repeat the rmesh in periodic directions
        _x_rmesh_extended = (_x_rmesh[None, :, :] + self._domain_shifts[:, None, :]).reshape(-1, 2)
        _, simplices, _edges = construct_triangulation(x=_x_rmesh_extended, x_bnd=x_bnd_ext, z_bnd=z_bnd_ext)
      else:
        _, simplices, _edges = construct_triangulation(x=_x_rmesh, x_bnd=x_bnd, z_bnd=z_bnd)
      # Keep the relevant edges
      domains_level = _edges // _rmesh_size
      edges_level = _edges % _rmesh_size
      idx_relevant_edges = np.any(domains_level == 0, axis=1) if self.periodic else np.all(domains_level == 0, axis=1)
      edges_level = edges_level[idx_relevant_edges]
      domains_level = domains_level[idx_relevant_edges]
      edges.append(edges_level)
      domains.append(domains_level)

    # Remove repeated edges
    edges = jnp.concatenate(edges)
    domains = jnp.concatenate(domains)
    _, unique_idx = jnp.unique(edges, axis=0, return_index=True)
    edges = edges[unique_idx]
    domains = domains[unique_idx]

    return edges, domains

  def build_metadata(self, x_inp: Array, x_out: Array, x_bnd: Array, z_bnd: Array, bbox: Array, rmesh_correction_dsf: float = 1.0, rngkey: Union[flax.typing.PRNGKey, None] = None) -> GraphMetadata:
    """Build graph metadata including node coordinates and edge indices.
    
    Computes regional mesh (rnodes) by subsampling input points, then builds
    edge connectivity for p2r (physical-to-regional), r2r, and r2p flows.
    """

    # Normalize coordinates in [-1, +1)
    x_inp = 2 * (x_inp - bbox[0]) / (bbox[1] - bbox[0]) - 1
    x_out = 2 * (x_out - bbox[0]) / (bbox[1] - bbox[0]) - 1
    x_bnd = 2 * (x_bnd - bbox[0]) / (bbox[1] - bbox[0]) - 1

    # Randomly sub-sample pmesh to get rmesh
    if rngkey is None: rngkey = jax.random.key(0)
    if self.gridres is not None:
      x_rnodes = jnp.stack(jnp.meshgrid(*[jnp.linspace(bbox[0][d], bbox[1][d], self.gridres[d]) for d in range(len(bbox))]), axis=-1)
      x_rnodes = x_rnodes.reshape(-1, len(bbox))
    else:
      x_rnodes = _subsample_pointset(rngkey=rngkey, x=x_inp, factor=self.pmesh_subsample_factor)

    # Downsample or upsample the rmesh
    if rmesh_correction_dsf > 1:
      x_rnodes = _subsample_pointset(rngkey=rngkey, x=x_rnodes, factor=rmesh_correction_dsf)
    elif rmesh_correction_dsf < 1:
      x_rnodes = _upsample_pointset(rngkey=rngkey, x=x_rnodes, x_bnd=x_bnd, z_bnd=z_bnd, factor=(1 / rmesh_correction_dsf))

    # Compute minimum support radius of each rmesh node
    r_rnodes = self._compute_minimum_support_radius(x_rnodes, x_bnd, z_bnd)

    # Get edge indices
    p2r_edge_indices = self._get_supported_pnodes_by_rnodes(
      centers=x_rnodes,
      points=x_inp,
      radii=(self.overlap_factor_p2r * r_rnodes),
    )
    r2r_edge_indices, r2r_edge_domains = self._get_r2r_edges(x_rnodes, x_bnd, z_bnd)
    r2p_edge_indices = self._get_supported_pnodes_by_rnodes(
      centers=x_rnodes,
      points=x_inp,
      radii=(self.overlap_factor_r2p * r_rnodes),
    )
    r2p_edge_indices = jnp.flip(r2p_edge_indices, axis=-1)

    # Add dummy nodes and edges
    p2r_edge_indices = jnp.concatenate([p2r_edge_indices, jnp.array([[x_inp.shape[0], x_rnodes.shape[0]]])], axis=0)
    r2r_edge_indices = jnp.concatenate([r2r_edge_indices, jnp.array([[x_rnodes.shape[0], x_rnodes.shape[0]]])], axis=0)
    r2r_edge_domains = jnp.concatenate([r2r_edge_domains, jnp.array([[0, 0]])], axis=0)
    r2p_edge_indices = jnp.concatenate([r2p_edge_indices, jnp.array([[x_rnodes.shape[0], x_out.shape[0]]])], axis=0)
    x_inp = jnp.concatenate([x_inp, jnp.zeros(shape=(1, x_inp.shape[-1]))], axis=0)
    x_out = jnp.concatenate([x_out, jnp.zeros(shape=(1, x_out.shape[-1]))], axis=0)
    x_rnodes = jnp.concatenate([x_rnodes, jnp.zeros(shape=(1, x_rnodes.shape[-1]))], axis=0)
    r_rnodes = jnp.concatenate([r_rnodes, jnp.zeros(shape=(1,))], axis=0)

    # Convert dtypes to save memory
    r2r_edge_domains = r2r_edge_domains.astype(jnp.uint8)
    if (max(x_inp.shape[0], x_out.shape[0]) < jnp.iinfo(jnp.uint16).max):
      p2r_edge_indices=p2r_edge_indices.astype(jnp.uint16)
      r2r_edge_indices=r2r_edge_indices.astype(jnp.uint16)
      r2p_edge_indices=r2p_edge_indices.astype(jnp.uint16)
    # Ommit storing duplicated edge indices
    if self.overlap_factor_p2r == self.overlap_factor_r2p:
      # NOTE: it will be the inverse of p2r edges
      r2p_edge_indices = None

    # Store the graph data
    graph_metadata = GraphMetadata(
      x_pnodes_inp=jnp.expand_dims(x_inp, axis=0),
      x_pnodes_out=jnp.expand_dims(x_out, axis=0),
      x_rnodes=jnp.expand_dims(x_rnodes, axis=0),
      r_rnodes=jnp.expand_dims(r_rnodes, axis=0),
      p2r_edge_indices=jnp.expand_dims(p2r_edge_indices, axis=0),
      r2r_edge_indices=jnp.expand_dims(r2r_edge_indices, axis=0),
      r2r_edge_domains=jnp.expand_dims(r2r_edge_domains, axis=0),
      r2p_edge_indices=(jnp.expand_dims(r2p_edge_indices, axis=0) if (r2p_edge_indices is not None) else None),
    )

    return graph_metadata

  def _init_structural_features(self,
    x_sen: Array,
    x_rec: Array,
    idx_sen: Array,
    idx_rec: Array,
    max_edge_length: float,
    feats_sen: Array = None,
    feats_rec: Array = None,
    shift: bool = False,
    domain_sen: Array = None,
    domain_rec: Array = None,
  ) -> Tuple[EdgeSet, NodeSet, NodeSet]:

    # Get number of nodes and the edges
    batch_size = x_sen.shape[0]
    num_sen = x_sen.shape[1]
    num_rec = x_rec.shape[1]
    assert idx_sen.shape[1] == idx_rec.shape[1]
    num_edg = idx_sen.shape[1]

    # Process coordinates
    phi_sen = jnp.pi * (x_sen + 1)  # [0, 2pi]
    phi_rec = jnp.pi * (x_rec + 1)  # [0, 2pi]

    # Define node features
    # NOTE: Sinusoidal features don't need normalization
    if self.periodic:
      k = jnp.arange(self.node_coordinate_freqs)
      phi_sen_sin = jax.vmap(fun=(lambda _v, _k: jnp.sin(_v * (_k+1))), in_axes=(None, 0), out_axes=-1)(phi_sen, k)
      phi_sen_cos = jax.vmap(fun=(lambda _v, _k: jnp.cos(_v * (_k+1))), in_axes=(None, 0), out_axes=-1)(phi_sen, k)
      sender_node_feats = jnp.concatenate(
        arrays=[
          phi_sen_sin.reshape(*phi_sen_sin.shape[:-2], -1),
          phi_sen_cos.reshape(*phi_sen_cos.shape[:-2], -1)
        ], axis=-1)
      phi_rec_sin = jax.vmap(fun=(lambda _v, _k: jnp.sin(_v * (_k+1))), in_axes=(None, 0), out_axes=-1)(phi_rec, k)
      phi_rec_cos = jax.vmap(fun=(lambda _v, _k: jnp.cos(_v * (_k+1))), in_axes=(None, 0), out_axes=-1)(phi_rec, k)
      receiver_node_feats = jnp.concatenate(
        arrays=[
          phi_rec_sin.reshape(*phi_rec_sin.shape[:-2], -1),
          phi_rec_cos.reshape(*phi_rec_cos.shape[:-2], -1)
        ], axis=-1)
    else:
      sender_node_feats = jnp.concatenate([x_sen], axis=-1)
      receiver_node_feats = jnp.concatenate([x_rec], axis=-1)
    # Concatenate with forced features
    if feats_sen is not None:
      sender_node_feats = jnp.concatenate([sender_node_feats, feats_sen], axis=-1)
    if feats_rec is not None:
      receiver_node_feats = jnp.concatenate([receiver_node_feats, feats_rec], axis=-1)

    # Build node sets
    sender_node_set = NodeSet(
      n_node=jnp.tile(jnp.array([num_sen]), reps=(batch_size, 1)),
      features=sender_node_feats,
    )
    receiver_node_set = NodeSet(
      n_node=jnp.tile(jnp.array([num_rec]), reps=(batch_size, 1)),
      features=receiver_node_feats,
    )

    # Define edge features
    batched_index = jax.vmap(lambda f, idx: f[idx])
    batched_index_single = jax.vmap(lambda f, idx: f[idx], in_axes=(None, 0))
    z_ij = batched_index(x_sen, idx_sen) - batched_index(x_rec, idx_rec)
    if self.periodic:
      # NOTE: For p2r and r2p, mirror the large relative coordinates
      if not shift:
        z_ij = jnp.where(z_ij < -1.0, z_ij + 2, z_ij)
        z_ij = jnp.where(z_ij >= 1.0, z_ij - 2, z_ij)
      # NOTE: For the r2r multi-mesh, use extended domain indices and shifts
      else:
        z_ij = (
          (batched_index(x_sen, idx_sen) + batched_index_single(self._domain_shifts, domain_sen))
          - (batched_index(x_rec, idx_rec) + batched_index_single(self._domain_shifts, domain_rec))
        )
    d_ij = jnp.linalg.norm(z_ij, axis=-1, keepdims=True)
    # Normalize and concatenate edge features
    z_ij = z_ij / max_edge_length
    d_ij = d_ij / max_edge_length
    edge_feats = jnp.concatenate([z_ij, d_ij], axis=-1)

    # Build edge set
    edge_set = EdgeSet(
      n_edge=jnp.tile(jnp.array([num_edg]), reps=(batch_size, 1)),
      indices=EdgesIndices(
        senders=idx_sen,
        receivers=idx_rec,
      ),
      features=edge_feats,
    )

    return edge_set, sender_node_set, receiver_node_set

  def _build_p2r_graph(self, x_pnodes: Array, x_rnodes: Array, idx_edges: Array, r_rmesh: Array) -> TypedGraph:
    """Constructs the encoder graph (pmesh to rmesh)"""

    # Get the initial features
    edge_set, pmesh_node_set, rmesh_node_set = self._init_structural_features(
      x_sen=x_pnodes,
      x_rec=x_rnodes,
      idx_sen=idx_edges[..., 0],
      idx_rec=idx_edges[..., 1],
      max_edge_length=(2. * jnp.sqrt(x_rnodes.shape[-1])),
      feats_rec=jnp.expand_dims(self.overlap_factor_p2r * r_rmesh, axis=-1),
    )

    # Construct the graph
    graph = TypedGraph(
      context=Context(n_graph=jnp.tile(jnp.array([1]), reps=(x_rnodes.shape[0], 1)), features=()),
      nodes={'pnodes': pmesh_node_set, 'rnodes': rmesh_node_set},
      edges={EdgeSetKey('p2r', ('pnodes', 'rnodes')): edge_set},
    )

    return graph

  def _build_r2r_graph(self, x_rnodes: Array, idx_edges: Array, idx_domains: Array, r_rmesh: Array) -> TypedGraph:
    """Constructs the processor graph (rmesh to rmesh)"""

    # Set the initial features
    edge_set, rmesh_node_set, _ = self._init_structural_features(
      x_sen=x_rnodes,
      x_rec=x_rnodes,
      idx_sen=idx_edges[..., 0],
      idx_rec=idx_edges[..., 1],
      max_edge_length=(2. * jnp.sqrt(x_rnodes.shape[-1])),
      feats_sen=jnp.expand_dims(self.overlap_factor_p2r * r_rmesh, axis=-1),
      feats_rec=jnp.expand_dims(self.overlap_factor_r2p * r_rmesh, axis=-1),
      shift=True,
      domain_sen=idx_domains[..., 0],
      domain_rec=idx_domains[..., 1],
    )

    # Construct the graph
    graph = TypedGraph(
      context=Context(n_graph=jnp.tile(jnp.array([1]), reps=(x_rnodes.shape[0], 1)), features=()),
      nodes={'rnodes': rmesh_node_set},
      edges={EdgeSetKey('r2r', ('rnodes', 'rnodes')): edge_set},
    )

    return graph

  def _build_r2p_graph(self, x_pnodes: Array, x_rnodes: Array, idx_edges: Array, r_rmesh: Array) -> TypedGraph:
    """Constructs the decoder graph (rmesh to pmesh)"""

    # Get the initial features
    edge_set, rmesh_node_set, pmesh_node_set = self._init_structural_features(
      x_sen=x_rnodes,
      x_rec=x_pnodes,
      idx_sen=idx_edges[..., 0],
      idx_rec=idx_edges[..., 1],
      max_edge_length=(2. * jnp.sqrt(x_rnodes.shape[-1])),
      feats_sen=jnp.expand_dims(self.overlap_factor_r2p * r_rmesh, axis=-1),
    )

    # Construct the graph
    graph = TypedGraph(
      context=Context(n_graph=jnp.tile(jnp.array([1]), reps=(x_rnodes.shape[0], 1)), features=()),
      nodes={'pnodes': pmesh_node_set, 'rnodes': rmesh_node_set},
      edges={EdgeSetKey('r2p', ('rnodes', 'pnodes')): edge_set},
    )

    return graph

  def build_graphs(self, metadata: GraphMetadata) -> GraphSet:
    """Build three typed graphs from metadata: p2r, r2r, and r2p.
    
    Creates graph structures with nodes and edges including node features
    (sinusoidal position encodings) and edge features (relative coordinates).
    """

    # Unwrap the attributes
    x_pnodes_inp = metadata.x_pnodes_inp
    x_pnodes_out = metadata.x_pnodes_out
    x_rnodes = metadata.x_rnodes
    r_rnodes = metadata.r_rnodes
    p2r_edge_indices = metadata.p2r_edge_indices
    r2r_edge_indices = metadata.r2r_edge_indices
    r2r_edge_domains = metadata.r2r_edge_domains
    r2p_edge_indices = metadata.r2p_edge_indices
    # Flip p2r indices if r2p is None
    if r2p_edge_indices is None:
      r2p_edge_indices = jnp.flip(metadata.p2r_edge_indices, axis=-1)

    # Build the graphs
    graphs = GraphSet(
      p2r=self._build_p2r_graph(x_pnodes_inp, x_rnodes, p2r_edge_indices, r_rnodes),
      r2r=self._build_r2r_graph(x_rnodes, r2r_edge_indices, r2r_edge_domains, r_rnodes),
      r2p=self._build_r2p_graph(x_pnodes_out, x_rnodes, r2p_edge_indices, r_rnodes),
    )

    return graphs

def construct_triangulation(x, x_bnd, z_bnd):
  """
  Constructs a triangulation on a given point cloud. By creating a shell around
  the boundaries, and later removing them, it supports non-convex geometries.

  Args:
      x: Point cloud coordinates.
      x_bnd: Coordinates of the boundaries.
      z_bnd: Gradient of the signed distance function at the boundary coordinates.
  """

  # Compute margin based on the given point cloud
  distances = np.linalg.norm(x_bnd[None, :] - x_bnd[:, None], axis=-1)
  margin = np.median(np.where(distances>0, distances, distances.max()).min(axis=-1))  # median distance to the closest point
  # Compute shell coordinates
  z_nrm = z_bnd / np.linalg.norm(z_bnd, axis=1, keepdims=True)
  x_shell = x_bnd - margin * z_nrm
  # Build Delaunay triangulation on the shelled coordinates
  x_shelled = np.concatenate([x, x_shell])
  tri = Delaunay(x_shelled)
  # Get the viable edges
  edges = _get_edges_from_triangulation(tri.simplices)
  edges = edges[np.all(edges < x.shape[0], axis=1)]
  # Remove the extra simplices
  simplices = tri.simplices[np.all(tri.simplices < x.shape[0], axis=1)]
  points = x

  return points, simplices, edges

def _subsample_pointset(rngkey, x: Array, factor: float) -> Array:
  """Downsamples a point cloud by randomly subsampling them"""

  x = jnp.array(x)
  x_shuffled, = shuffle_arrays(rngkey, [x])

  return x_shuffled[:int(x.shape[0] / factor)]

def _upsample_pointset(rngkey, x: Array, x_bnd: Array, z_bnd: Array, factor: float) -> Array:
  """Upsamples a point cloud by adding the middle point of randomly selected simplices."""

  factor = factor ** x.shape[-1]
  num_new_points = int(x.shape[0] * (factor - 1))
  _, simplices, _ = construct_triangulation(x, x_bnd, z_bnd)
  simplices = jax.random.permutation(key=rngkey, x=simplices)[jnp.arange(num_new_points)]
  x_ext = np.mean(x[simplices], axis=1)

  return np.concatenate([x, x_ext], axis=0)

def _get_edges_from_triangulation(simplices: Array, bidirectional: bool = True):
  """Reads unique edges from a given set of simplices."""

  edges = np.concatenate([np.delete(simplices, i, axis=1) for i in range(simplices.shape[1])])
  edges = np.unique(np.sort(edges, axis=1), axis=0)
  if bidirectional:
    edges = np.concatenate([edges, np.flip(edges, axis=-1)], axis=0)

  return edges

def _compute_triangulation_medians(points: Array, simplices: Array) -> Array:
  """
  Computes the medians of all the triangles in the triangulation.
  Only supports 2D triangulations.

  Args:
      points: The points of the triangulation.
      simplices: The simplices of the triangulation.

  Returns:
      A matrix with the shape of simplices, where each index shows the median
        that crosses the corresponding vertex in the simplex.
  """

  edge_lengths = np.zeros(shape=simplices.shape)
  medians = np.zeros(shape=simplices.shape)
  for i in range(simplices.shape[1]):
    _points = points[np.delete(simplices, i, axis=1)]
    _points = [p.squeeze(1) for p in np.split(_points, axis=1, indices_or_sections=2)]
    edge_lengths[:, i] = np.linalg.norm(np.subtract(*_points), axis=1)
  for i in range(simplices.shape[1]):
    medians[:, i] = .67 * np.sqrt((2 * np.sum(np.power(np.delete(edge_lengths, i, axis=1), 2), axis=1) - np.power(edge_lengths[:, i], 2)) / 4)

  return medians

def _compute_triangulation_boundary_edge_lengths(points: Array, simplices: Array, num_internal_points: int) -> Array:
  """
  Computes the length of the edge that connect a boundary node to an internal node.
  Only supports 2D triangulations.

  Args:
      points: The points of the triangulation including both the internal points (leading)
        and the boundary points.
      simplices: The simplices of the triangulation.
      num_internal_points: Number of the internal points.

  Returns:
      A matrix with the shape of simplices, where each index shows the length of the edge that
        is in front of the corresponding vertex.
  """

  edge_lengths = np.zeros(shape=simplices.shape)
  boundary_edge = np.zeros(shape=simplices.shape).astype(bool)
  for i in range(simplices.shape[1]):
    _points = points[np.delete(simplices, i, axis=1)]
    _simplices = np.delete(simplices, i, axis=1)
    _points = [p.squeeze(1) for p in np.split(_points, axis=1, indices_or_sections=2)]
    edge_lengths[:, i] = np.linalg.norm(np.subtract(*_points), axis=1)
    boundary_edge[:, i] = np.any(_simplices >= num_internal_points, axis=1)
  boundary_edge_lengths = np.where(boundary_edge, edge_lengths, 0.0)

  return boundary_edge_lengths
