"""RIGNO: Region Interaction Graph Neural Operator.

This module implements a graph-based neural operator that uses regional
integration through p2r (physical to regional) and r2p (regional to physical)
graph connections for multi-scale operator learning.
"""

from typing import Union, Mapping

import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn

from ol.graph.entities import TypedGraph, EdgeSet, EdgesIndices
from ol.graph.graphbuilder import GraphSet
from ol.graph.graphnet import DeepTypedGraphNet
from ol.models.common import AbstractOperator, Inputs
from ol.models.extender import CrossAttentionExtender
from ol.utils import Array, shuffle_arrays


class Encoder(nn.Module):
  """Graph Neural Operator encoder for physical-to-regional message passing.
  
  Lifts physical node features to latent space and aggregates to regional nodes
  using a graph neural network with p2r (physical-to-regional) edges.
  
  Attributes:
      node_latent_size: Latent feature dimension for nodes
      edge_latent_size: Latent feature dimension for edges
      mlp_hidden_layers: Number of hidden layers in MLPs (default: 1)
      use_layer_norm: Apply layer normalization (default: True)
      conditioned_normalization: Use lead-time conditioning (default: True)
      cond_norm_hidden_size: Conditional normalization hidden size (default: True)
      p_edge_masking: Fraction of edges to randomly drop (default: 0.0)
  """

  node_latent_size: int
  edge_latent_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: bool = True
  p_edge_masking: float = .0

  def setup(self):
    # Initialize typed graph neural network with embedding and message passing
    self.gnn = DeepTypedGraphNet(
      embed_nodes=True,  # Embed input node features
      embed_edges=True,  # Embed edge features
      edge_latent_size=dict(p2r=self.edge_latent_size),
      node_latent_size=dict(rnodes=self.node_latent_size, pnodes=self.node_latent_size),
      mlp_num_hidden_layers=self.mlp_hidden_layers,
      num_message_passing_steps=1,  # Single GNN step for encoding
      use_layer_norm=self.use_layer_norm,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=True,  # Use float32 for aggregation stability
      aggregate_edges_for_nodes_fn=jraph.segment_mean,
    )

  def __call__(self,
    graph: TypedGraph,
    input_pnode_features: Array,
    input_rnode_features: Array,
    tau: Union[None, float],
    deterministic: bool = False,
  ) -> tuple[Array, Array]:
    """Extract latent features and aggregate physical nodes to regional nodes.
    
    Args:
        graph: Typed graph with p2r edges and node features
        input_pnode_features: Physical node input features [batch, num_pnodes, feat]
        input_rnode_features: Regional node input features [batch, num_rnodes, feat]
        tau: Lead time parameter for conditional normalization
        deterministic: If True, disable dropout and edge masking (default: False)
    
    Returns:
        Tuple of (regional_node_features, physical_node_features)
        - Regional nodes: [batch, num_rnodes, latent_size]
        - Physical nodes: [batch, num_pnodes, latent_size]
    """
    # Get batch size
    batch_size = input_pnode_features.shape[0]

    # Concatenate input features with structural node features
    pnodes = graph.nodes['pnodes']
    rnodes = graph.nodes['rnodes']
    new_pnodes = pnodes._replace(features=jnp.concatenate([input_pnode_features, pnodes.features], axis=-1))
    new_rnodes = rnodes._replace(features=jnp.concatenate([input_rnode_features, rnodes.features], axis=-1))

    # Get p2r edges from graph
    p2r_edges_key = graph.edge_key_by_name('p2r')
    edges = graph.edges[p2r_edges_key]
    
    # Optionally drop edges for regularization
    if deterministic:
      # Use all edges during evaluation
      n_edges_after = edges.features.shape[1]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    else:
      # Randomly mask out edges during training
      rngkey = self.make_rng('masking')
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[1])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        rngkey=rngkey, arrays=[edges.features, edges.indices.senders, edges.indices.receivers], axis=1)
      new_edge_features = new_edge_features[:, :n_edges_after]
      new_edge_senders = new_edge_senders[:, :n_edges_after]
      new_edge_receivers = new_edge_receivers[:, :n_edges_after]
    
    # Ensure edge features match node feature dtype
    new_edge_features = new_edge_features.astype(input_pnode_features.dtype)
    
    # Construct new edge set with optional masking
    new_edges = EdgeSet(
      n_edge=jnp.tile(jnp.array([n_edges_after]), reps=(batch_size, 1)),
      indices=EdgesIndices(senders=new_edge_senders, receivers=new_edge_receivers),
      features=new_edge_features,
    )

    # Build input graph with updated edges and node features
    input_graph = graph._replace(
      edges={p2r_edges_key: new_edges},
      nodes={'pnodes': new_pnodes, 'rnodes': new_rnodes}
    )

    # Run graph neural network for p2r aggregation
    p2r_out = self.gnn(input_graph, condition=tau)
    latent_rnodes = p2r_out.nodes['rnodes'].features
    latent_pnodes = p2r_out.nodes['pnodes'].features

    return latent_rnodes, latent_pnodes

class Processor(nn.Module):
  """Graph Neural Operator processor for regional-to-regional message passing.
  
  Applies multiple message passing steps on regional nodes using r2r
  (regional-to-regional) edges to refine regional representations.
  
  Attributes:
      steps: Number of message passing iterations
      node_latent_size: Latent feature dimension for nodes
      edge_latent_size: Latent feature dimension for edges
      mlp_hidden_layers: Number of hidden layers in MLPs (default: 1)
      use_layer_norm: Apply layer normalization (default: True)
      conditioned_normalization: Use lead-time conditioning (default: True)
      cond_norm_hidden_size: Conditional normalization hidden size (default: True)
      p_edge_masking: Fraction of edges to randomly drop (default: 0.0)
  """

  steps: int
  node_latent_size: int
  edge_latent_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: bool = True
  p_edge_masking: float = .0

  def setup(self):
    # Initialize typed graph neural network for regional mesh processing
    self.gnn = DeepTypedGraphNet(
      embed_nodes=False,  # Node features already embedded by encoder
      embed_edges=True,  # Embed edge features
      edge_latent_size=dict(r2r=self.edge_latent_size),
      node_latent_size=dict(rnodes=self.node_latent_size),
      mlp_num_hidden_layers=self.mlp_hidden_layers,
      num_message_passing_steps=self.steps,  # Multiple steps for deeper processing
      use_layer_norm=True,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      aggregate_edges_for_nodes_fn=jraph.segment_mean,
    )

  def __call__(self,
    graph: TypedGraph,
    rnode_features: Array,
    tau: Union[None, float],
    deterministic: bool = False,
  ) -> Array:
    """Apply message passing steps on regional nodes.
    
    Args:
        graph: Typed graph with r2r edges
        rnode_features: Regional node features [batch, num_rnodes, latent_size]
        tau: Lead time parameter for conditional normalization
        deterministic: If True, disable dropout and edge masking (default: False)
    
    Returns:
        Updated regional node features [batch, num_rnodes, latent_size]
    """
    # Get batch size
    batch_size = rnode_features.shape[0]

    # Update regional node features (structural features already included from p2r)
    rnodes = graph.nodes['rnodes']
    new_rnodes = rnodes._replace(features=rnode_features)

    # Get r2r edges from graph
    r2r_edges_key = graph.edge_key_by_name('r2r')
    # NOTE: Assumes single edge type in regional mesh
    msg = ('The setup currently requires to only have one kind of edge in the mesh GNN.')
    assert len(graph.edges) == 1, msg
    edges = graph.edges[r2r_edges_key]
    
    # Optionally drop edges for regularization
    if deterministic:
      # Use all edges during evaluation
      n_edges_after = edges.features.shape[1]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    else:
      # Randomly mask out edges during training
      rngkey = self.make_rng('masking')
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[1])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        rngkey=rngkey, arrays=[edges.features, edges.indices.senders, edges.indices.receivers], axis=1)
      new_edge_features = new_edge_features[:, :n_edges_after]
      new_edge_senders = new_edge_senders[:, :n_edges_after]
      new_edge_receivers = new_edge_receivers[:, :n_edges_after]
    
    # Ensure edge features match node feature dtype
    new_edge_features = new_edge_features.astype(rnode_features.dtype)
    
    # Construct new edge set with optional masking
    new_edges = EdgeSet(
      n_edge=jnp.tile(jnp.array([n_edges_after]), reps=(batch_size, 1)),
      indices=EdgesIndices(
        senders=new_edge_senders,
        receivers=new_edge_receivers,
      ),
      features=new_edge_features,
    )

    # Build input graph with updated edges and node features
    input_graph = graph._replace(
      edges={r2r_edges_key: new_edges},
      nodes={'rnodes': new_rnodes},
    )

    # Run graph neural network for r2r message passing
    output_graph = self.gnn(input_graph, condition=tau)
    output_rnodes = output_graph.nodes['rnodes'].features

    return output_rnodes

class Decoder(nn.Module):
  """Graph Neural Operator decoder for regional-to-physical message passing.
  
  Aggregates processed regional node features back to physical nodes via r2p
  (regional-to-physical) edges and projects to output space.
  
  Attributes:
      variable_mesh: If True, input and output meshes can differ (default: False)
      num_outputs: Output feature dimension (output variables)
      node_latent_size: Latent feature dimension for nodes
      edge_latent_size: Latent feature dimension for edges
      mlp_hidden_layers: Number of hidden layers in MLPs (default: 1)
      use_layer_norm: Apply layer normalization (default: True)
      conditioned_normalization: Use lead-time conditioning (default: True)
      cond_norm_hidden_size: Conditional normalization hidden size (default: True)
      p_edge_masking: Fraction of edges to randomly drop (default: 0.0)
  """

  variable_mesh: bool
  num_outputs: int
  node_latent_size: int
  edge_latent_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: bool = True
  p_edge_masking: float = .0

  def setup(self):
    # Initialize typed graph neural network for r2p decoding
    self.gnn = DeepTypedGraphNet(
      # Embed physical node features if meshes differ (variable_mesh)
      embed_nodes=(dict(pnodes=True) if self.variable_mesh else False),
      embed_edges=True,  # Embed edge features
      # Specify output dimension for physical nodes
      node_output_size=dict(pnodes=self.num_outputs),
      edge_latent_size=dict(r2p=self.edge_latent_size),
      node_latent_size=dict(rnodes=self.node_latent_size, pnodes=self.node_latent_size),
      mlp_num_hidden_layers=self.mlp_hidden_layers,
      num_message_passing_steps=1,  # Single GNN step for decoding
      use_layer_norm=True,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      include_sent_messages_in_node_update=False,
      activation='swish',
      f32_aggregation=False,
      # Use segment_mean for handling imbalanced edges
      aggregate_edges_for_nodes_fn=jraph.segment_mean,
    )

  def __call__(self,
    graph: TypedGraph,
    rnode_features: Array,
    pnode_features: Array,
    tau: Union[None, float],
    deterministic: bool = False,
  ) -> Array:
    """Aggregate regional nodes back to physical nodes and project to output.
    
    Args:
        graph: Typed graph with r2p edges
        rnode_features: Regional node features [batch, num_rnodes, latent_size]
        pnode_features: Physical node features [batch, num_pnodes, latent_size]
        tau: Lead time parameter for conditional normalization
        deterministic: If True, disable dropout and edge masking (default: False)
    
    Returns:
        Predicted physical node values [batch, num_pnodes, num_outputs]
    """
    # Get batch size
    batch_size = rnode_features.shape[0]

    # Update regional and physical node features
    rnodes = graph.nodes['rnodes']
    pnodes = graph.nodes['pnodes']
    new_rnodes = rnodes._replace(features=rnode_features)
    # For variable mesh: use struct features; else use latent features
    if self.variable_mesh:
      new_pnodes = pnodes._replace(features=pnodes.features)
    else:
      new_pnodes = pnodes._replace(features=pnode_features)

    # Get r2p edges from graph
    r2p_edges_key = graph.edge_key_by_name('r2p')
    edges = graph.edges[r2p_edges_key]
    
    # Optionally drop edges for regularization
    if deterministic:
      # Use all edges during evaluation
      n_edges_after = edges.features.shape[1]
      new_edge_features = edges.features
      new_edge_senders = edges.indices.senders
      new_edge_receivers = edges.indices.receivers
    else:
      # Randomly mask out edges during training
      rngkey = self.make_rng('masking')
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[1])
      [new_edge_features, new_edge_senders, new_edge_receivers] = shuffle_arrays(
        rngkey=rngkey, arrays=[edges.features, edges.indices.senders, edges.indices.receivers], axis=1)
      new_edge_features = new_edge_features[:, :n_edges_after]
      new_edge_senders = new_edge_senders[:, :n_edges_after]
      new_edge_receivers = new_edge_receivers[:, :n_edges_after]
    
    # Ensure edge features match node feature dtype
    new_edge_features = new_edge_features.astype(pnode_features.dtype)
    
    # Construct new edge set with optional masking
    new_edges = EdgeSet(
      n_edge=jnp.tile(jnp.array([n_edges_after]), reps=(batch_size, 1)),
      indices=EdgesIndices(
        senders=new_edge_senders,
        receivers=new_edge_receivers,
      ),
      features=new_edge_features,
    )

    # Build input graph with updated edges and node features
    input_graph = graph._replace(
      edges={r2p_edges_key: new_edges},
      nodes={'rnodes': new_rnodes, 'pnodes': new_pnodes}
    )

    # Run graph neural network for r2p aggregation and projection
    output_graph = self.gnn(input_graph, condition=tau)
    output_pnodes = output_graph.nodes['pnodes'].features

    return output_pnodes

class RIGNO(AbstractOperator):
  """Region Interaction Graph Neural Operator.
  
  Multi-scale graph neural operator using regional integration:
  (1) Encoder: Aggregates physical nodes to regional nodes (p2r)
  (2) Processor: Refines regional nodes through message passing (r2r)
  (3) Decoder: Maps processed regional features back to physical nodes (r2p)
  
  Supports optional time-dependence with lead-time conditioning.
  
  Attributes:
      num_outputs: Output feature dimension (number of output variables)
      processor_steps: Number of message passing iterations (default: 18)
      node_latent_size: Latent feature dimension for nodes (default: 128)
      edge_latent_size: Latent feature dimension for edges (default: 128)
      mlp_hidden_layers: Hidden layers in MLPs (default: 1)
      p_edge_masking: Edge dropout rate (default: 0.5)
      tdep: Whether to condition on time/lead-time (default: False)
  """

  num_outputs: int
  processor_steps: int = 18
  node_latent_size: int = 128
  edge_latent_size: int = 128
  mlp_hidden_layers: int = 1
  p_edge_masking: int = 0.5
  tdep: bool = False

  def setup(self):
    # NOTE: variable_mesh=True allows input and output meshes to differ
    self.variable_mesh = False

    # Encoder: physical -> regional (p2r)
    self.encoder = Encoder(
      edge_latent_size=self.edge_latent_size,
      node_latent_size=self.node_latent_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      conditioned_normalization=self.tdep,
      cond_norm_hidden_size=16,
      p_edge_masking=self.p_edge_masking,
      name='encoder',
    )

    # Processor: regional -> regional (r2r) message passing
    self.processor = Processor(
      steps=self.processor_steps,
      edge_latent_size=self.edge_latent_size,
      node_latent_size=self.node_latent_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      conditioned_normalization=self.tdep,
      cond_norm_hidden_size=16,
      p_edge_masking=self.p_edge_masking,
      name='processor',
    )

    # Decoder: regional -> physical (r2p)
    self.decoder = Decoder(
      variable_mesh=self.variable_mesh,
      num_outputs=self.num_outputs,
      edge_latent_size=self.edge_latent_size,
      node_latent_size=self.node_latent_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      conditioned_normalization=self.tdep,
      cond_norm_hidden_size=16,
      p_edge_masking=self.p_edge_masking,
      name='decoder',
    )

  @staticmethod
  def _prepare_features(feats: Array) -> Array:
    """Expand time axis from [batch, space, feat] to [batch, 1, space, feat]."""
    feats = jnp.expand_dims(feats, axis=1)
    return feats

  def _encode_process_decode(self, graphs: GraphSet, input_pnode_features: Array, input_rnode_features: Array, tau: Union[None, float], deterministic: bool = False) -> Array:
    """Run the full encode-process-decode pipeline.
    
    Args:
        graphs: GraphSet containing p2r, r2r, and r2p graphs
        input_pnode_features: Physical node features [batch, num_pnodes, feat]
        input_rnode_features: Regional node features [batch, num_rnodes, feat]
        tau: Lead time parameter (optional)
        deterministic: If True, disable dropout and edge masking
    
    Returns:
        Output physical node features [batch, num_pnodes, num_outputs]
    """
    # Add dummy node feature for stable aggregation
    dummy_pnode_features = jnp.zeros(shape=(input_pnode_features.shape[0], 1, input_pnode_features.shape[2]))
    input_pnode_features = jnp.concatenate([input_pnode_features, dummy_pnode_features], axis=1)

    # Encode: Aggregate physical nodes to regional nodes
    latent_rnode_features, latent_pnode_features = self.encoder(graphs.p2r, input_pnode_features, input_rnode_features, tau, deterministic=deterministic)
    self.sow(col='intermediates', name='pnodes-encoded', value=self._prepare_features(latent_pnode_features[:, :-1]))
    self.sow(col='intermediates', name='rnodes-encoded', value=self._prepare_features(latent_rnode_features[:, :-1]))

    # Process: Refine regional nodes through message passing
    processed_rnode_features = self.processor(graphs.r2r, latent_rnode_features, tau, deterministic=deterministic)
    self.sow(col='intermediates', name='rnodes-processed', value=self._prepare_features(processed_rnode_features[:, :-1]))

    # Decode: Aggregate processed regional nodes back to physical nodes
    output_pnode_features = self.decoder(graphs.r2p, processed_rnode_features, latent_pnode_features, tau, deterministic=deterministic)
    self.sow(col='intermediates', name='pnodes-decoded', value=self._prepare_features(output_pnode_features[:, :-1]))

    # Remove dummy node features
    output_pnode_features = output_pnode_features[:, :-1, :]

    return output_pnode_features

  def call(self, inputs: Inputs, graphs: GraphSet, input_pnode_features: Array = None, input_rnode_features: Array = None, deterministic: bool = False) -> Array:
    """Forward pass through the multi-scale graph operator.
    
    Args:
        inputs: Inputs NamedTuple with all operator inputs
        graphs: GraphSet containing p2r, r2r, and r2p graphs
        input_pnode_features: Optional additional physical node features
        input_rnode_features: Optional initialization for regional nodes
        deterministic: If True, disable stochastic components
    
    Returns:
        Predicted function values [batch, 1, num_pnodes_out, num_variables]
    """
    # Read input dimensions
    batch_size = inputs.a.shape[0]
    num_pnodes_inp = inputs.x_inp.shape[2]
    num_pnodes_out = inputs.x_out.shape[2]
    num_rnodes = graphs.p2r.nodes['rnodes'].features.shape[1]

    # Prepare time-dependent conditioning if enabled
    if self.tdep:
      # Extract and normalize time coordinate
      assert inputs.t is not None
      t = jnp.array(inputs.t, dtype=jnp.float32)
      if t.ndim == 4:
        t = t[:, :, 0, 0]
      if t.size == 1:
        t = jnp.tile(t.reshape(1, 1), reps=(batch_size, 1))
      # Extract and normalize lead time parameter
      assert inputs.tau is not None
      tau = jnp.array(inputs.tau, dtype=jnp.float32)
      if tau.ndim == 4:
        tau = tau[:, :, 0, 0]
      if tau.size == 1:
        tau = jnp.tile(tau.reshape(1, 1), reps=(batch_size, 1))
    else:
      tau = None

    # Prepare physical node features from input function
    pnode_features = inputs.a.squeeze(1)
    # Optionally concatenate additional features and time conditioning
    pnode_features_forced = []
    if input_pnode_features is not None:
      pnode_features_forced.append(input_pnode_features)
    if self.tdep:
      pnode_features_forced.append(jnp.tile(jnp.expand_dims(t, axis=1), reps=(1, num_pnodes_inp, 1)))
      pnode_features_forced.append(jnp.tile(jnp.expand_dims(tau, axis=1), reps=(1, num_pnodes_inp, 1)))
    pnode_features = jnp.concatenate([pnode_features, *pnode_features_forced], axis=-1)

    # Initialize regional features if not provided
    if input_rnode_features is None:
      rnode_features = jnp.zeros(shape=(batch_size, num_rnodes, 1), dtype=inputs.a.dtype)
    else:
      rnode_features = input_rnode_features

    # Run encode-process-decode pipeline
    output_pnodes = self._encode_process_decode(graphs=graphs, input_pnode_features=pnode_features, input_rnode_features=rnode_features, tau=tau, deterministic=deterministic)

    # Reshape output to [batch, 1, num_pnodes_out, num_variables]
    output = self._prepare_features(output_pnodes)
    self._check_function(output, x=inputs.x_out)

    return output

class XRIGNO(AbstractOperator):
  """Extended RIGNO with boundary condition incorporation via cross-attention.
  
  Combines RIGNO neural operator with a CrossAttentionExtender module to
  incorporate boundary conditions into the solution. Boundary functions can be
  used to seed regional node features (when use_extender=True) or physical node
  features (when use_extender=False).
  
  Attributes:
      configs_core: Configuration dict for RIGNO operator (num_outputs, latent_size, etc.)
      configs_extender: Configuration dict for CrossAttentionExtender
      use_extender: If True, use cross-attention to extend boundary to regional nodes;
          if False, directly mask and extend boundary to physical nodes
      boundary_size: Maximum number of boundary nodes to process
  """

  configs_core: Mapping
  configs_extender: Mapping
  use_extender: bool
  boundary_size: int

  def setup(self):
    """Initialize RIGNO operator and CrossAttentionExtender modules."""
    # Instantiate core RIGNO operator
    self.operator = RIGNO(**self.configs_core)
    # Instantiate cross-attention based boundary extender
    self.extender = CrossAttentionExtender(**self.configs_extender)

  @nn.compact
  def call(self, inputs: Inputs, graphs: GraphSet, deterministic: bool = False):
    """Forward pass with boundary condition incorporation.
    
    Args:
        inputs: Inputs NamedTuple with domain features (s, a), boundary functions (q),
            boundary masks (m), and spatial coordinates (x_inp, x_out)
        graphs: GraphSet containing p2r, r2r, and r2p graph connectivity
        deterministic: If True, disable stochastic components (dropout, edge masking)
    
    Returns:
        Predicted function values [batch, 1, num_pnodes_out, num_variables]
    """

    if self.use_extender:
      # Project all the boundary functions (e.g., bc-0-dir, bc-1-dir) separately
      xq = jnp.concatenate([inputs.x_inp, inputs.s], axis=-1)  # Geometric features of the whole domain (to be used for boundaries)
      rnodes = graphs.p2r.nodes['rnodes'].features  # Contains only positional encodings of the rnodes
      def extend_group(_q, _m):
        _extensions = self._extend(xq, _q, _m, f_domain=rnodes, deterministic=deterministic)
        return _extensions
      extensions = jax.tree.map(extend_group, inputs.q, inputs.m)
      # Concatenate all the extensions
      jax.tree.map_with_path(lambda key, psi: self.sow(col='intermediates', name=f'extensions-{key[0].key}', value=psi), extensions)
      extensions = jnp.concatenate(jax.tree.flatten(extensions)[0], axis=-1)
      # Pass through the operator
      # NOTE: feeding in rnode features
      output = self.operator(inputs, graphs=graphs, input_rnode_features=extensions, deterministic=deterministic)
    else:
      # Add a variable dimension to the masks
      m_unsqueezed = jax.tree.map(lambda m: m[..., None], inputs.m)
      # Extend the boundary functions with zeros and add a binary mask
      extensions = jax.tree.map(lambda _q, _m: jnp.concatenate([_q, _m.astype(float)], axis=-1), inputs.q, m_unsqueezed)
      extensions = jnp.concatenate(jax.tree.flatten(extensions)[0], axis=-1)
      extensions = jnp.squeeze(extensions, axis=1)

      # Pass through the operator
      # NOTE: feeding in pnode features
      output = self.operator(inputs, graphs=graphs, input_pnode_features=extensions, deterministic=deterministic)

    return output

  def _extend(self, xq: Array, q: Array, m: Array, f_domain: Array, deterministic: bool = False) -> Array:
    """Extract and extend boundary condition functions to domain via cross-attention.
    
    Selects boundary points from the domain using masks, applies cross-attention
    between boundary conditions and domain features, and returns extended features.
    
    Args:
        xq: Domain features (coordinates + geometry) [batch, 1, space, feat]
        q: Boundary condition values [batch, 1, boundary_space, variables]
        m: Binary mask indicating boundary locations [batch, 1, boundary_space]
        f_domain: Domain feature embeddings (regional node features) [batch, num_rnodes, feat]
        deterministic: If True, disable stochastic components
    
    Returns:
        Extended boundary features for conditioning [batch, num_rnodes, out_dim]
    """
    # Vectorized extraction of boundary nodes using masks
    batched_slice = jax.vmap(lambda f, m: f[jnp.where(m, size=self.boundary_size)[0]])
    # Create padding mask for boundary entries (handling variable-length boundaries)
    batched_padmask = jax.vmap(lambda m: jnp.where(jnp.where(m, size=self.boundary_size, fill_value=-1)[0] > -1, 1, 0))
    
    # Extract boundary coordinates and values from domain using masks
    xs_bnd = batched_slice(xq.squeeze(1), m.squeeze(1))  # [batch, boundary_size, feat_coords]
    q_bnd = batched_slice(q.squeeze(1), m.squeeze(1))  # [batch, boundary_size, variables]
    m_bnd = batched_padmask(m.squeeze(1))  # [batch, boundary_size] - marks valid (non-padded) entries
    
    # Randomly shuffle boundary nodes during training for augmentation
    if not deterministic:
      rngkey = self.make_rng(name='other')
      xs_bnd, q_bnd, m_bnd = shuffle_arrays(rngkey=rngkey, arrays=[xs_bnd, q_bnd, m_bnd], axis=1)
    
    # Apply cross-attention to extend boundary conditions to domain
    psi = self.extender(
      f_boundary=jnp.concatenate([xs_bnd, q_bnd], axis=-1),  # Concatenate coordinates + values
      f_domain=f_domain,  # Regional/physical node features to be conditioned
      m_boundary=m_bnd,  # Validity mask for boundary nodes
      deterministic=deterministic,
    )

    return psi
