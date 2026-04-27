from typing import Tuple, Union, Mapping, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from ol.graph.graphbuilder import GraphSet, TypedGraph
from ol.models.common import AbstractOperator, Inputs, LeadTimeConditionedNorm, FeedForward, segment_softmax
from ol.models.extender import CrossAttentionExtender
from ol.utils import Array, shuffle_arrays


class AGNO(nn.Module):
  """Attention-based Graph Neural Operator for message aggregation.
  
  Implements attention-weighted message passing where edge messages are weighted
  by the attention score between receiver and sender features. All edges from
  a receiver are aggregated together with learned attention weights.
  
  Attributes:
      output_size: Output feature dimension
      mlp_hidden_layers: Number of hidden layers in feature MLP (default: 3)
      mlp_hidden_size: Hidden dimension size (default: 64)
      use_layer_norm: Apply layer normalization in MLP (default: True)
      conditioned_normalization: Use conditioning-dependent normalization (default: True)
      cond_norm_hidden_size: Hidden size for conditional norm (default: 4)
      sorted_per_receiver: Assume receiver indices are sorted for efficiency (default: False)
  """

  output_size: int
  mlp_hidden_layers: int = 3
  mlp_hidden_size: int = 64
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: int = 4
  sorted_per_receiver: bool = False

  def setup(self):
    # Construct MLP layer sizes: hidden layers + output layer
    mlp_layer_sizes = [self.mlp_hidden_size]*self.mlp_hidden_layers + [self.output_size]
    # Feature transformation MLP
    self.ff = FeedForward(
      layer_sizes=mlp_layer_sizes,
      use_layer_norm=self.use_layer_norm,
      use_conditional_norm=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      activation=nn.gelu,
    )
    # Projection for computing attention key from sender features
    self.query_proj = nn.Dense(features=self.mlp_hidden_size)
    self.key_proj = nn.Dense(features=self.mlp_hidden_size)
    # Scaling factor for attention scores: 1/sqrt(hidden_dim)
    self.scaling_factor = 1.0 / (self.mlp_hidden_size ** 0.5)

  def __call__(self, src: Array, rcv: Array, edg_indices: Array, weights: Array = None, condition: Array = None, deterministic: bool = False):
    """Apply attention-weighted message passing on graph.
    
    Args:
        src: Source node features [batch, num_src, feat_dim]
        rcv: Receiver node features [batch, num_rcv, feat_dim]
        edg_indices: Edge indices [batch, num_edges, 2] where edg_indices[..., 0] = senders, [..., 1] = receivers
        weights: Optional pre-computed edge weights [batch, num_edges]. If None, computed from attention
        condition: Optional conditioning input for conditional normalization [batch, 1]
        deterministic: If True, disable dropout (default: False)
    
    Returns:
        Aggregated messages per receiver [batch, num_rcv, output_size]
    """
    # Create batched indexing function for gathering features by index
    batched_index = jax.vmap(lambda f, idx: f[idx])
    # Extract sender and receiver indices from edge representation
    src_indices, rcv_indices = edg_indices[:, :, 0], edg_indices[:, :, 1]
    # Gather sender and receiver features for each edge
    edg_sen = batched_index(src, src_indices)
    edg_rcv = batched_index(rcv, rcv_indices)
    # Concatenate sender and receiver features
    edg = jnp.concatenate([edg_sen, edg_rcv], axis=-1)

    # Prepare aggregation weights using attention or pre-computed weights
    if weights is None:
      # Compute attention-based weights: key from senders, query from receivers
      key = self.key_proj(edg_sen)
      query = self.query_proj(edg_rcv)
      # Scaled dot-product attention scores
      attention_scores = jnp.sum(query * key, axis=-1) * self.scaling_factor
      # Normalize attention per receiver using segment softmax
      weights = jax.vmap(segment_softmax, in_axes=(0, 0, None, None))(attention_scores, rcv_indices, rcv.shape[1], self.sorted_per_receiver)
    else:
      # Normalize pre-computed weights to sum to 1 per receiver
      weights /= jax.vmap(jax.ops.segment_sum, in_axes=(0, 0, None, None))(weights, rcv_indices, rcv.shape[1], self.sorted_per_receiver)[rcv_indices]

    # Transform edge features through MLP and scale by attention weights
    edg = jnp.expand_dims(weights, axis=-1) * self.ff(edg, c=condition, deterministic=deterministic)
    # Aggregate weighted edge features (messages) to each receiver node
    out_features = jax.vmap(jax.ops.segment_sum, in_axes=(0, 0, None, None))(edg, rcv_indices, rcv.shape[1], self.sorted_per_receiver)

    return out_features

class Encoder(nn.Module):
  """Graph Neural Operator encoder for physical-to-regional message passing.
  
  Lifts physical domain features to latent space and aggregates to regional nodes
  via attention-weighted p2r (physical-to-regional) edges. Forms the first layer
  of the multi-scale operator architecture.
  
  Attributes:
      latent_size: Latent feature dimension
      output_size: Output feature dimension per regional node
      mlp_hidden_layers: Hidden layers in feature MLP (default: 1)
      use_layer_norm: Apply layer normalization (default: True)
      conditioned_normalization: Use lead-time conditioned norm (default: True)
      cond_norm_hidden_size: Conditional norm hidden size (default: 4)
      p_edge_masking: Fraction of p2r edges to randomly drop (default: 0.0)
      dropout: Dropout rate in feature transformations (default: 0.0)
  """
  latent_size: int
  output_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: int = 4
  p_edge_masking: float = .0
  dropout: float = .0

  def setup(self):
    # Feature lifting for physical nodes
    self.lifting_pnodes = FeedForward(
      conv=True,
      layer_sizes=[self.latent_size],
      activation=nn.gelu,
      use_layer_norm=self.use_layer_norm,
      use_conditional_norm=False,
      dropout=self.dropout,
    )
    # Feature lifting for regional nodes
    self.lifting_rnodes = FeedForward(
      conv=True,
      layer_sizes=[self.latent_size],
      activation=nn.gelu,
      use_layer_norm=self.use_layer_norm,
      use_conditional_norm=False,
      dropout=self.dropout,
    )
    # Attention-based aggregation from physical to regional nodes
    self.agno = AGNO(
      output_size=self.output_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      mlp_hidden_size=self.latent_size,
      use_layer_norm=self.use_layer_norm,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      sorted_per_receiver=True,
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
        - Regional nodes: [batch, num_rnodes, output_size]
        - Physical nodes: [batch, num_pnodes, latent_size]
    """
    # Concatenate input features with structural features
    pnodes = jnp.concatenate([input_pnode_features, graph.nodes['pnodes'].features], axis=-1)
    rnodes = jnp.concatenate([input_rnode_features, graph.nodes['rnodes'].features], axis=-1)
    # Get p2r edges from graph
    p2r_edges_key = graph.edge_key_by_name('p2r')
    edges = graph.edges[p2r_edges_key]

    # Optionally drop edges for regularization
    if deterministic:
      # Use all edges during evaluation
      n_edges_after = edges.features.shape[1]
      edge_senders = edges.indices.senders
      edge_receivers = edges.indices.receivers
    else:
      # Randomly mask out edges during training
      rngkey = self.make_rng('masking')
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[1])
      [edge_senders, edge_receivers] = shuffle_arrays(
        rngkey=rngkey, arrays=[edges.indices.senders, edges.indices.receivers], axis=1)
      edge_senders = edge_senders[:, :n_edges_after]
      edge_receivers = edge_receivers[:, :n_edges_after]

    # Lift physical and regional features to latent space
    pnodes = self.lifting_pnodes(pnodes, c=tau, deterministic=deterministic)
    rnodes = self.lifting_rnodes(rnodes, c=tau, deterministic=deterministic)

    # Sort edges by receiver for efficient per-receiver aggregation
    order = jnp.argsort(edge_receivers, axis=1)
    edge_senders = jnp.take_along_axis(edge_senders, order, axis=1)
    edge_receivers = jnp.take_along_axis(edge_receivers, order, axis=1)
    indices = jnp.stack([edge_senders, edge_receivers], axis=-1)
    # Aggregate physical node messages to regional nodes with attention
    rnodes = self.agno(src=pnodes, rcv=rnodes, edg_indices=indices, weights=None, condition=tau, deterministic=deterministic)

    return rnodes, pnodes

class Decoder(nn.Module):
  """Graph Neural Operator decoder for regional-to-physical message passing.
  
  Aggregates regional node features back to physical nodes via r2p
  (regional-to-physical) edges and projects to output space. Forms the final
  layer of the multi-scale operator, producing physical domain predictions.
  
  Attributes:
      latent_size: Latent feature dimension
      output_size: Output feature dimension (output variables)
      mlp_hidden_layers: Hidden layers in feature MLP (default: 1)
      use_layer_norm: Apply layer normalization (default: True)
      conditioned_normalization: Use lead-time conditioned norm (default: True)
      cond_norm_hidden_size: Conditional norm hidden size (default: 4)
      p_edge_masking: Fraction of r2p edges to randomly drop (default: 0.0)
      dropout: Dropout rate in feature transformations (default: 0.0)
  """
  latent_size: int
  output_size: int
  mlp_hidden_layers: int = 1
  use_layer_norm: bool = True
  conditioned_normalization: bool = True
  cond_norm_hidden_size: int = 4
  p_edge_masking: float = .0
  dropout: float = .0

  def setup(self):
    # Attention-based aggregation from regional to physical nodes
    self.agno = AGNO(
      output_size=self.latent_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      mlp_hidden_size=self.latent_size,
      use_layer_norm=self.use_layer_norm,
      conditioned_normalization=self.conditioned_normalization,
      cond_norm_hidden_size=self.cond_norm_hidden_size,
      sorted_per_receiver=True,
    )
    # Projection to output feature dimension
    self.projection = FeedForward(
      conv=True,
      layer_sizes=[self.output_size],
      activation=nn.gelu,
      use_layer_norm=False,
      use_conditional_norm=False,
      dropout=self.dropout,
    )

  def __call__(self,
    graph: TypedGraph,
    rnode_features: Array,
    pnode_features: Array,
    tau: Union[None, float],
    deterministic: bool = False,
  ) -> tuple[Array, Array]:
    """Aggregate regional nodes back to physical nodes and project to output.
    
    Args:
        graph: Typed graph with r2p edges and node features
        rnode_features: Regional node features [batch, num_rnodes, feat]
        pnode_features: Physical node features [batch, num_pnodes, feat]
        tau: Lead time parameter for conditional normalization
        deterministic: If True, disable dropout and edge masking (default: False)
    
    Returns:
        Tuple of (physical_node_predictions, regional_node_features)
        - Physical predictions: [batch, num_pnodes, output_size]
        - Regional features: [batch, num_rnodes, feat] (unchanged)
    """
    rnodes = rnode_features
    pnodes = pnode_features
    # Get r2p edges from graph
    r2p_edges_key = graph.edge_key_by_name('r2p')
    edges = graph.edges[r2p_edges_key]

    # Optionally drop edges for regularization
    if deterministic:
      # Use all edges during evaluation
      n_edges_after = edges.features.shape[1]
      edge_senders = edges.indices.senders
      edge_receivers = edges.indices.receivers
    else:
      # Randomly mask out edges during training
      rngkey = self.make_rng('masking')
      n_edges_after = int((1 - self.p_edge_masking) * edges.features.shape[1])
      [edge_senders, edge_receivers] = shuffle_arrays(
        rngkey=rngkey, arrays=[edges.indices.senders, edges.indices.receivers], axis=1)
      edge_senders = edge_senders[:, :n_edges_after]
      edge_receivers = edge_receivers[:, :n_edges_after]

    # Sort edges by receiver for efficient per-receiver aggregation
    order = jnp.argsort(edge_receivers, axis=1)
    edge_senders = jnp.take_along_axis(edge_senders, order, axis=1)
    edge_receivers = jnp.take_along_axis(edge_receivers, order, axis=1)
    indices = jnp.stack([edge_senders, edge_receivers], axis=-1)
    # Aggregate regional node messages to physical nodes with attention
    pnodes_update = self.agno(src=rnodes, rcv=pnodes, edg_indices=indices, weights=None, condition=tau, deterministic=deterministic)
    # Add residual connection if dimensions match, otherwise replace
    if pnodes.shape[-1] == self.latent_size:
      pnodes += pnodes_update
    else:
      pnodes = pnodes_update
    # Project latent features to output dimension
    pnodes = self.projection(pnodes, c=tau, deterministic=deterministic)

    return pnodes

class GroupQueryFlashAttention(nn.Module):
  num_heads: int
  head_dim: int
  use_conditional_norm: bool = False
  cond_norm_hidden_size: int = 4

  def setup(self):
    self.q_proj = nn.Dense(features=(self.num_heads * self.head_dim), use_bias=False)
    self.k_proj = nn.Dense(features=(self.num_heads * self.head_dim), use_bias=False)
    self.v_proj = nn.Dense(features=(self.num_heads * self.head_dim), use_bias=False)

  @nn.compact
  def __call__(self, x, condition: Optional[float] = None):
    input_size = x.shape[-1]

    if self.use_conditional_norm:
      x = LeadTimeConditionedNorm(self.cond_norm_hidden_size, x.shape[-1])(c=condition, x=x)

    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    batch_size, seq_len, _ = q.shape

    # NOTE: The input dimensions of jax.nn.dot_product_attention are different from torch.nn.functional.scaled_dot_product_attention
    # [batch_size, seq_len, num_heads, head_dim]
    q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
    k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
    v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
    x = jax.nn.dot_product_attention(q, k, v)  # [batch_size, seq_len, num_heads, head_dim]
    x = x.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, num_heads * head_dim]
    x = nn.Dense(input_size, use_bias=False)(x)  # [batch_size, seq_len, input_size]

    return x

class FFN(nn.Module):
  output_size: int
  ffn_hidden_size: int
  use_conditional_norm: bool = False
  cond_norm_hidden_size: int = 4

  @nn.compact
  def __call__(self, x, condition: Optional[float] = None):
    x = nn.Dense(self.output_size, use_bias=False)(nn.silu(nn.Dense(self.ffn_hidden_size, use_bias=False)(x)) * nn.Dense(self.ffn_hidden_size, use_bias=False)(x))
    if self.use_conditional_norm:
      x = LeadTimeConditionedNorm(self.cond_norm_hidden_size, x.shape[-1])(c=condition, x=x)

    return x

class RMSNorm(nn.Module):
  eps: float = 1e-6

  def _norm(self, x):
    x = jnp.astype(x, float)
    return x * jax.lax.rsqrt(jnp.pow(x, 2).mean(-1, keepdims=True) + self.eps)

  @nn.compact
  def __call__(self, x):
    input_shape = x.shape[-1]
    weight = self.param('weights', (lambda rng, shape: jnp.ones(shape)), input_shape)
    output = jnp.astype(self._norm(x), x.dtype)
    return output * weight

class TransformerBlock(nn.Module):
  num_heads: int
  hidden_size: int
  ffn_multiplier: int = 2
  skip_connection: bool = False
  use_attn_norm: bool = True
  use_ffn_norm: bool = True
  use_conditional_norm: bool = False

  def setup(self):
    self.attn = GroupQueryFlashAttention(
      num_heads=self.num_heads,
      head_dim=self.hidden_size,
      use_conditional_norm=self.use_conditional_norm,
      cond_norm_hidden_size=4,
    )

  @nn.compact
  def __call__(self, x, condition=None, skip=None):
    input_shape = x.shape[-1]
    if self.skip_connection and skip is not None:
      x = jnp.concatenate([x, skip], axis=-1)
      x = nn.Dense(input_shape)(x)

    h = x if not self.use_attn_norm else RMSNorm()(x)
    h = x + self.attn(h, condition=condition)
    h = h if not self.use_ffn_norm else RMSNorm()(h)
    out = h + FFN(
      output_size=input_shape, ffn_hidden_size=(self.hidden_size * self.ffn_multiplier),
      use_conditional_norm=self.use_conditional_norm, cond_norm_hidden_size=4,
    )(h, condition=condition)

    return out

class Transformer(nn.Module):
  output_size: int
  patch_size: int
  hidden_size: int
  num_layers: int
  num_heads: int
  use_long_range_skip: bool = True

  def setup(self):
    self.encoder_layers = [TransformerBlock(hidden_size=self.hidden_size, num_heads=self.num_heads, skip_connection=False) for _ in range(self.num_layers // 2)]
    self.middle_layer = TransformerBlock(hidden_size=self.hidden_size, num_heads=self.num_heads, skip_connection=False) if (self.num_layers % 2 == 1) else None
    self.decoder_layers = [TransformerBlock(hidden_size=self.hidden_size, num_heads=self.num_heads, skip_connection=True) for _ in range(self.num_layers // 2)]

  @nn.compact
  def __call__(self, x, condition):
    if x.shape[-1] != self.hidden_size:
      x = nn.Dense(self.hidden_size)(x)
    skips = []

    for layer in self.encoder_layers:
      x = layer(x, condition=condition)
      skips.append(x)

    if self.middle_layer is not None:
      x = self.middle_layer(x, condition=condition)

    for layer in self.decoder_layers:
      skip = skips.pop() if self.use_long_range_skip else None
      x = layer(x, condition=condition, skip=skip)

    if x.shape[-1] != self.output_size:
      x = nn.Dense(self.output_size)(x)

    return x

class Processor(nn.Module):
  """Transformer-based processor for regional grid refinement.
  
  Applies transformer layers on regional node features arranged as a spatial grid,
  using patch-wise processing with positional embeddings. Features are grouped into
  patches, processed through transformer blocks, then reconstructed.
  
  Attributes:
      gridres: Grid resolution as (width, height) tuple
      patch_size: Patch size for grouping features
      hidden_size: Transformer hidden dimension
      num_layers: Number of transformer layers
      num_heads: Number of attention heads
      conditioned_normalization: Use lead-time conditioning (default: True)
      cond_norm_hidden_size: Conditional norm hidden size (default: True)
  """
  gridres: Tuple
  patch_size: int
  hidden_size: int
  num_layers: int
  num_heads: int
  conditioned_normalization: bool = True
  cond_norm_hidden_size: bool = True

  @nn.compact
  def __call__(self,
    rnode_features: Array,
    tau: Union[None, float],
    deterministic: bool = False,
  ) -> Array:
    """Apply transformer processing on regional grid features.
    
    Args:
        rnode_features: Regional features on grid [batch, width, height, feat]
        tau: Lead time parameter for conditional normalization
        deterministic: If True, disable dropout (default: False)
    
    Returns:
        Processed features with same shape [batch, width, height, feat]
    """
    # Get spatial and feature dimensions
    B, W, H, C = rnode_features.shape
    P = self.patch_size

    # Reshape grid to patches: [bsz, W, H, C] -> [bsz, (W/P)*(H/P), P*P*C]
    rnode_features = rnode_features.reshape(B, (W // P), P, (H // P), P, C)
    rnode_features = jnp.permute_dims(rnode_features, axes=(0, 1, 3, 2, 4, 5))
    patch_features = rnode_features.reshape(B, (W // P) * (H // P), P * P * C)

    # Apply linear transformation to patches
    patch_features = nn.Dense(patch_features.shape[-1])(patch_features)

    # Compute positional embeddings for each patch
    pos = jnp.stack(jnp.meshgrid(jnp.arange((W // P)), jnp.arange((H // P)), indexing='ij'), axis=-1).reshape(-1, 2).astype(jnp.float32)
    pos_emb = self._compute_absolute_embeddings(pos, (P * P * C))
    pos_emb = jnp.tile(pos_emb[None, :, :], reps=(B, 1, 1))
    # Add positional embeddings to patch features
    patch_features += pos_emb

    # Apply transformer blocks on patches
    patch_features = Transformer(output_size=(P * P * C), patch_size=self.patch_size, hidden_size=self.hidden_size, num_layers=self.num_layers, num_heads=self.num_heads)(x=patch_features, condition=tau)

    # Reshape patches back to grid: [bsz, (W/P)*(H/P), P*P*C] -> [bsz, W, H, C]
    rnode_features = patch_features.reshape(B, (W // P), (H // P), P, P, C)
    rnode_features = jnp.permute_dims(rnode_features, axes=(0, 1, 3, 2, 4, 5))
    rnode_features = rnode_features.reshape(B, W, H, C)

    return rnode_features

  def _compute_absolute_embeddings(self, positions, embed_dim):
    """Compute sinusoidal positional embeddings for patch locations.
    
    Args:
        positions: Grid positions [num_patches, num_pos_dims]
        embed_dim: Embedding dimension
    
    Returns:
        Positional embeddings [num_patches, embed_dim]
    """
    # Split embedding dimension across spatial dimensions
    num_pos_dims = positions.shape[1]
    dim_touse = embed_dim // (2 * num_pos_dims)
    # Create frequency bands for sinusoidal encoding
    freq_seq = jnp.arange(dim_touse, dtype=jnp.float32)
    inv_freq = 1.0 / (10000 ** (freq_seq / dim_touse))
    # Compute sinusoid inputs from positions and frequencies
    sinusoid_inp = positions[:, :, None] * inv_freq[None, None, :]
    # Apply sin and cos transformations
    pos_emb = jnp.concatenate([jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)
    # Flatten to embedding dimension
    pos_emb = pos_emb.reshape(positions.shape[0], -1)
    return pos_emb

class GAOT(AbstractOperator):
  """Graph-based Attention Operator Transformer neural operator.
  
  Multi-scale architecture using attention-based graph neural operators:
  (1) Encoder: Lifts physical features and aggregates to coarse regional nodes (p2r)
  (2) Processor: Applies transformer layers on regional grid
  (3) Decoder: Aggregates processed regional features back to physical nodes (r2p)
  
  Supports optional time-dependence with lead-time conditioning.
  
  Attributes:
      num_outputs: Output feature dimension (number of output variables)
      gridres: Regional grid resolution as (height, width) tuple
      patch_size: Patch size for regional grid processing
      transformer_hidden_size: Transformer hidden dimension (default: 256)
      processor_steps: Number of transformer layers in processor (default: 5)
      processor_attn_heads: Number of attention heads (default: 1)
      latent_size: Latent feature dimension (default: 128)
      mlp_hidden_layers: Hidden layers in feature MLPs (default: 1)
      p_edge_masking: Edge dropout rate (default: 0.5)
      tdep: Whether to condition on time/lead-time (default: False)
  """

  num_outputs: int
  gridres: Tuple
  patch_size: int
  transformer_hidden_size: int = 256
  processor_steps: int = 5
  processor_attn_heads: int = 1
  latent_size: int = 128
  mlp_hidden_layers: int = 1
  p_edge_masking: int = 0.5
  tdep: bool = False

  def setup(self):
    # NOTE: variable_mesh=True allows input and output meshes to be different
    self.variable_mesh = False

    # Encoder: physical -> regional (p2r)
    self.encoder = Encoder(
      latent_size=self.latent_size,
      output_size=self.latent_size,
      mlp_hidden_layers=self.mlp_hidden_layers,
      use_layer_norm=True,
      conditioned_normalization=self.tdep,
      cond_norm_hidden_size=4,
      p_edge_masking=self.p_edge_masking,
      dropout=.0,
      name='encoder',
    )

    # Processor: regional -> regional (r2r) transformer blocks
    self.processor = Processor(
      gridres=self.gridres,
      patch_size=self.patch_size,
      hidden_size=self.transformer_hidden_size,
      num_layers=self.processor_steps,
      num_heads=self.processor_attn_heads,
      conditioned_normalization=self.tdep,
      cond_norm_hidden_size=4,
      name='processor',
    )

    # Decoder: regional -> physical (r2p)
    self.decoder = Decoder(
      latent_size=self.latent_size,
      output_size=self.num_outputs,
      mlp_hidden_layers=self.mlp_hidden_layers,
      use_layer_norm=True,
      conditioned_normalization=self.tdep,
      cond_norm_hidden_size=4,
      p_edge_masking=self.p_edge_masking,
      dropout=.0,
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
        graphs: GraphSet containing p2r and r2p graphs
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
    self.sow(col='intermediates', name='rnodes-encoded', value=self._prepare_features(latent_rnode_features[:, :-1]))

    # Reshape regional nodes from flattened to grid for transformer processing
    latent_rnode_features_on_grid = latent_rnode_features[:, :-1, :].reshape(latent_rnode_features.shape[0], *self.gridres, latent_rnode_features.shape[2])

    # Process: Apply transformer layers on regional grid
    processed_rnode_features_on_grid = self.processor(latent_rnode_features_on_grid, tau, deterministic=deterministic)
    # Reshape back from grid to flattened representation
    processed_rnode_features = processed_rnode_features_on_grid.reshape(latent_rnode_features.shape[0], latent_rnode_features.shape[1]-1, latent_rnode_features.shape[2])
    # Append dummy node features
    processed_rnode_features = jnp.concatenate([processed_rnode_features, latent_rnode_features[:, [-1], :]], axis=1)
    self.sow(col='intermediates', name='rnodes-processed', value=self._prepare_features(processed_rnode_features[:, :-1]))

    # Decode: Aggregate regional nodes back to physical nodes
    output_pnode_features = self.decoder(graphs.r2p, processed_rnode_features, latent_pnode_features, tau, deterministic=deterministic)
    self.sow(col='intermediates', name='pnodes-decoded', value=self._prepare_features(output_pnode_features[:, :-1]))

    # Remove dummy node features
    output_pnode_features = output_pnode_features[:, :-1, :]

    return output_pnode_features

  def call(self, inputs: Inputs, graphs: GraphSet, input_pnode_features: Array = None, input_rnode_features: Array = None, deterministic: bool = False) -> Array:
    """Forward pass through the multi-scale operator.
    
    Args:
        inputs: Inputs NamedTuple with all operator inputs
        graphs: GraphSet containing p2r and r2p graphs
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

class XGAOT(AbstractOperator):
  """Extended GAOT with boundary condition incorporation via cross-attention.
  
  Combines GAOT (Graph-based Attention Operator Transformer) with a
  CrossAttentionExtender module to incorporate boundary conditions. Boundary
  functions can condition regional node features (use_extender=True) or
  physical node features (use_extender=False).
  
  Attributes:
      configs_core: Configuration dict for GAOT operator (num_outputs, gridres, etc.)
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
    """Initialize GAOT operator and CrossAttentionExtender modules."""
    # Instantiate core GAOT operator with transformer-based processor
    self.operator = GAOT(**self.configs_core)
    # Instantiate cross-attention based boundary extender
    self.extender = CrossAttentionExtender(**self.configs_extender)

  @nn.compact
  def call(self, inputs: Inputs, graphs: GraphSet, deterministic: bool = False):
    """Forward pass with boundary condition incorporation.
    
    Args:
        inputs: Inputs NamedTuple with domain features (s, a), boundary functions (q),
            boundary masks (m), and spatial coordinates (x_inp, x_out)
        graphs: GraphSet containing p2r and r2p graph connectivity
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
