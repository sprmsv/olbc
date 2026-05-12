"""Boundary function extenders and attention mechanisms.

This module provides classes for extending boundary conditions to domain
functions and cross-attention modules used in neural operator architectures.
"""

import jax.numpy as jnp
from einops import rearrange, repeat
from flax import linen as nn

from ol.utils import Array, shuffle_arrays


class Extender(nn.Module):
  """Abstract base class for boundary function extenders.
  
  Extenders project boundary condition functions to full domain functions,
  allowing the neural operator to utilize boundary information to refine
  predictions in the domain interior.
  """

  def setup(self):
    """Initialize extender components. Override in subclasses."""
    raise NotImplementedError

  def __call__(self,
    f_boundary: Array,
    f_domain: Array,
    **kwargs,
  ) -> Array:
    """Apply boundary function extension.
    
    Args:
        f_boundary: Boundary condition function values
        f_domain: Domain function values
        **kwargs: Additional arguments for extension
    
    Returns:
        Extended/modified domain function values
    """
    return self.call(f_boundary, f_domain, **kwargs)

  def call(self, f_boundary, f_domain, **kwargs) -> Array:
    """Extender-specific implementation. Override in subclasses.
    
    Args:
        f_boundary: Boundary condition function values
        f_domain: Domain function values
        **kwargs: Additional arguments
    
    Returns:
        Extended domain function values
    """
    raise NotImplementedError

  @property
  def configs(self):
    """Get configuration parameters from class attributes.
    
    Returns:
        Dict mapping attribute names to their values (excluding 'parent')
    """
    configs = {
      attr: self.__getattr__(attr)
      for attr in self.__annotations__.keys() if attr != 'parent'
    }
    return configs

class FeedForward(nn.Module):
  """Multi-layer feed-forward network with activation and dropout.
  
  Attributes:
      features: Output feature dimension (default: input dim if None)
      mult: Hidden dimension multiplier (default: 4)
      dropout: Dropout rate (default: 0.0)
  """

  features: int = None
  mult: int = 4
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=False):
    """Apply feed-forward transformation.
    
    Args:
        x: Input features [batch, seq, feat]
        deterministic: If True, disable dropout
    
    Returns:
        Transformed features with same shape as input
    """
    # Use input dimension if output dimension not specified
    features = self.features if (self.features is not None) else x.shape[-1]
    # Hidden layer with activation
    x = nn.Dense(features * self.mult)(x)
    x = nn.swish(x)
    # Apply dropout for regularization
    x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
    # Project back to output dimension
    x = nn.Dense(features)(x)
    return x

class Attention(nn.Module):
  """Multi-head scaled dot-product attention mechanism.
  
  Attributes:
      heads: Number of attention heads (default: 8)
      head_dim: Dimension per head (default: 64)
      dropout: Dropout rate (default: 0.0)
  """

  heads: int = 8
  head_dim: int = 64
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, context=None, mask=None, deterministic=False):
    """Apply multi-head attention.
    
    Args:
        x: Query input [batch, seq_x, feat]
        context: Key/value context (default: use x). [batch, seq_ctx, feat]
        mask: Attention mask [batch, seq_ctx]
        deterministic: If True, disable dropout
    
    Returns:
        Attention output [batch, seq_x, feat]
    """
    # Compute query projections
    q = nn.Dense((self.heads * self.head_dim), use_bias=False)(x)
    # Compute key and value projections from context
    k, v = jnp.split(nn.Dense((self.heads * self.head_dim * 2), use_bias=False)(context if context is not None else x), 2, axis=-1)
    # Reshape to multi-head format [batch*heads, seq, head_dim]
    q, k, v = map(lambda arr: rearrange(arr, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v),)

    # Compute scaled dot-product attention scores
    sim = jnp.einsum('b i d, b j d -> b i j', q, k) * self.head_dim ** -0.5
    # Expand mask to multi-head format if provided
    mask = repeat(mask, 'b m -> (b h) n m', h=self.heads, n=sim.shape[1]) if (mask is not None) else None
    # Apply softmax normalization with mask
    attn = nn.softmax(sim, where=mask.astype(jnp.bool), axis=-1)
    # Compute average attention for intermediate logging
    attn_avg = rearrange(attn, '(b h) i j -> b h i j', h=self.heads).mean(axis=1)
    self.sow(col='intermediates', name='scores', value=attn_avg)

    # Apply attention weights to values
    out = jnp.einsum('b i j, b j d -> b i d', attn, v)
    # Reshape back to original format
    out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
    # Final projection and dropout
    out = nn.Dense(x.shape[-1])(out)
    out = nn.Dropout(self.dropout)(out, deterministic=deterministic)

    return out

class CrossAttentionExtender(Extender):
  """Cross-attention based boundary function extender.
  
  Uses multi-head cross-attention to condition domain predictions on boundary
  values. Incorporates learned spatial embeddings and residual connections.
  
  Attributes:
      out_dim: Output feature dimension (default: 4)
      latent_dim: Latent feature dimension (default: 16)
      depth: Number of cross-attention layers (default: 4)
      n_heads: Number of attention heads (default: 2)
      ff_mult: Feed-forward hidden dimension multiplier (default: 1)
      p_masking: Boundary node masking rate (default: 0.0)
      attn_dropout: Attention dropout rate (default: 0.0)
      ff_dropout: Feed-forward dropout rate (default: 0.0)
  """

  out_dim: int = 4
  latent_dim: int = 16
  depth: int = 4
  n_heads: int = 2
  ff_mult: int = 1
  p_masking: float = 0.0
  attn_dropout: float = 0.0
  ff_dropout: float = 0.0

  def setup(self):
    # Initial embedding layers for boundary and domain features
    self.ff_initial_boundary = nn.Sequential([
      FeedForward(mult=self.ff_mult, dropout=self.ff_dropout, features=self.latent_dim),
      nn.LayerNorm(),
    ], name='ff_initial_boundary')
    self.ff_initial_domain = nn.Sequential([
      FeedForward(mult=self.ff_mult, dropout=self.ff_dropout, features=self.latent_dim),
      nn.LayerNorm(),
    ], name='ff_initial_domain')
    # Attention and feed-forward layers for depth iterations
    self.attention_layers = [
      Attention(heads=self.n_heads, head_dim=self.latent_dim, dropout=self.attn_dropout, name=f'attention_{i}')
      for i in range(self.depth)
    ]
    self.attention_lns = [nn.LayerNorm() for _ in range(self.depth)]
    self.ff_layers = [
      FeedForward(mult=self.ff_mult, dropout=self.ff_dropout, name=f'ff_{i}')
      for i in range(self.depth)
    ]
    self.ff_lns = [nn.LayerNorm() for _ in range(self.depth)]
    # Final projection to output dimension
    self.ff_final = FeedForward(mult=self.ff_mult, dropout=self.ff_dropout, features=self.out_dim, name='ff_final')

  def call(self, f_boundary: Array, f_domain: Array, m_boundary: Array = None, deterministic: bool = False) -> Array:
    """Apply cross-attention extension to domain from boundary conditions.
    
    Args:
        f_boundary: Boundary function values [batch, num_boundary, feat_b]
        f_domain: Domain function values [batch, num_domain, feat_d]
        m_boundary: Boundary mask indicating valid nodes [batch, num_boundary]
        deterministic: If True, disable dropout and masking
    
    Returns:
        Extended domain values [batch, num_domain, out_dim]
    """
    # Embed boundary and domain features into latent space
    f_boundary = self.ff_initial_boundary(f_boundary, deterministic=deterministic)
    f_domain = self.ff_initial_domain(f_domain, deterministic=deterministic)

    # Cross-attention refinement blocks
    for i in range(self.depth):
      # Optionally mask boundary nodes during training for regularization
      if deterministic:
        _f_boundary, _m_boundary = f_boundary, m_boundary
      else:
        # Randomly shuffle and mask boundary nodes
        rngkey = self.make_rng('masking')
        size_boundary_masked = int(f_boundary.shape[1] * (1 - self.p_masking))
        _f_boundary, _m_boundary = shuffle_arrays(rngkey=rngkey, arrays=[f_boundary, m_boundary], axis=1)
        _f_boundary, _m_boundary = _f_boundary[:, :size_boundary_masked], _m_boundary[:, :size_boundary_masked]
      # Cross-attention: condition domain with boundary information
      f_domain += self.attention_layers[i](f_domain, _f_boundary, _m_boundary, deterministic=deterministic)
      f_domain = self.attention_lns[i](f_domain)
      # Feed-forward refinement with residual connection
      f_domain += self.ff_layers[i](f_domain, deterministic=deterministic)
      f_domain = self.ff_lns[i](f_domain)

    # Project output to desired dimension
    f_domain = self.ff_final(f_domain, deterministic=deterministic)

    return f_domain
