"""Stepper base class for operator application and input normalization.

This module provides utilities for normalizing inputs using dataset statistics
and handling boundary condition merging in a standardized way.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from ol.dataset.dataset import StatsCollectionType
from ol.models.common import AbstractOperator, Inputs
from ol.utils import Array, normalize, unnormalize


class Stepper(ABC):
  """Base class for normalizing inputs and applying neural operators.
  
  Provides interface for input normalization using dataset statistics and
  boundary condition handling including Dirichlet, Neumann, and Robin types.
  """

  def __init__(self, operator: AbstractOperator):
    self._apply_operator = operator.apply

  def normalize_inputs(self, stats: StatsCollectionType, inputs: Inputs) -> Inputs:
    """Normalize inputs using dataset statistics.
    
    Scales coordinates to [-1, 1] and normalizes all function channels using
    the provided statistics.
    """

    # Coordinates
    x_inp_nrm = 2 * ((inputs.x_inp - stats['x'].min) / (stats['x'].max - stats['x'].min)) - 1
    x_out_nrm = 2 * ((inputs.x_out - stats['x'].min) / (stats['x'].max - stats['x'].min)) - 1
    # Domain functions
    s_nrm = normalize(inputs.s, shift=stats['geo'].mean, scale=stats['geo'].std)
    a_nrm = normalize(inputs.a, shift=stats['dom'].mean, scale=stats['dom'].std)
    # Segmented functions except Robin-type boundary conditions
    q_nrm = {
      key: normalize(inputs.q[key], shift=stats['seg'][key].mean, scale=stats['seg'][key].std)
      for key in inputs.q.keys() if not key.startswith('bc')
    }
    m_nrm = {
      key: inputs.m[key]
      for key in inputs.q.keys() if not key.startswith('bc')
    }
    # Robin-type boundary conditions
    for dim in set([key.split('-')[1] for key in inputs.q.keys() if key.startswith('bc')]):
      q_bcs_dim, m_bcs_dim = self.merge_normalize_bcs(
        gamma_D=(inputs.q[f'bc-{dim}-dir'][..., 0] if f'bc-{dim}-dir' in inputs.q else None),
        mask_D=(inputs.m[f'bc-{dim}-dir'] if f'bc-{dim}-dir' in inputs.m else None),
        mean_D=(stats['seg'][f'bc-{dim}-dir'].mean[..., 0] if f'bc-{dim}-dir' in stats['seg'] else None),
        std_D=(stats['seg'][f'bc-{dim}-dir'].std[..., 0] if f'bc-{dim}-dir' in stats['seg'] else None),
        gamma_N=(inputs.q[f'bc-{dim}-neu'][..., 0] if f'bc-{dim}-neu' in inputs.q else None),
        mask_N=(inputs.m[f'bc-{dim}-neu'] if f'bc-{dim}-neu' in inputs.m else None),
        mean_N=(stats['seg'][f'bc-{dim}-neu'].mean[..., 0] if f'bc-{dim}-neu' in stats['seg'] else None),
        std_N=(stats['seg'][f'bc-{dim}-neu'].std[..., 0] if f'bc-{dim}-neu' in stats['seg'] else None),
        alpha_R=(inputs.q[f'bc-{dim}-rob'][..., 0] if f'bc-{dim}-rob' in inputs.q else None),
        gamma_R=(inputs.q[f'bc-{dim}-rob'][..., 1] if f'bc-{dim}-rob' in inputs.q else None),
        mask_R=(inputs.m[f'bc-{dim}-rob'] if f'bc-{dim}-rob' in inputs.m else None),
        mean_R=(stats['seg'][f'bc-{dim}-rob'].mean[..., 0] if f'bc-{dim}-rob' in stats['seg'] else None),
        std_R=(stats['seg'][f'bc-{dim}-rob'].std[..., 0] if f'bc-{dim}-rob' in stats['seg'] else None),
        shape=inputs.x_inp.shape[:3],
      )
      q_nrm[f'bc-{dim}'] = q_bcs_dim
      m_nrm[f'bc-{dim}'] = m_bcs_dim

    # Time inputs
    if inputs.t is None:
      t_nrm = None
    else:
      t_nrm = (inputs.t - stats['t'].min) / (stats['t'].max - stats['t'].min)
    if inputs.tau is None:
      tau_nrm = None
    else:
      tau_nrm = (inputs.tau) / (stats['t'].max - stats['t'].min)

    inputs_nrm = Inputs(
      s=s_nrm,
      a=a_nrm,
      q=q_nrm,
      m=m_nrm,
      x_inp=x_inp_nrm,
      x_out=x_out_nrm,
      t=t_nrm,
      tau=tau_nrm,
    )

    return inputs_nrm

  def merge_normalize_bcs(self, gamma_D, mask_D, mean_D, std_D, gamma_N, mask_N, mean_N, std_N, alpha_R, gamma_R, mask_R, mean_R, std_R, shape):
    """Merge and normalize Dirichlet, Neumann, and Robin boundary conditions.
    
    Returns normalized boundary condition coefficients and a mask indicating
    active boundary regions.
    """

    # Set fallback statistics
    if mean_D is None:
      assert std_D is None
      mean_D = 0.0
      std_D = 1.0
    if mean_N is None:
      assert std_N is None
      mean_N = 0.0 if mean_R is None else mean_R
      std_N = 1.0 if std_R is None else std_R

    # Set fallback masks
    if mask_D is None:
      mask_D = jnp.full(shape=shape, fill_value=False)
    if mask_N is None:
      mask_N = jnp.full(shape=shape, fill_value=False)
    if mask_R is None:
      mask_R = jnp.full(shape=shape, fill_value=False)

    # Normalize input functions with the given statistics
    if gamma_D is not None:
      gamma_D = normalize(gamma_D, shift=mean_D, scale=std_D)
    else:
      gamma_D = jnp.zeros(shape=shape)
    if gamma_N is not None:
      gamma_N = normalize(gamma_N, shift=mean_N, scale=std_N)
    else:
      gamma_N = jnp.zeros(shape=shape)
    if gamma_R is not None:
      gamma_R = gamma_R - alpha_R * mean_D - mean_N
      alpha_R = alpha_R * std_D
      beta_R = std_N
      norm_R = jnp.sqrt(alpha_R**2 + beta_R**2)
      alpha_R = normalize(alpha_R, shift=0.0, scale=norm_R)
      beta_R = normalize(beta_R, shift=0.0, scale=norm_R)
      gamma_R = normalize(gamma_R, shift=0.0, scale=norm_R)
    else:
      alpha_R = jnp.zeros(shape=shape)
      beta_R = jnp.zeros(shape=shape)
      gamma_R = jnp.zeros(shape=shape)

    # Get Dirichlet coefficients
    alpha_D = jnp.ones(shape=shape)
    beta_D = jnp.zeros(shape=shape)
    # Get Neumann coefficients
    alpha_N = jnp.zeros(shape=shape)
    beta_N = jnp.ones(shape=shape)

    # Assemble the masks
    mask = jnp.any(jnp.stack([mask_D, mask_N, mask_R], axis=-1), axis=-1)
    # Assemble the functions
    alpha = jnp.sum(jnp.stack([alpha_D*mask_D, alpha_N*mask_N, alpha_R*mask_R], axis=-1), axis=-1)
    beta = jnp.sum(jnp.stack([beta_D*mask_D, beta_N*mask_N, beta_R*mask_R], axis=-1), axis=-1)
    gamma = jnp.sum(jnp.stack([gamma_D*mask_D, gamma_N*mask_N, gamma_R*mask_R], axis=-1), axis=-1)
    # Concatenate
    bcs = jnp.stack([alpha, beta, gamma], axis=-1)

    return bcs, mask

  @abstractmethod
  def apply(self,
    variables,
    stats: StatsCollectionType,
    inputs: Inputs,
    **kwargs,
  ):
    """Normalize inputs and apply the operator.
    
    Args:
        variables: Model parameters.
        stats: Dataset statistics for normalization.
        inputs: Raw input data.
        **kwargs: Additional keyword arguments passed to the operator.
    """
    pass

  @abstractmethod
  def get_loss_inputs(self,
    variables,
    stats: StatsCollectionType,
    inputs: Inputs,
    **kwargs,
  ):
    """
    Calculates prediction and target variables, ready to be given as input to the loss function.

    t_inp is the time of the input and must be a non-negative integer.
    tau is the time difference and must be an integer greater than zero.
    """
    pass

  def get_intermediates(self,
    variables,
    stats: StatsCollectionType,
    inputs: Inputs,
    **kwargs,
  ):
    # Normalize inputs
    inputs_nrm = self.normalize_inputs(stats, inputs)

    # Get predicted normalized derivatives
    _, state = self._apply_operator(
      variables,
      inputs=inputs_nrm,
      capture_intermediates=(lambda mdl, method_name: False), # Only get the registered intermediates
      **kwargs,
    )

    return state['intermediates']

class OutputStepper(Stepper):

  def apply(self,
    variables,
    stats: StatsCollectionType,
    inputs: Inputs,
    **kwargs,
  ):
    """
    Normalizes raw inputs and applies the operator on it.

    t_inp is the time of the input and must be a non-negative integer.
    tau is the time difference and must be an integer greater than zero.
    """

    # Normalize inputs
    inputs_nrm = self.normalize_inputs(stats, inputs)

    # Get predicted normalized output
    u_prd_nrm = self._apply_operator(
      variables,
      inputs=inputs_nrm,
      **kwargs,
    )

    # Unnormalize predicted output
    u_prd = unnormalize(
      u_prd_nrm,
      mean=stats['out'].mean,
      std=stats['out'].std,
    )

    return u_prd

  def get_loss_inputs(self,
    variables,
    stats: StatsCollectionType,
    u_tgt: Array,
    inputs: Inputs,
    **kwargs,
  ):
    """
    Calculates prediction and target variables, ready to be given as input to the loss function.

    t_inp is the time of the input and must be a non-negative integer.
    tau is the time difference and must be an integer greater than zero.
    """

    # Normalize inputs
    inputs_nrm = self.normalize_inputs(stats, inputs)

    # Get predicted normalized output
    u_prd_nrm = self._apply_operator(
      variables,
      inputs=inputs_nrm,
      **kwargs,
    )

    # Get target normalized output
    u_tgt_nrm = normalize(
      u_tgt,
      shift=stats['out'].mean,
      scale=stats['out'].std,
    )

    return (u_tgt_nrm, u_prd_nrm)
