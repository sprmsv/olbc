"""Training loop and experiment management.

This module implements the main training loop.
It is designed to be run as a script with command-line arguments.
"""

import argparse
import json
import pickle
import functools
from datetime import datetime
from time import time
from typing import Tuple, Any, Mapping, Iterable, Callable

import flax.typing
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import replicate, unreplicate
from matplotlib import pyplot as plt

from ol import models
from ol.dataset.dataset import Dataset, Batch, StatsCollectionType
from ol.experiments import DIR_EXPERIMENTS
from ol.graph.graphbuilder import GraphBuilder
from ol.metrics import BatchMetrics, Metrics, EvalMetrics
from ol.metrics import rel_lp_loss
from ol.metrics import mse_error, rel_lp_error_mean, recall_tol_mean, chamfer_mean
from ol.models.common import AbstractOperator, Inputs
from ol.stepping import OutputStepper, Stepper
from ol.utils import log, Array, Tree
from ol.test import RECALL_TOLERANCE, EXCLUDE_PERCENTILE


NUM_DEVICES = jax.local_device_count()
FLAGS = None
EXCLUDE_PERCENTILE_STATS = 0.4
EXCLUDE_PERCENTILE_LOSS = 0.2

def define_flags():
  parser = argparse.ArgumentParser()

  # FLAGS::general
  parser.add_argument('--exp', type=str, default='000',
    help='Name of the experiment')
  parser.add_argument('--datetime', type=str, default=None,
    help='A string representing the current datetime')
  parser.add_argument('--datadir', type=str, required=True,
    help='Path of the folder containing the datasets')
  parser.add_argument('--datapath', type=str, required=True,
    help='Relative path inside the data directory')
  parser.add_argument('--params', type=str, default=None,
    help='Path of the previous experiment containing the initial parameters')
  parser.add_argument('--seed', type=int, default=45,
    help='Seed for random number generator')
  parser.add_argument('--space_downsample_factor', type=float, default=1.0,
    help='Factor for downsampling the space resolution (including the boundaries)')
  parser.add_argument('--boundary_downsample_factor', type=float, default=1.0,
    help='Factor for downsampling the resolution of the boundary functions')
  parser.add_argument('--use_extensions', action=argparse.BooleanOptionalAction, default=False,
    help='Wether to use pre-computed harminic extensions as input domain functions')

  # FLAGS::training
  parser.add_argument('--batch_size', type=int, default=2,
    help='Size of a batch of training samples per device and per optimization step (input batch size)')
  parser.add_argument('--superbatch_size', type=int, default=1,
    help='Number of batches to load and process together')
  parser.add_argument('--superbatch_repeats', type=int, default=1,
    help='Number of repetitions of each super batch before moving on to the next')
  parser.add_argument('--epochs', type=int, default=20,
    help='Number of training epochs')
  parser.add_argument('--long_epoch_gap', type=int, default=5,
    help='Number of training epochs to wait before each evaluation')
  parser.add_argument('--lr_init', type=float, default=1e-05,
    help='Initial learning rate in the onecycle scheduler')
  parser.add_argument('--lr_peak', type=float, default=2e-04,
    help='Peak learning rate in the onecycle scheduler')
  parser.add_argument('--lr_base', type=float, default=1e-05,
    help='Final learning rate in the onecycle scheduler')
  parser.add_argument('--lr_lowr', type=float, default=1e-06,
    help='Final learning rate in the exponential decay')
  parser.add_argument('--gclip', type=float, default=1e-01,
    help='Norm for adaptive gradient clipping')
  parser.add_argument('--weight_decay', type=float, default=1e-04,
    help='weight decay for the optimizer')
  parser.add_argument('--n_train', type=int, default=(2**4),
    help='Number of training samples')
  parser.add_argument('--n_valid', type=int, default=(2**4),
    help='Number of validation samples')

  # FLAGS::core
  parser.add_argument('--core_name', type=str, default='XRIGNO',
    help='Name of the core neural operator architecture')
  parser.add_argument('--pmesh_subsample_factor', type=float, default=16.0,
    help='Factor for random subsampling of the physical mesh (ignored if rmesh_gridres is not None)')
  parser.add_argument('--rmesh_gridres', type=int, default=0,
    help='Grid resolution of rmesh (overrides pmesh_subsample_factor)')
  parser.add_argument('--rmesh_subsample_factor', type=float, default=2.0,
    help='Factor for random subsampling of hierarchical regional meshes')
  parser.add_argument('--overlap_factor_p2r', type=float, default=1.0,
    help='Overlap factor for p2r edges (encoder)')
  parser.add_argument('--overlap_factor_r2p', type=float, default=1.0,
    help='Overlap factor for r2p edges (decoder)')
  parser.add_argument('--rmesh_levels', type=int, default=10,
    help='Number of multimesh connection levels (processor)')
  parser.add_argument('--node_coordinate_freqs', type=int, default=4,
    help='Number of frequencies for encoding periodic node coordinates')
  parser.add_argument('--node_latent_size', type=int, default=4,
    help='Size of latent node features')
  parser.add_argument('--edge_latent_size', type=int, default=4,
    help='Size of latent edge features')
  parser.add_argument('--processor_steps', type=int, default=2,
    help='Number of message-passing steps in the processor')
  parser.add_argument('--p_edge_masking', type=float, default=0.5,
    help='Probability for random masking of the edges')

  # FLAGS::model::CrossAttentionExtender
  parser.add_argument('--extender', action=argparse.BooleanOptionalAction, default=True,
    help='Wether to use the extender module or not')
  parser.add_argument('--ext_depth', type=int, default=1,
    help='Number of attention blocks in the extender')
  parser.add_argument('--ext_vars', type=int, default=1,
    help='Number of output domain functions by the extender')
  parser.add_argument('--ext_latent_size', type=int, default=4,
    help='Dimension of the latent features after the initial extender feed-forward blocks')
  parser.add_argument('--ext_heads', type=int, default=1,
    help='Number of attention heads in each block of the extender')
  parser.add_argument('--ext_p_masking', type=float, default=0.0,
    help='Probabilty for random masking of the boundary nodes')

  return parser.parse_args()

def train(
  rngkey: flax.typing.PRNGKey,
  stepper: Stepper,
  state: TrainState,
  dataset: Dataset,
  graph_builder: GraphBuilder,
  epochs: int,
  loss_fn: Callable,
  epochs_before: int = 0,
) -> TrainState:
  """Trains a model and returns the state."""

  # Set constants
  num_samples_trn = (dataset.splits[0][1] - dataset.splits[0][0])
  num_times = dataset.metadata.shape[1]
  num_pnodes = dataset.metadata.shape[2]
  big_batch_size = FLAGS.superbatch_size * NUM_DEVICES * FLAGS.batch_size
  assert num_samples_trn % big_batch_size == 0

  # Store the initial time
  time_int_pre = time()

  # Replicate state, stats, and graphs
  # NOTE: Internally uses jax.device_put_replicate
  state = replicate(state)
  stats = replicate(dataset.stats)

  @functools.partial(jax.pmap, axis_name='device')
  def _train_one_batch(rngkey: flax.typing.PRNGKey, state: TrainState, stats: dict, batch: Batch) -> Tuple[TrainState, Array, Array]:
    """Loads a batch, normalizes it, updates the state based on it, and returns it."""

    # Split the rngkey
    subrngkeys = jax.random.split(rngkey, num=3)

    # Define helper function for updating the state per sub-batch
    # NOTE: Each batch of trajectories (time-dependent) can result in multiple sub-batches of input-output pairs
    def _update_state_per_subbatch(state: TrainState,
      q: Array, u: Array, h: Array, m: Array, x: Array, t: Array, tau: Array, r: Array, g,
    ) -> Tuple[TrainState, Array, Tree]:
      # NOTE: INPUT SHAPES [superbatch_size*batch_size, ...]

      # Get random number generators
      subsubrngkey = jax.random.fold_in(subrngkeys[0], data=state.step)
      subsubsubrngkeys = jax.random.split(subsubrngkey, num=3)
      rngs = {'dropout': subsubsubrngkeys[0], 'masking': subsubsubrngkeys[1], 'other': subsubsubrngkeys[2]}

      def _compute_loss(params: flax.typing.Collection,
        q: Array, u: Array, h: Array, m: Array, x: Array, t: Array, tau: Array, r: Array, g,
      ) -> Array:
        """Computes the prediction of the model and returns its loss."""
        inputs = Inputs(s=q, a=u, q=h, m=m, x_inp=x, x_out=x, t=t, tau=tau)
        _loss_inputs = stepper.get_loss_inputs(
          variables={'params': params},
          stats=stats,
          u_tgt=r,
          inputs=inputs,
          graphs=graph_builder.build_graphs(g),
          deterministic=False,
          rngs=rngs,
        )
        return loss_fn(*_loss_inputs)

      # Get loss and gradients w.r.t. the parameters
      _loss, _grads = jax.value_and_grad(_compute_loss)(state.params, q, u, h, m, x, t, tau, r, g)
      # Synchronize loss and gradients
      loss = jax.lax.pmean(_loss, axis_name='device')
      grads = jax.lax.pmean(_grads, axis_name='device')
      # Apply gradients
      state = state.apply_gradients(grads=grads)

      return state, loss, grads

    # Shuffle along the space axis
    batch = batch.shuffled(rngkey=subrngkeys[1])

    # Prepare input-output pairs
    # -> [1, superbatch_size*batch_size, ...]
    num_valid_pairs = 1  # NOTE: Relevant for all2all training of time-dependent datasets
    s_batch = jnp.concatenate([batch.functions[key].values[None] for key in dataset.metadata.geo], axis=-1)
    a_batch = jnp.concatenate([batch.functions[key].values[None] for key in dataset.metadata.dom], axis=-1)
    q_batch = {key: batch.functions[key].values[None] for key in dataset.metadata.seg}
    m_batch = {key: batch.functions[key].mask[None] for key in dataset.metadata.seg}
    x_batch = batch.x[None]
    r_batch = jnp.concatenate([batch.functions[key].values[None] for key in dataset.metadata.out], axis=-1)
    t_batch = None
    tau_batch = None
    g_batch = jax.tree.map(lambda arr: arr[None], batch.g)

    # Split a big batch into small batches
    # -> [superbatch_size, batch_size, ...]
    # NOTE: No need to do so with all2all training
    num_valid_pairs *= FLAGS.superbatch_size
    split_big_batch = lambda t: jax.tree.map(lambda arr: jnp.reshape(arr, shape=(FLAGS.superbatch_size, FLAGS.batch_size, *arr.shape[2:])), t)
    s_batch = split_big_batch(s_batch)
    a_batch = split_big_batch(a_batch)
    q_batch = split_big_batch(q_batch)
    m_batch = split_big_batch(m_batch)
    x_batch = split_big_batch(x_batch)
    r_batch = split_big_batch(r_batch)
    g_batch = split_big_batch(g_batch)

    # Repeat the super batch in the same epoch for faster training
    # NOTE: It is vital to repeat them in tiles or randomly (so that the SGD step sees different samples every time)
    num_valid_pairs *= FLAGS.superbatch_repeats
    repeat_batches = lambda t: jax.tree.map(lambda arr: jnp.tile(arr, reps=(FLAGS.superbatch_repeats, *([1]*(arr.ndim-1)))), t)
    s_batch = repeat_batches(s_batch)
    a_batch = repeat_batches(a_batch)
    q_batch = repeat_batches(q_batch)
    m_batch = repeat_batches(m_batch)
    x_batch = repeat_batches(x_batch)
    r_batch = repeat_batches(r_batch)
    g_batch = repeat_batches(g_batch)
    # Shuffle the repeated big batch
    shuffle_batches = lambda t: jax.tree.map(lambda arr: jax.random.permutation(subrngkeys[2], arr.reshape(-1, *arr.shape[2:])).reshape(*arr.shape), t)
    s_batch = shuffle_batches(s_batch)
    a_batch = shuffle_batches(a_batch)
    q_batch = shuffle_batches(q_batch)
    m_batch = shuffle_batches(m_batch)
    x_batch = shuffle_batches(x_batch)
    r_batch = shuffle_batches(r_batch)
    g_batch = shuffle_batches(g_batch)

    # Define helper function for updating the state
    def _update_state(i, carry):
      # Update state, loss, and gradients
      _state, _loss_carried, _grads_carried = carry
      _state, _loss_subbatch, _grads_subbatch = _update_state_per_subbatch(
        state=_state,
        q=s_batch[i],
        u=a_batch[i],
        h=jax.tree.map(lambda h: h[i], q_batch),
        m=jax.tree.map(lambda m: m[i], m_batch),
        x=x_batch[i],
        t=(t_batch[i] if (t_batch is not None) else None),
        tau=(tau_batch[i] if (tau_batch is not None) else None),
        r=r_batch[i],
        g=jax.tree.map(lambda arr: arr[i], g_batch)
      )
      # Update the carried loss and gradients of the subbatch
      _loss_updated = _loss_carried + _loss_subbatch / num_valid_pairs
      _grads_updated = jax.tree.map(
        lambda g_old, g_new: (g_old + g_new / num_valid_pairs),
        _grads_carried, _grads_subbatch,
      )

      return _state, _loss_updated, _grads_updated

    # Loop over the pairs
    _init_state = state
    _init_loss = 0.
    _init_grads = jax.tree.map(lambda p: jnp.zeros_like(p), state.params)
    state, loss, grads = jax.lax.fori_loop(
      lower=0,
      upper=num_valid_pairs,
      body_fun=_update_state,
      init_val=(_init_state, _init_loss, _init_grads)
    )

    return state, loss, grads

  def train_one_epoch(rngkey: flax.typing.PRNGKey, state: TrainState, batches: Iterable[Batch]) -> Tuple[TrainState, Array, Array]:
    """Updates the state based on accumulated losses and gradients."""

    # Split the rngkey
    subrngkeys = jax.random.split(rngkey, num=1)

    # Loop over the batches
    loss_epoch = 0.
    grad_epoch = 0.
    for i, batch in enumerate(batches):
      # Split the batch between devices
      # [superbatch_size*NUM_DEVICES*batch_size, ...] -> [NUM_DEVICES, superbatch_size*batch_size, ...]
      batch = Batch(
        x=shard(batch.x),
        t=shard(batch.t),
        g=shard(batch.g),
        functions=shard(batch.functions),
      )

      # Get loss and updated state
      subsubrngkey = jax.random.fold_in(subrngkeys[0], data=i)
      subsubrngkey = shard_prng_key(subsubrngkey)
      state, loss, grads = _train_one_batch(subsubrngkey, state, stats, batch)
      # NOTE: Using the first element of replicated loss and grads
      num_big_batches = num_samples_trn / big_batch_size
      loss_epoch += loss[0] / num_big_batches
      grad_epoch += np.mean(jax.tree.flatten(
        jax.tree.map(jnp.mean, jax.tree.map(lambda g: jnp.abs(g[0]), grads)))[0]) / num_big_batches

    return state, loss_epoch, grad_epoch

  @jax.pmap
  def _evaluate_one_batch(state: TrainState, stats: StatsCollectionType, batch: Batch) -> Mapping:

    u_tgt = jnp.concatenate([batch.functions[key].values[:, [0]] for key in dataset.metadata.out], axis=-1)
    u_prd = stepper.apply(
      variables={'params': state.params},
      stats=stats,
      inputs=Inputs(
        s=jnp.concatenate([batch.functions[key].values[:, [0]] for key in dataset.metadata.geo], axis=-1),
        a=jnp.concatenate([batch.functions[key].values[:, [0]] for key in dataset.metadata.dom], axis=-1),
        q={key: batch.functions[key].values[:, [0]] for key in dataset.metadata.seg},
        m={key: batch.functions[key].mask[:, [0]] for key in dataset.metadata.seg},
        x_inp=batch.x,
        x_out=batch.x,
      ),
      graphs=graph_builder.build_graphs(batch.g),
      deterministic=True,
    )

    # Calculate the errors
    bbox_x = jnp.array(dataset.metadata.bbox_x)
    char_length = jnp.linalg.norm(bbox_x[1, :]-bbox_x[0, :])
    tol = RECALL_TOLERANCE * char_length
    batch_metrics = BatchMetrics(
      mse=mse_error(u_tgt, u_prd),
      l1=rel_lp_error_mean(u_tgt, u_prd, p=1, exclude_percentile=EXCLUDE_PERCENTILE),
      l2=rel_lp_error_mean(u_tgt, u_prd, p=2, exclude_percentile=EXCLUDE_PERCENTILE),
      recall=recall_tol_mean(batch.x, u_tgt, u_prd, q=EXCLUDE_PERCENTILE, tol=tol),
      chamfer=chamfer_mean(batch.x, u_tgt, u_prd, q=EXCLUDE_PERCENTILE),
    )

    return batch_metrics.__dict__

  def evaluate(state: TrainState, batches: Iterable[Batch]) -> EvalMetrics:
    """Evaluates the model on a dataset based on multiple trajectory lengths."""

    metrics: list[BatchMetrics] = []

    for batch in batches:
      # Split the batch between devices
      # [NUM_DEVICES*batch_size, ...] -> [NUM_DEVICES, batch_size, ...]
      batch = Batch(
        x=shard(batch.x),
        t=shard(batch.t),
        g=shard(batch.g),
        functions=shard(batch.functions),
      )
      # Get evaluation metrics
      batch_metrics = _evaluate_one_batch(state, stats, batch)
      batch_metrics = BatchMetrics(**batch_metrics)
      # Re-arrange the sub-batches gotten from each device
      batch_metrics.reshape(shape=(FLAGS.batch_size * NUM_DEVICES, 1))
      # Append the errors to the list
      metrics.append(batch_metrics)

    # Aggregate over the batch dimension and compute norm per variable
    metrics_med = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics]), axis=0).item(),
      recall=jnp.median(jnp.concatenate([m.recall for m in metrics]), axis=0).item(),
      chamfer=jnp.median(jnp.concatenate([m.chamfer for m in metrics]), axis=0).item(),
    )
    metrics_std = Metrics(
      mse=jnp.std(jnp.concatenate([m.mse for m in metrics]), axis=0).item(),
      l1=jnp.std(jnp.concatenate([m.l1 for m in metrics]), axis=0).item(),
      l2=jnp.std(jnp.concatenate([m.l2 for m in metrics]), axis=0).item(),
      recall=jnp.std(jnp.concatenate([m.recall for m in metrics]), axis=0).item(),
      chamfer=jnp.std(jnp.concatenate([m.chamfer for m in metrics]), axis=0).item(),
    )
    metrics_max = Metrics(
      mse=jnp.max(jnp.concatenate([m.mse for m in metrics]), axis=0).item(),
      l1=jnp.max(jnp.concatenate([m.l1 for m in metrics]), axis=0).item(),
      l2=jnp.max(jnp.concatenate([m.l2 for m in metrics]), axis=0).item(),
      recall=jnp.max(jnp.concatenate([m.recall for m in metrics]), axis=0).item(),
      chamfer=jnp.max(jnp.concatenate([m.chamfer for m in metrics]), axis=0).item(),
    )

    # Build the metrics object
    metrics = EvalMetrics(median=metrics_med, std=metrics_std, maximum=metrics_max)

    return metrics

  # Evaluate before training
  metrics_trn = evaluate(state=state, batches=dataset.batches(split=0, batch_size=(FLAGS.batch_size * NUM_DEVICES)))
  metrics_val = evaluate(state=state, batches=dataset.batches(split=1, batch_size=(FLAGS.batch_size * NUM_DEVICES)))
  # Report the initial evaluations
  time_tot_pre = time() - time_int_pre
  lr = state.opt_state[-1].hyperparams['learning_rate'][0].item()
  log('\t'.join([
    f'EPCH: {epochs_before : 04d}/{FLAGS.epochs : 04d}',
    f'LR: {lr : .2e}',
    f'TIME: {time_tot_pre : 06.1f}s',
    f'GRAD: {0. : .2e}',
    f'LOSS: {0. : .2e}',
    f'ERR-VAL (ERR-TRN): {metrics_val.median.l2 : .2%} ({metrics_trn.median.l2 : .2%})',
    f'RCL-VAL (RCL-TRN): {metrics_val.median.recall : .2%} ({metrics_trn.median.recall : .2%})',
    f'CHD-VAL (CHD-TRN): {metrics_val.median.chamfer : .2f} ({metrics_trn.median.chamfer : .2f})',
  ]), flush=True)

  # Set the checkpoint manager up
  DIR = DIR_EXPERIMENTS / f'E{FLAGS.exp}' / FLAGS.datapath / FLAGS.datetime
  (DIR / 'metrics').mkdir(exist_ok=True)
  (DIR / 'metrics/plots').mkdir(exist_ok=True)
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  checkpointer_options = orbax.checkpoint.CheckpointManagerOptions(
    max_to_keep=1,
    keep_period=None,
    best_fn=(lambda metrics: metrics['valid']['median']['l2']),
    best_mode='min',
    create=True,)
  checkpointer_save_args = orbax_utils.save_args_from_target(target={'state': state})
  checkpoint_manager = orbax.checkpoint.CheckpointManager(
    (DIR / 'checkpoints'), checkpointer, checkpointer_options)

  # Loop over epochs
  checkpointed_metrics = []
  _epochs = epochs // FLAGS.superbatch_repeats
  for _epoch in range(1, _epochs+1):
    # Get rngkey of the epoch
    subrngkeys = jax.random.split(jax.random.fold_in(rngkey, data=_epoch), num=3)

    # Store the initial time
    epoch = _epoch * FLAGS.superbatch_repeats
    time_int = time()

    # Re-construct the graphs with a new PRNG key
    # NOTE: In order to prevent training with the same regional nodes
    if dataset.metadata.fix and ((epoch % FLAGS.long_epoch_gap) == 0):
      dataset.build_graphs(builder=graph_builder, batch_size=FLAGS.batch_size, rngkey=subrngkeys[0])

    # Train one epoch
    state, loss, grad = train_one_epoch(
      rngkey=subrngkeys[1],
      state=state,
      batches=dataset.batches(split=0, batch_size=FLAGS.superbatch_size*NUM_DEVICES*FLAGS.batch_size, rngkey=subrngkeys[2]),
    )

    # Evaluate during training
    if ((epoch % FLAGS.long_epoch_gap) == 0) or (_epoch == _epochs):
      # Evaluate on training and validation datasets
      metrics_trn = evaluate(state=state, batches=dataset.batches(split=0, batch_size=(FLAGS.batch_size * NUM_DEVICES)))
      metrics_val = evaluate(state=state, batches=dataset.batches(split=1, batch_size=(FLAGS.batch_size * NUM_DEVICES)))
      # Log the results
      time_tot = (time() - time_int) / FLAGS.superbatch_repeats
      lr = state.opt_state[-1].hyperparams['learning_rate'][0].item()
      log('\t'.join([
        f'EPCH: {epochs_before + epoch : 04d}/{epochs : 04d}',
        f'LR: {lr : .2e}',
        f'TIME: {time_tot : 06.1f}s',
        f'GRAD: {grad.item() : .2e}',
        f'LOSS: {loss.item() : .2e}',
        f'ERR-VAL (ERR-TRN): {metrics_val.median.l2 : .2%} ({metrics_trn.median.l2 : .2%})',
        f'RCL-VAL (RCL-TRN): {metrics_val.median.recall : .2%} ({metrics_trn.median.recall : .2%})',
        f'CHD-VAL (CHD-TRN): {metrics_val.median.chamfer : .2f} ({metrics_trn.median.chamfer : .2f})',
      ]), flush=True)
      # Checkpoint
      step = epochs_before + epoch
      checkpoint_metrics = {
        'step': step,
        'grad': grad.item(),
        'loss': loss.item(),
        'lr': lr,
        'train': metrics_trn.to_dict(),
        'valid': metrics_val.to_dict(),
      }
      checkpointed_metrics.append(checkpoint_metrics)
      # Store the state and the metrics
      checkpoint_manager.save(
        step=step,
        items={'state': jax.device_get(unreplicate(state)),},
        metrics=checkpoint_metrics,
        save_kwargs={'save_args': checkpointer_save_args}
      )
      with open(DIR / 'metrics' / f'{str(step)}.json', 'w') as f:
        json.dump(checkpoint_metrics, f)
      # Plot the history of the metrics
      metrics_to_plot = {
        'optimization': (
          {'label': 'Training gradients', 'values': lambda m: m['grad']},
          {'label': 'Training loss', 'values': lambda m: m['loss']},
        ),
        'error': (
          {'label': 'Training error [%]', 'values': lambda m: m['train']['median']['l2'] * 100},
          {'label': 'Validation error [%]', 'values': lambda m: m['valid']['median']['l2'] * 100}
        ),
      }
      steps = [m['step'] for m in checkpointed_metrics]
      for filename, mtp in metrics_to_plot.items():
        fig, axs = plt.subplots(
          ncols=2,
          figsize=(10, 3),
          sharex=True,
          sharey=(filename != 'optimization'),
          tight_layout=True,
        )
        for i, item in enumerate(mtp):
          values = [item['values'](m) for m in checkpointed_metrics]
          ax: plt.Axes = axs[i]
          ax.scatter(steps, values, s=10, color='black', zorder=3)
          ax.set(ylabel=item['label'], yscale='log')
          ax.grid(which='both')
        file = DIR / 'metrics/plots' / f'{filename}.pdf'
        fig.savefig(file, dpi=100, bbox_inches='tight')
        plt.close(fig)

    # Or just report the loss
    else:
      time_tot = (time() - time_int) / FLAGS.superbatch_repeats
      log('\t'.join([
        f'EPCH: {epochs_before + epoch : 04d}/{epochs : 04d}',
        f'LR: {state.opt_state[-1].hyperparams["learning_rate"][0].item() : .2e}',
        f'TIME: {time_tot : 06.1f}s',
        f'GRAD: {grad.item() : .2e}',
        f'LOSS: {loss.item() : .2e}',
      ]), flush=True)

  return unreplicate(state)

def get_operator(model_configs: Mapping[str, Any], dataset: Dataset) -> AbstractOperator:
  """Build the model based on the given configurations."""

  # Set model kwargs
  if model_configs is None:
    if FLAGS.core_name == 'XRIGNO':
      configs_core = dict(
        num_outputs=sum([dataset.sample.functions[key].values.shape[-1] for key in dataset.metadata.out]),
        processor_steps=FLAGS.processor_steps,
        node_latent_size=FLAGS.node_latent_size,
        edge_latent_size=FLAGS.edge_latent_size,
        mlp_hidden_layers=1,
        p_edge_masking=FLAGS.p_edge_masking,
        tdep=dataset.time_dependent,
      )
    elif FLAGS.core_name == 'XGAOT':
      configs_core = dict(
        num_outputs=sum([dataset.sample.functions[key].values.shape[-1] for key in dataset.metadata.out]),
        gridres=FLAGS.rmesh_gridres,
        patch_size=2,
        transformer_hidden_size=FLAGS.node_latent_size,
        processor_steps=FLAGS.processor_steps,
        processor_attn_heads=4,
        latent_size=FLAGS.node_latent_size,
        mlp_hidden_layers=1,
        p_edge_masking=FLAGS.p_edge_masking,
        tdep=dataset.time_dependent,
      )

    model_configs = {
      'configs_core': configs_core,
      'configs_extender': dict(
        depth=FLAGS.ext_depth,
        out_dim=FLAGS.ext_vars,
        latent_dim=FLAGS.ext_latent_size,
        n_heads=FLAGS.ext_heads,
        ff_mult=1,
        p_masking=FLAGS.ext_p_masking,
        attn_dropout=0.0,
        ff_dropout=0.0,
      ),
      'use_extender': FLAGS.extender,
      'boundary_size': dataset.metadata.boundary_size,
    }

  model = models.__dict__[FLAGS.core_name](**model_configs)

  return model

def main():
  # Check the available devices
  process_index = jax.process_index()
  process_count = jax.process_count()
  local_devices = jax.local_devices()
  log(f'JAX host: {process_index} / {process_count}', flush=True)
  log(f'JAX local devices: {local_devices}', flush=True)
  # We only support single-host training
  assert process_count == 1
  # Check the inputs
  if not FLAGS.datetime:
    FLAGS.datetime = datetime.now().strftime('%Y%m%d-%H%M%S')
  assert (FLAGS.epochs % FLAGS.superbatch_repeats) == 0
  if FLAGS.core_name == 'XRIGNO':
    assert FLAGS.rmesh_gridres == 0
    FLAGS.rmesh_gridres = None
  elif FLAGS.core_name == 'XGAOT':
    assert FLAGS.rmesh_gridres > 0
    assert FLAGS.rmesh_gridres % 2 == 0
    FLAGS.rmesh_gridres = (FLAGS.rmesh_gridres, FLAGS.rmesh_gridres)
  else:
    raise ValueError(FLAGS.core_name)

  # Initialize the random keys
  rngkey = jax.random.key(FLAGS.seed)
  subrngkeys = jax.random.split(rngkey, num=3)

  # Read the dataset
  dataset = Dataset(
    dir=FLAGS.datadir,
    name=FLAGS.datapath,
    file='train.nc',
    space_downsample_factor=FLAGS.space_downsample_factor,
    boundary_downsample_factor=FLAGS.boundary_downsample_factor,
    splits=[(0, FLAGS.n_train), (FLAGS.n_train, FLAGS.n_train + FLAGS.n_valid)],
    replace_nan='maxabs',
    rngkey=subrngkeys[0],
  )
  # Add geometric features to the input domain functions
  dataset.metadata.dom += dataset.metadata.geo
  # Add pre-computed extensions to the input domain functions
  if FLAGS.use_extensions:
    dataset.metadata.dom += dataset.metadata.ext
  # Compute dataset statistics
  dataset.compute_stats(split=0, exclude_percentile=EXCLUDE_PERCENTILE_STATS, batch_size=FLAGS.batch_size)

  # Read the checkpoint
  if FLAGS.params:
    DIR_OLD_EXPERIMENT = DIR_EXPERIMENTS / FLAGS.params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    step = orbax.checkpoint.CheckpointManager(DIR_OLD_EXPERIMENT / 'checkpoints', orbax_checkpointer).latest_step()
    ckpt = orbax_checkpointer.restore(directory=(DIR_OLD_EXPERIMENT / 'checkpoints' / str(step) / 'default'))
    state = ckpt['state']
    params = state['params']
    with open(DIR_OLD_EXPERIMENT / 'configs.json', 'rb') as f:
      old_configs = json.load(f)
      model_configs = old_configs['model_configs']
  else:
    params = None
    model_configs = None
  # Get the model and set a stepper
  model = get_operator(model_configs, dataset)
  stepper = OutputStepper(operator=model)

  # Store the configurations
  DIR = DIR_EXPERIMENTS / f'E{FLAGS.exp}' / FLAGS.datapath / FLAGS.datetime
  DIR.mkdir(parents=True)
  log(f'Experiment stored in {DIR.relative_to(DIR_EXPERIMENTS).as_posix()}', flush=True)
  flag_values = vars(FLAGS)
  with open(DIR / 'configs.json', 'w') as f:
    json.dump(fp=f,
      obj={'flags': flag_values, 'model_configs': model.configs},
      indent=2,
    )
  # Store the statistics
  with open(DIR / 'stats.pkl', 'wb') as f:
    pickle.dump(file=f, obj=dataset.stats)

  # Construct the graphs
  log('Constructing the graphs for all dataset samples...', flush=True)
  graph_builder = GraphBuilder(
    pmesh_subsample_factor=FLAGS.pmesh_subsample_factor,
    overlap_factor_p2r=FLAGS.overlap_factor_p2r,
    overlap_factor_r2p=FLAGS.overlap_factor_r2p,
    rmesh_levels=FLAGS.rmesh_levels,
    rmesh_subsample_factor=FLAGS.rmesh_subsample_factor,
    periodic=dataset.metadata.periodic,
    node_coordinate_freqs=FLAGS.node_coordinate_freqs,
    gridres=FLAGS.rmesh_gridres,
  )
  dataset.build_graphs(builder=graph_builder, batch_size=FLAGS.batch_size)
  num_edges_p2r = dataset.rigs.p2r_edge_indices.shape[1]
  num_edges_r2r = dataset.rigs.r2r_edge_indices.shape[1]
  num_edges_r2p = (dataset.rigs.r2p_edge_indices.shape[1]
    if (dataset.rigs.r2p_edge_indices is not None) else dataset.rigs.p2r_edge_indices.shape[1])
  log(f'Constructed {len(dataset.rigs)} graph(s).', flush=True)
  log(f'Each graph has [ {num_edges_p2r / 1000 : .1f}k + '
    f'{num_edges_r2r / 1000 : .1f}k + {num_edges_r2p / 1000 : .1f}k ] edges', flush=True)

  # Initialzize the model or use the loaded parameters
  if params is None:
    dummy_graph_builder = GraphBuilder(
      pmesh_subsample_factor=16,
      overlap_factor_p2r=.01,
      overlap_factor_r2p=.01,
      rmesh_levels=1,
      rmesh_subsample_factor=4,
      periodic=dataset.metadata.periodic,
      node_coordinate_freqs=FLAGS.node_coordinate_freqs,
      gridres=FLAGS.rmesh_gridres,
    )
    dummy_graphs = dummy_graph_builder.build_graphs(
        dummy_graph_builder.build_metadata(
        x_inp=dataset.sample.x[0, 0],
        x_out=dataset.sample.x[0, 0],
        x_bnd=dataset.sample.x[0, 0, :100],
        z_bnd=dataset.sample.functions['sdfgrad'].values[0, 0, :100],
        bbox=np.array(dataset.metadata.bbox_x),
      )
    )
    dummy_graphs = jax.tree.map(lambda v: jnp.repeat(v, repeats=FLAGS.batch_size, axis=0), dummy_graphs)
    dummy_inputs = Inputs(
      s=jnp.ones(shape=(FLAGS.batch_size, 1, dataset.sample.x.shape[2], sum([dataset.sample.functions[key].values.shape[-1] for key in dataset.metadata.geo]))),
      a=jnp.ones(shape=(FLAGS.batch_size, 1, dataset.sample.x.shape[2], sum([dataset.sample.functions[key].values.shape[-1] for key in dataset.metadata.dom]))),
      q={key: jnp.ones(shape=(FLAGS.batch_size, 1, dataset.sample.x.shape[2], sum([dataset.sample.functions[key].values.shape[-1]]))) for key in dataset.metadata.seg},
      m={key: jnp.ones(shape=(FLAGS.batch_size, 1, dataset.sample.x.shape[2]), dtype=bool) for key in dataset.metadata.seg},
      x_inp=jnp.tile(dataset.sample.x, reps=(FLAGS.batch_size, 1, 1, 1)),
      x_out=jnp.tile(dataset.sample.x, reps=(FLAGS.batch_size, 1, 1, 1)),
      t=None,
      tau=None,
    )
    dummy_inputs = stepper.normalize_inputs(dataset.stats, dummy_inputs)
    variables = model.init(rngs=subrngkeys[1], inputs=dummy_inputs, graphs=dummy_graphs, deterministic=True)
    params = variables['params']

  # Report the total number of parameters
  n_model_parameters = sum([arr.size for arr in jax.tree.leaves(params)])
  log(f'Training a {model.__class__.__name__} with {n_model_parameters} parameters', flush=True)

  # Set optimizer transition steps
  num_big_batches = (dataset.splits[0][1] - dataset.splits[0][0]) // (FLAGS.batch_size * NUM_DEVICES * FLAGS.superbatch_size)
  transition_steps = FLAGS.epochs * num_big_batches * FLAGS.superbatch_size
  # Set learning rate and optimizer
  pct_start = .05  # Warmup cosine onecycle
  lr = optax.join_schedules(
    schedules=[
      optax.cosine_onecycle_schedule(
        transition_steps=transition_steps,
        peak_value=FLAGS.lr_peak,
        pct_start=pct_start,
        div_factor=(FLAGS.lr_peak / FLAGS.lr_init),
        final_div_factor=(FLAGS.lr_init / FLAGS.lr_base),
      ),
    ],
    boundaries=[transition_steps],
  )
  tx = optax.chain(
    optax.adaptive_grad_clip(clipping=FLAGS.gclip),
    optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=FLAGS.weight_decay)
  )
  state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

  # Train the model
  loss_fn = lambda gtr, prd: rel_lp_loss(gtr, prd, p=2, q=EXCLUDE_PERCENTILE_LOSS)
  state = train(
    rngkey=subrngkeys[2],
    stepper=stepper,
    state=state,
    dataset=dataset,
    graph_builder=graph_builder,
    epochs=FLAGS.epochs,
    epochs_before=0,
    loss_fn=loss_fn,
  )

if __name__ == '__main__':
  FLAGS = define_flags()
  main()
