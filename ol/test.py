"""Evaluation and inference on test datasets.

This module provides utilities for model evaluation, inference on held-out
test data, and metric computation. Can be run as a standalone script.
"""

import argparse
import json
import pickle
import shutil
from typing import Mapping, Callable

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from flax.training.common_utils import shard
from flax.jax_utils import replicate

from ol import models
from ol.dataset.dataset import Dataset, Batch, StatsCollectionType
from ol.experiments import DIR_EXPERIMENTS
from ol.graph.graphbuilder import GraphBuilder
from ol.metrics import rel_lp_error, chamfer, recall_tol
from ol.metrics import rel_lp_error_mean, chamfer_mean, recall_tol_mean
from ol.models.common import Inputs
from ol.stepping import Stepper, OutputStepper
from ol.utils import log, profile, segment_mean


NUM_DEVICES = jax.local_device_count()
IDX_FN = 14
FLAGS = None
RECALL_TOLERANCE = 2e-02
EXCLUDE_PERCENTILE = 0.2

def define_flags():
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp', type=str, required=True,
    help='Relative path of the experiment')
  parser.add_argument('--datadir', type=str, required=True,
    help='Path of the folder containing the datasets')
  parser.add_argument('--datapath', type=str, required=True,
    help='Relative path inside the data directory')
  parser.add_argument('--batch_size_per_device', type=int, default=16,
    help='Size of a batch of test samples per device')
  parser.add_argument('--seed', type=int, default=45,
    help='Seed for random number generator')
  return parser.parse_args()

def _print_between_dashes(msg):
  log('-' * 80, flush=True)
  log(msg, flush=True)
  log('-' * 80, flush=True)

def time_inference(
  dataset: Dataset,
  variables,
  stats: StatsCollectionType,
  stepper: Stepper,
  graph_builder: GraphBuilder,
  batch_size: int = 1,
  repeats: int = 10,
  jit: bool = True,
  output_buffer: Callable = (lambda msg: log(msg, flush=True)),
):
  # Set the functions to be timed
  apply_fn = stepper._apply_operator
  if jit: apply_fn = jax.jit(apply_fn, static_argnames=['deterministic'])
  graph_fn = lambda x, m, z: graph_builder.build_metadata(x, x, x[m], z[m], np.array(dataset.metadata.bbox_x))

  # Build dummy inputs and model parameters
  mask_bnd = np.any(np.stack([dataset.sample.functions[key].mask[0, 0] for key in dataset.metadata.seg]), axis=0)
  dummy_graphs = graph_builder.build_graphs(
    graph_builder.build_metadata(
    x_inp=dataset.sample.x[0, 0],
    x_out=dataset.sample.x[0, 0],
    x_bnd=dataset.sample.x[0, 0, mask_bnd],
    z_bnd=dataset.sample.functions['sdfgrad'].values[0, 0, mask_bnd],
    bbox=np.array(dataset.metadata.bbox_x),
    rmesh_correction_dsf=1.0,
  ))
  dummy_graphs = jax.tree.map(lambda v: jnp.repeat(v, repeats=batch_size, axis=0), dummy_graphs)
  dummy_inputs = Inputs(
    s=jnp.ones(shape=(batch_size, 1, dataset.sample.x.shape[2], sum([dataset.sample.functions[key].values.shape[-1] for key in dataset.metadata.geo]))),
    a=jnp.ones(shape=(batch_size, 1, dataset.sample.x.shape[2], sum([dataset.sample.functions[key].values.shape[-1] for key in dataset.metadata.dom]))),
    q={key: jnp.ones(shape=(batch_size, 1, dataset.sample.x.shape[2], sum([dataset.sample.functions[key].values.shape[-1]]))) for key in dataset.metadata.seg},
    m={key: jnp.ones(shape=(batch_size, 1, dataset.sample.x.shape[2]), dtype=bool) for key in dataset.metadata.seg},
    x_inp=jnp.tile(dataset.sample.x, reps=(batch_size, 1, 1, 1)),
    x_out=jnp.tile(dataset.sample.x, reps=(batch_size, 1, 1, 1)),
    t=None,
    tau=None,
  )
  dummy_inputs = stepper.normalize_inputs(stats, dummy_inputs)

  # Set model arguments
  kwargs = dict(variables=variables, inputs=dummy_inputs, graphs=dummy_graphs, deterministic=True)
  # Profile graph building
  t_graph = profile(graph_fn, kwargs=dict(x=dataset.sample.x[0, 0], m=mask_bnd, z=dataset.sample.functions['sdfgrad'].values[0, 0]), repeats=3)
  # Profile compilation
  t_compilation = profile(f=apply_fn, kwargs=kwargs, repeats=1, block_until_ready=True)
  # Profile inferrence after compilation
  t = profile(f=apply_fn, kwargs=kwargs, repeats=repeats, block_until_ready=True)

  # Report the results
  msgs = [
    '-' * 80,
    'NUMBER OF DEVICES: 1',
    f'BATCH SIZE: {batch_size}',
    f'Graph building: {t_graph * 1000: .2f}ms',
    f'Compilation: {t_compilation : .2f}s',
    f'Inference: {t * 1000 : .2f}ms per batch',
    f'Inference: {t * 1000 / batch_size : .2f}ms per sample',
    '-' * 80,
  ]
  for line in msgs:
    output_buffer(line)

def infer_model(
  stepper: Stepper,
  state: Mapping,
  stats: StatsCollectionType,
  dataset: Dataset,
  graph_builder: GraphBuilder,
  batch_size: int = 1,
  get_intermediates: bool = False,
  boundary_noise_to_signal_ratio: float = 0.0,
):

  @jax.pmap
  def _push_one_batch(
    state: Mapping,
    stats: StatsCollectionType,
    batch: Batch,
  ):
    # Set inputs
    inputs = Inputs(
      s=jnp.concatenate([batch.functions[key].values[:, [0]] for key in dataset.metadata.geo], axis=-1),
      a=jnp.concatenate([batch.functions[key].values[:, [0]] for key in dataset.metadata.dom], axis=-1),
      q={key: batch.functions[key].values[:, [0]] for key in dataset.metadata.seg},
      m={key: batch.functions[key].mask[:, [0]] for key in dataset.metadata.seg},
      x_inp=batch.x,
      x_out=batch.x,
      t=None,
      tau=None,
    )
    # Add noise
    if boundary_noise_to_signal_ratio > 0:
      for key in inputs.q.keys():
        batched_slice = jax.vmap(lambda f, m: f[jnp.where(m, size=dataset.metadata.boundary_size)[0]])
        _q = batched_slice(inputs.q[key].squeeze(1), inputs.m[key].squeeze(1))
        noise_var = boundary_noise_to_signal_ratio * (jnp.std(_q, axis=1, keepdims=True)**2 + jnp.mean(_q, axis=1, keepdims=True)**2)[:, None]
        noise_std = jnp.sqrt(noise_var)
        noise = noise_std * jax.random.normal(jax.random.PRNGKey(0), shape=inputs.q[key].shape)
        inputs.q[key] += noise
    # Set the keyword arguments
    kwargs = dict(
      variables={'params': state['params']},
      stats=stats,
      inputs=inputs,
      graphs=graph_builder.build_graphs(batch.g),
      deterministic=True,
    )
    # Infer the model
    output = stepper.apply(**kwargs)
    intermediates = stepper.get_intermediates(**kwargs) if get_intermediates else None
    return output, intermediates

  # Replicate state, stats, and shard keys
  state = replicate(state)
  stats = replicate(stats)

  # Loop over the batches and infer the model
  model_outputs = []
  intermediates = []
  for batch in dataset.batches(split=0, batch_size=batch_size):
    # [NUM_DEVICES*batch_size_per_device, ...] -> [NUM_DEVICES, batch_size_per_device, ...]
    batch = Batch(
      x=shard(batch.x),
      t=shard(batch.t),
      g=shard(batch.g),
      functions=shard(batch.functions),
    )
    _outputs, _intermediates = _push_one_batch(state, stats, batch)
    model_outputs.append(_outputs.reshape(batch_size, *_outputs.shape[2:]))
    intermediates.append(jax.tree.map(lambda arr: arr.reshape(batch_size, -1, *arr.shape[2:]), _intermediates))
  model_outputs = jnp.concatenate(model_outputs)
  intermediates = jax.tree.map(lambda *arrs: jnp.concatenate(arrs), *intermediates) if get_intermediates else None

  return model_outputs, intermediates

def main():
  # Check the available devices
  process_index = jax.process_index()
  process_count = jax.process_count()
  local_devices = jax.local_devices()
  log(f'JAX host: {process_index} / {process_count}', flush=True)
  log(f'JAX local devices: {local_devices}', flush=True)
  # We only support single-host testing
  assert process_count == 1

  # # Initialize the random key
  rngkey = jax.random.key(FLAGS.seed)
  subrngkeys = jax.random.split(rngkey, num=1)

  # Set the directory
  DIR = DIR_EXPERIMENTS / FLAGS.exp
  # Read the stats
  with open(DIR / 'stats.pkl', 'rb') as f:
    stats: StatsCollectionType = pickle.load(f)
  # Read the configs
  with open(DIR / 'configs.json', 'rb') as f:
    configs = json.load(f)
  flags = configs['flags']
  model_configs = configs['model_configs']
  # Read the state
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  mngr = orbax.checkpoint.CheckpointManager(DIR / 'checkpoints')
  best_checkpointed_step = mngr.best_step()
  ckpt = orbax_checkpointer.restore(directory=(DIR / 'checkpoints' / str(best_checkpointed_step) / 'default'))
  state = jax.tree.map(jnp.array, ckpt['state'])
  # Read the metrics
  with open(DIR / f'metrics/{best_checkpointed_step}.json', 'r') as f:
    metrics = json.load(f)
  # Reports
  n_model_parameters_total = sum([arr.size for arr in jax.tree.leaves(state['params'])])
  n_model_parameters_operator = sum([arr.size for arr in jax.tree.leaves(state['params']['operator'])])
  log(f'Best epoch: {best_checkpointed_step}', flush=True)
  log(f'Training error: {metrics["train"]["median"]["l2"] : .2%} ± {metrics["train"]["std"]["l2"] : .2%}', flush=True)
  log(f'Validation error: {metrics["valid"]["median"]["l2"] : .2%} ± {metrics["valid"]["std"]["l2"] : .2%}', flush=True)
  log(f'Number of parameters: {n_model_parameters_total} = {n_model_parameters_total-n_model_parameters_operator} (extender) + {n_model_parameters_operator} (core)', flush=True)

  # Read the dataset
  dataset = Dataset(
    dir=FLAGS.datadir,
    name=FLAGS.datapath,
    file='test.nc',
    space_downsample_factor=flags['space_downsample_factor'],
    boundary_downsample_factor=flags['boundary_downsample_factor'],
    splits=None,
    replace_nan='maxabs',
    rngkey=subrngkeys[0],
  )
  # Add geometric features to the input domain functions
  dataset.metadata.dom += dataset.metadata.geo
  # Add pre-computed extensions to the input domain functions
  if flags['use_extensions']:
    dataset.metadata.dom += dataset.metadata.ext
  assert NUM_DEVICES*FLAGS.batch_size_per_device <= dataset.splits[0][1]
  # Set the graph builder
  builder = GraphBuilder(
    pmesh_subsample_factor=flags['pmesh_subsample_factor'],
    overlap_factor_p2r=flags['overlap_factor_p2r'],
    overlap_factor_r2p=flags['overlap_factor_r2p'],
    rmesh_levels=flags['rmesh_levels'],
    rmesh_subsample_factor=flags['rmesh_subsample_factor'],
    periodic=dataset.metadata.periodic,
    node_coordinate_freqs=flags['node_coordinate_freqs'],
    gridres=flags['rmesh_gridres'],
  )
  # Build the graphs
  rmesh_correction_dsf = flags['space_downsample_factor'] / dataset.space_downsample_factor
  dataset.build_graphs(builder, rmesh_correction_dsf=rmesh_correction_dsf)
  # Set the operator
  model = models.__dict__[flags['core_name']](**model_configs)
  # Set the stepper
  stepper = OutputStepper(operator=model)

  # Profile inference time
  # NOTE: One compilation per each profiling
  time_inference(dataset=dataset, variables={'params': state['params']}, stats=stats, stepper=stepper, graph_builder=builder, batch_size=1)

  # Create a clean directory for tests
  DIR_TESTS = DIR / 'tests'
  if DIR_TESTS.exists():
    shutil.rmtree(DIR_TESTS)
  DIR_TESTS.mkdir()

  # Get model estimations with all settings
  u_prd, _ = infer_model(
    stepper=stepper,
    state=state,
    stats=stats,
    dataset=dataset,
    graph_builder=builder,
    batch_size=(NUM_DEVICES*FLAGS.batch_size_per_device),
    get_intermediates=False,
  )

  # Compute and store the errors
  batch = next(dataset.batches(split=0, batch_size=dataset.splits[0][1]))
  u_gtr = jnp.concatenate([batch.functions[key].values[:, [0]] for key in dataset.metadata.out], axis=-1)
  x = batch.x
  bbox_x = jnp.array(dataset.metadata.bbox_x)
  char_length = jnp.linalg.norm(bbox_x[1, :]-bbox_x[0, :])
  tol = RECALL_TOLERANCE * char_length
  errors = {
    'l1': rel_lp_error_mean(u_gtr, u_prd, p=1, exclude_percentile=EXCLUDE_PERCENTILE).tolist(),
    'l2': rel_lp_error_mean(u_gtr, u_prd, p=2, exclude_percentile=EXCLUDE_PERCENTILE).tolist(),
    'recall': recall_tol_mean(x, u_gtr, u_prd, q=EXCLUDE_PERCENTILE, tol=tol).tolist(),
    'chamfer': chamfer_mean(x, u_gtr, u_prd, q=EXCLUDE_PERCENTILE).tolist(),
    'l1-var': rel_lp_error(u_gtr, u_prd, p=1, exclude_percentile=EXCLUDE_PERCENTILE).tolist(),
    'l2-var': rel_lp_error(u_gtr, u_prd, p=2, exclude_percentile=EXCLUDE_PERCENTILE).tolist(),
    'recall-var': recall_tol(x, u_gtr, u_prd, q=EXCLUDE_PERCENTILE, tol=tol).tolist(),
    'chamfer-var': chamfer(x, u_gtr, u_prd, q=EXCLUDE_PERCENTILE).tolist(),
  }
  with open(DIR_TESTS / 'errors.json', 'w') as f:
    json.dump(obj=errors, fp=f)
  chunks = [idx_chunk for idx_chunk, key in enumerate(dataset.metadata.out) for _ in range(sum([len(arr.indices) for arr in dataset.metadata.functions[key].arrays]))]
  _print_between_dashes(
    f'Median relative L2 test error: {np.median(errors["l2"]).item() : .2%} ± {np.std(errors["l2"]).item() : .2%}\n' +
    f'\t\t\t\t (per variable): {", ".join([f"{item : .2%}" for item in np.median(errors["l2-var"], axis=0)])}\n' +
    f'\t\t\t\t (per function): {", ".join([f"{item : .2%}" for item in np.median(segment_mean(errors["l2-var"], chunks, axis=1), axis=0)])}'
  )
  _print_between_dashes(
    f'Median recall (tol={RECALL_TOLERANCE:.1%}) test score: {np.median(errors["recall"]).item() : .2%}\n' +
    f'\t\t\t\t (per variable): {", ".join([f"{item : .2%}" for item in np.median(errors["recall-var"], axis=0)])}\n' +
    f'\t\t\t\t (per function): {", ".join([f"{item : .2%}" for item in np.median(segment_mean(errors["recall-var"], chunks, axis=1), axis=0)])}'
  )
  _print_between_dashes(
    f'Median test Chamfer distance: {np.median(errors["chamfer"]).item() : .2e} ({np.median(errors["chamfer"]).item()/char_length : .2%})\n' +
    f'\t\t\t\t (per variable): {", ".join([f"{item/char_length : .2%}" for item in np.median(errors["chamfer-var"], axis=0)])}\n' +
    f'\t\t\t\t (per function): {", ".join([f"{item/char_length : .2%}" for item in np.median(segment_mean(errors["chamfer-var"], chunks, axis=1), axis=0)])}'
  )

  _print_between_dashes('DONE')

if __name__ == '__main__':
  FLAGS = define_flags()
  main()
