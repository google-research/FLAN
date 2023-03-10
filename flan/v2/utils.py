# Copyright 2022 The FLAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions."""

import collections
import copy
import re
from typing import Any, Callable, Dict, List, Mapping

from flan.v2 import constants_t0
from flan.v2 import preprocessors as prep
from flan.v2 import task_configs_v1
import numpy as np
import seqio
import tensorflow as tf

TaskConfig = task_configs_v1.TaskConfig


def sst_value_dataset(filepattern: str) -> tf.data.Dataset:
  """A SSTable reader that loads values only."""
  return tf.data.SSTableDataset(filepattern).map(lambda k, v: v)


def inplace_modify_preprocessors(preprocessors: List[Callable[..., Any]],
                                 replacements=Mapping[Callable[..., Any],
                                                      Callable[..., Any]]):
  for idx, p in enumerate(preprocessors):
    if p in replacements:
      preprocessors[idx] = replacements[p]


class DecoderOnlyTaskIdFeatureConverter(seqio.PrefixLMFeatureConverter):
  """Feature converter that passes through the "task_ids features."""
  TASK_FEATURES = {
      "inputs": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "task_ids": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      "decoder_target_tokens":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_causal_attention":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "task_ids":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "decoder_segment_ids": tf.int32,
      "decoder_positions": tf.int32,
      "task_ids": tf.int32,
  }

  def _lm_convert_example(
      self, features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    """Convert an LM example into an example with model features."""
    # targets_segment_id is present only for a packed dataset.
    decoder_input_tokens = seqio.utils.make_autoregressive_inputs(
        features["targets"],
        sequence_id=features.get("targets_segment_ids", None))

    d = {
        "decoder_target_tokens": features["targets"],
        "decoder_input_tokens": decoder_input_tokens,
        "decoder_loss_weights": seqio.non_padding_position(features["targets"]),
        "task_ids": features["task_ids"]
    }

    if self.pack:
      d["decoder_segment_ids"] = features["targets_segment_ids"]
      d["decoder_positions"] = features["targets_positions"]

    return d

  def _convert_example(
      self, features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    """Convert a Prefix LM example into an example with model features."""
    # First use the standard LM conversion.
    # lm_features = super()._convert_example(features)
    lm_features = self._lm_convert_example(features)

    # Initialize the return dictionary with the lm features.
    d = dict(lm_features)

    if self.pack:
      positions = features["targets_positions"]
    # Without packing, targets_positions field does not exist.
    else:
      positions = tf.range(tf.size(features["targets"]))

    inputs_width = features["inputs_width_add_pos"]
    # Binary mask where 1 represents a position in a non-causal attention region
    d["decoder_causal_attention"] = tf.cast(
        positions < inputs_width, dtype=features["targets"].dtype)

    # When computing the loss weights with self.loss_on_targets_only = True, we
    # use features["inputs_width"], which encodes the number of "inputs" tokens.
    if self.loss_on_targets_only:
      # 1's on inputs and 0's on targets and padding.
      inputs = positions < features["inputs_width"]

      # 1's on inputs and targets and 0's on padding.
      padding_mask = tf.cast(d["decoder_loss_weights"], dtype=tf.bool)

      # XOR picks targets only. See docstring for an example.
      d["decoder_loss_weights"] = tf.cast(
          tf.math.logical_xor(inputs, padding_mask),
          dtype=features["targets"].dtype)

    return d

  def _convert_features(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Convert the input dataset to an output dataset to be fed to the model."""

    def concat_and_add_masks(features):
      inputs = features["inputs"]
      targets = features["targets"]
      # If the targets are empty, we add one padding target.
      targets = tf.cond(
          tf.size(targets) > 0, lambda: targets,
          lambda: tf.zeros(1, dtype="int32"))

      # Width of the "inputs" portion in the concatenated sequence.
      width = tf.size(inputs)
      inputs_width = tf.fill([tf.size(inputs) + tf.size(targets)], width)

      # Width with an extra position to the right in the inputs mask. See
      # docstring for details.
      inputs_width_add_pos = tf.fill([tf.size(inputs) + tf.size(targets)],
                                     width + 1)

      return {
          "targets": tf.concat([inputs, targets], axis=-1),
          "inputs_width": inputs_width,
          "inputs_width_add_pos": inputs_width_add_pos,
          "task_ids": features["task_ids"],
      }

    ds = ds.map(
        concat_and_add_masks, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # concat_length = sum(task_feature_lengths.values())
    concat_length = task_feature_lengths["inputs"] + task_feature_lengths[
        "targets"]
    concat_task_feature_lengths = {
        "targets": concat_length,
        "inputs_width": concat_length,
        "inputs_width_add_pos": concat_length,
        "task_ids": concat_length
    }

    ds = self._pack_or_pad(ds, concat_task_feature_lengths)
    return ds.map(
        self._convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    decoder_length = task_feature_lengths["inputs"] + task_feature_lengths[
        "targets"]
    concat_length = {"targets": decoder_length}
    lm_model_feature_lengths = super().get_model_feature_lengths(concat_length)
    model_feature_lengths = dict(lm_model_feature_lengths)
    model_feature_lengths["decoder_causal_attention"] = decoder_length
    model_feature_lengths["task_ids"] = decoder_length
    return model_feature_lengths


@seqio.map_over_dataset
def add_task_id(example, task_id):
  """Add task IDs at the example level."""
  example["task_ids"] = [task_id]
  return example


# TODO(hwchung): maybe enable shuffle and seed, num_epochs
def _get_mixture_examples(mixture, num_steps, batch_size, task_feature_lengths):
  """Given a mixture, return the examples after adding task ids to the tasks."""
  datasets = []
  rates = []
  task_id_to_task = {}
  for i, task in enumerate(mixture.tasks):
    task_ds = task.get_dataset(
        task_feature_lengths,
        split="train",
        use_cached=False,
        shuffle=False,
        shard_info=seqio.ShardInfo(index=0, num_shards=1),
        num_epochs=None)
    task_id = i + 1
    task_id_to_task[task_id] = task
    datasets.append(add_task_id(task_ds, task_id))
    rates.append(mixture.get_rate(task))

  # `sample_seed = 42` is used in `mixture.get_datset` if `seed` arg is None.
  mixture_ds = mixture._sample_fn(datasets, rates, seed=42)  # pylint:disable=protected-access
  fc = DecoderOnlyTaskIdFeatureConverter(
      pack=True, use_custom_packing_ops=False)
  task_feature_lengths = dict(task_feature_lengths)
  task_feature_lengths["task_ids"] = max(task_feature_lengths.values())
  model_ds = fc(mixture_ds, task_feature_lengths)
  model_ds = model_ds.batch(batch_size, drop_remainder=True)
  model_ds = model_ds.take(num_steps)
  return list(model_ds.as_numpy_iterator()), task_id_to_task


def compute_stats(mixture, num_steps, batch_size, task_feature_lengths):
  """Compute the mixture statistics."""
  examples, task_id_to_task = _get_mixture_examples(mixture, num_steps,
                                                    batch_size,
                                                    task_feature_lengths)
  task_ids_counts = collections.Counter()
  non_padding_fractions = []
  for example in examples:
    task_ids = example["task_ids"]
    if task_ids.ndim == 2:
      task_ids = task_ids.reshape(-1)
    task_ids_counts.update(task_ids)
    non_padding_fractions.append(
        np.mean(example["decoder_target_tokens"] != 0, axis=1))

  non_padding_fractions_array = np.stack(non_padding_fractions, axis=0)
  assert non_padding_fractions_array.shape == (num_steps, batch_size)
  non_padding_fraction = non_padding_fractions_array.mean()
  task_ids_counts.pop(0)  # don't need padding count

  # Convert task_id to task_names
  task_counts = collections.Counter()
  for task_id, count in task_ids_counts.items():
    task_counts[task_id_to_task[task_id]] = count
  return task_counts, non_padding_fraction


def reset_split_maxes_on_flan_v0_configs(
    original_flan_configs: Dict[str, TaskConfig],):
  """This function creates new FLAN v0 Task Configs without the 30k train limit.

  Args:
    original_flan_configs: A dict of task keys to their TaskConfigs.

  Returns:
    new_flan_configs: A dict of task keys to their TaskConfigs without 30k limit
  """
  new_flan_configs = {}
  for key, tconfig in original_flan_configs.items():
    if key == "wsc273":
      # Remove wsc273 since it doesn't contain training data.
      continue

    if (not isinstance(tconfig.source,
                       seqio.dataset_providers.TfdsDataSource)) or (key in [
                           "true_case",
                           "fix_punct",
                           "word_segment",
                           "para_crawl_enes",
                           "opinion_abstracts_idebate",
                           "opinion_abstracts_rotten_tomatoes",
                       ]) or (not tconfig.source._tfds_dataset._split_map):  # pylint: disable=protected-access
      new_flan_configs[key] = tconfig
      continue

    reserve_exs = None
    old_split_map = tconfig.source._tfds_dataset._split_map  # pylint: disable=protected-access
    new_split_map = copy.deepcopy(old_split_map)
    train_split_idx = new_split_map["train"].find("[")
    val_split_idx = new_split_map["validation"].find("[")
    test_split_idx = new_split_map["test"].find("[")

    # If we are only taking the last part of the train set as validation, then
    # it is to add to validation/test, not cap train.
    val_neg_split = new_split_map["validation"].find("-")
    if val_neg_split != -1:
      end_reserve_val = new_split_map["validation"].find("]") - 1
      reserve_exs = new_split_map["validation"][val_neg_split +
                                                1:end_reserve_val]

    # Fix the existing splits
    if train_split_idx != -1:
      new_split_map["train"] = new_split_map["train"][:train_split_idx]
      if reserve_exs:
        new_split_map["train"] += f"[:-{reserve_exs}]"
    if val_split_idx != -1 and reserve_exs is None:
      new_split_map["validation"] = new_split_map["validation"][:val_split_idx]
    if test_split_idx != -1 and reserve_exs is None:
      new_split_map["test"] = new_split_map["test"][:test_split_idx]

    new_prep_fn = functools.partial(prep.add_source_info,
      task_name="Flan2021", task_source=tconfig.source._tfds_dataset.name)
    new_flan_configs[key] = TaskConfig(
        source=seqio.TfdsDataSource(
            tfds_name=tconfig.source._tfds_dataset.name,  # pylint: disable=protected-access
            splits=new_split_map,
        ),
        preprocessors=tconfig.preprocessors + [new_prep_fn],
        postprocess_fn=tconfig.postprocess_fn,
        metric_fns=tconfig.metric_fns,
    )
  return new_flan_configs


def t_name_to_flan_pattern_name(t_name: str) -> str:
  """Converts `t_name` to flan `PATTERN` key.

  Some seqio tasks use the same flan patterns.
  Args:
    t_name: Task config name.

  Returns:
    a key for `PATTERNS`.
  """

  if "para_crawl" in t_name:
    mapped_name = "para_crawl"
  elif "wmt16_translate" in t_name:
    mapped_name = "wmt16_translate"
  elif t_name in {"arc_challenge", "arc_easy"}:
    mapped_name = "arc"
  elif t_name in {"anli_r1", "anli_r2", "anli_r3"}:
    mapped_name = "anli"
  elif t_name in {"mnli_matched", "mnli_mismatched"}:
    mapped_name = "mnli"
  elif t_name in {"trivia_qa_wiki_missing", "trivia_qa"}:
    mapped_name = "trivia_qa"
  elif t_name in constants_t0.T0_TRAIN_TASK_METADATA:
    mapped_name = str(constants_t0.T0_TRAIN_TASK_METADATA[t_name]["task_type"])
  elif t_name == "tfds_natural_instructions":
    mapped_name = "natinst_v2"
  elif t_name.replace("t0_", "") in constants_t0.T0_TRAIN_TASKS_ABBREV:
    mod_t_name = t_name.replace("t0_", "t0_task_adaptation:")
    mapped_name = str(
        constants_t0.T0_TRAIN_TASK_METADATA[mod_t_name]["task_type"])
  elif re.search(r"task(\d+)_.*", t_name):
    mapped_name = "natinst_v2"
  else:
    mapped_name = t_name

  return mapped_name
