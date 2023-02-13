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

"""Tests for utils."""

import functools
from absl.testing import absltest
from flan.v2 import utils
import jax
import numpy as np
import seqio
import tensorflow as tf


def dataset_fn(split, shuffle_files, value, length, dataset_size):
  del split, shuffle_files
  data = []
  for _ in range(dataset_size):
    data.append({
        "inputs": np.ones((length,), dtype=np.int32) * value,
        "targets": np.ones((length,), dtype=np.int32) * (value + 1),
    })

  output_types = {"inputs": np.int32, "targets": np.int32}
  output_shapes = {"inputs": [length], "targets": [length]}

  return tf.data.Dataset.from_generator(
      lambda: data, output_types=output_types, output_shapes=output_shapes)


def create_test_mixture(dataset_size=400,
                        length=2,
                        task1_rate=1.0,
                        task2_rate=1.0,
                        task1_name="task1",
                        task2_name="task2",
                        mixture_name="mix"):
  output_features = {
      "inputs": seqio.Feature(seqio.PassThroughVocabulary(1)),
      "targets": seqio.Feature(seqio.PassThroughVocabulary(1)),
  }
  task1_size = dataset_size * 10
  task2_size = dataset_size

  dataset_fn1 = functools.partial(
      dataset_fn, value=6, length=length * 10, dataset_size=task1_size)
  dataset_fn2 = functools.partial(
      dataset_fn, value=8, length=length, dataset_size=task2_size)

  task1 = seqio.TaskRegistry.add(
      task1_name,
      source=seqio.FunctionDataSource(dataset_fn1, splits=["train"]),
      output_features=output_features)
  task2 = seqio.TaskRegistry.add(
      task2_name,
      source=seqio.FunctionDataSource(dataset_fn2, splits=["train"]),
      output_features=output_features)

  return seqio.MixtureRegistry.add(mixture_name, [(task1.name, task1_rate),
                                                  (task2.name, task2_rate)])


class UtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    seqio.TaskRegistry.reset()
    seqio.MixtureRegistry.reset()

  def test_inplace_modify_preprocessors(self):

    def _a():
      pass

    def _b():
      pass

    def _c():
      pass

    list0 = [_a, _b, _a, _c]
    utils.inplace_modify_preprocessors(list0, {_a: _b, _c: _b})
    self.assertListEqual(list0, [_b, _b, _b, _b])

  def test_get_mixture_examples(self):
    dataset_size = 20
    length = 2
    batch_size = 16
    num_steps = 10
    task1_rate = 2.0
    task2_rate = 3.0
    task1_name = "task1"
    task2_name = "task2"
    task_feature_lengths = {"inputs": 800, "targets": 300}
    mixture = create_test_mixture(dataset_size, length, task1_rate, task2_rate,
                                  task1_name, task2_name)
    # The examples returned are not directly from the `mixture.get_dataset`;
    # they are created by first adding task_ids to the tasks and mimicing the
    # `mixture.get_dataset` logic in a minimal manner.
    examples, _ = utils._get_mixture_examples(mixture, num_steps, batch_size,
                                              task_feature_lengths)

    # Obtained the examples directly from the mixture.
    mixture_ds = mixture.get_dataset(
        task_feature_lengths, num_epochs=None, shuffle=False)
    fc = seqio.DecoderFeatureConverter(pack=True, use_custom_packing_ops=False)
    model_ds = fc(mixture_ds, task_feature_lengths)
    model_ds = model_ds.batch(batch_size, drop_remainder=True)
    mixture_examples = list(model_ds.take(num_steps).as_numpy_iterator())

    for example, mixture_example in zip(examples, mixture_examples):
      example_tmp = dict(example)
      example_tmp.pop("task_ids")
      jax.tree_map(np.testing.assert_array_equal, example_tmp, mixture_example)

  def test_task_proportions(self):
    dataset_size = 100
    length = 2
    batch_size = 64
    num_steps = 10
    task1_rate = 2.0
    task2_rate = 3.0
    task1_name = "task1"
    task2_name = "task2"
    task_feature_lengths = {"inputs": 800, "targets": 300}
    mixture = create_test_mixture(dataset_size, length, task1_rate, task2_rate,
                                  task1_name, task2_name)
    task_counts, _ = utils.compute_stats(mixture, num_steps, batch_size,
                                         task_feature_lengths)
    task1 = seqio.get_mixture_or_task(task1_name)
    task2 = seqio.get_mixture_or_task(task2_name)
    np.testing.assert_allclose(
        task_counts[task1] / task_counts[task2],
        task1_rate / task2_rate,
        rtol=1e-1)


if __name__ == "__main__":
  absltest.main()
