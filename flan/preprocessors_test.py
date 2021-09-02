# Copyright 2021 The FLAN Authors.
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

"""Tests for preprocessors."""

from absl.testing import parameterized
from absl.testing import absltest
import seqio
import tensorflow as tf

from flan import preprocessors
from flan import tasks
from flan import templates


class PreprocessorsTest(tf.test.TestCase, parameterized.TestCase):

  def test_get_training_keys(self):
    patterns = templates.PATTERNS['ag_news_subset']
    training_keys = preprocessors.get_training_keys(patterns)
    expected_training_keys = {'title', 'text', 'options_', 'options', 'answer'}
    self.assertEqual(training_keys, expected_training_keys)

  def test_remove_unbatchable_items_ds(self):
    task_name = 'natural_questions'
    nq_config = tasks.TASK_CONFIGS[task_name]
    ds = nq_config.source.get_dataset(split='train')
    ds = tasks._process_natural_questions(ds)

    # These are the original keys.
    expected_original_keys = {'question', 'answer', 'answers'}
    for example in ds:
      self.assertEqual(set(example.keys()), expected_original_keys)

    # Test whether the unbatchable items are successfully removed.
    training_keys = preprocessors.get_training_keys(
        templates.PATTERNS[task_name])
    ds = preprocessors.remove_unbatchable_items_ds(ds, training_keys)
    expected_training_keys = {'question', 'answer'}
    for example in ds:
      self.assertEqual(set(example.keys()), expected_training_keys)

  def test_remove_trailing_spaces(self):
    example = {
        'foo': tf.constant('some trailing spaces    \n', dtype=tf.string)
    }
    rst = preprocessors.remove_trailing_spaces.__wrapped__(
        example, features=['foo'])
    self.assertEqual(rst['foo'], 'some trailing spaces')

  def test_get_fewshot_num_tokens(self):
    example = {
        'train': {
            'inputs': tf.constant(['first input', 'second input second']),
            'targets': tf.constant(['first target', 'second target']),
        },
        'eval': {
            'inputs': 'eval input',
            'targets': 'eval target',
        }
    }

    class SimpleVocab(seqio.vocabularies.Vocabulary):

      def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        return tf.strings.split(s)

      def _base_vocab_size(self):
        return None

      def _decode(self, ids):
        return None

      def _decode_tf(self, ids):
        return None

      def _encode(self, s):
        return None

      def eos_id(self):
        return 0

      def unk_id(self):
        return 1

    output_features = {
        'inputs':
            seqio.Feature(
                vocabulary=SimpleVocab(), add_eos=True, required=False),
        'targets':
            seqio.Feature(vocabulary=SimpleVocab(), add_eos=True)
    }
    rst = preprocessors.get_fewshot_num_tokens.__wrapped__(
        example, output_features)
    self.assertAllEqual(rst['train']['inputs_num_tokens'], [2, 3])
    self.assertAllEqual(rst['train']['targets_num_tokens'], [2, 2])
    self.assertAllEqual(rst['eval']['inputs_num_tokens'], 2)
    self.assertAllEqual(rst['eval']['targets_num_tokens'], 2)

  @parameterized.named_parameters(
      ('allow_all_exemplars', 15, {
          'train': {
              'inputs': [111, 222, 333],
              'inputs_num_tokens': [1, 2, 3],
              'targets': [444, 555, 666],
              'targets_num_tokens': [1, 1, 1],
          },
          'eval': {
              'inputs_num_tokens': 5,
              'targets_num_tokens': 1,
              'num_exemplars': 3,
          }
      }),
      ('allow_two_exmemplars', 10, {
          'train': {
              'inputs': [111, 222],
              'inputs_num_tokens': [1, 2],
              'targets': [444, 555],
              'targets_num_tokens': [1, 1],
          },
          'eval': {
              'inputs_num_tokens': 5,
              'targets_num_tokens': 1,
              'num_exemplars': 2,
          }
      }),
      ('allow_two_exmemplars_2', 13, {
          'train': {
              'inputs': [111, 222],
              'inputs_num_tokens': [1, 2],
              'targets': [444, 555],
              'targets_num_tokens': [1, 1],
          },
          'eval': {
              'inputs_num_tokens': 5,
              'targets_num_tokens': 1,
              'num_exemplars': 2,
          }
      }),
      ('allow_one_exmemplars', 8, {
          'train': {
              'inputs': [111],
              'inputs_num_tokens': [1],
              'targets': [444],
              'targets_num_tokens': [1],
          },
          'eval': {
              'inputs_num_tokens': 5,
              'targets_num_tokens': 1,
              'num_exemplars': 1,
          }
      }),
      ('allow_none', 1, {
          'train': {
              'inputs': [],
              'inputs_num_tokens': [],
              'targets': [],
              'targets_num_tokens': [],
          },
          'eval': {
              'inputs_num_tokens': 5,
              'targets_num_tokens': 1,
              'num_exemplars': 0,
          }
      }),
  )
  def test_prune_fewshot_examples_by_length(self, max_input_length, expected):
    example = {
        'train': {
            'inputs': tf.constant([111, 222, 333], dtype=tf.int32),
            'inputs_num_tokens': tf.constant([1, 2, 3], dtype=tf.int32),
            'targets': tf.constant([444, 555, 666], dtype=tf.int32),
            'targets_num_tokens': tf.constant([1, 1, 1], dtype=tf.int32),
        },
        'eval': {
            'inputs_num_tokens': tf.constant(5, dtype=tf.int32),
            'targets_num_tokens': tf.constant(1, dtype=tf.int32),
        }
    }
    rst = preprocessors.prune_fewshot_examples_by_length.__wrapped__(
        example, max_input_length)

    self.assertAllClose(self.evaluate(rst), expected)


if __name__ == '__main__':
  absltest.main()
