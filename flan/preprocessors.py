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

"""SeqIO preprocessors for FLAN."""
import functools
import re

import seqio
from t5.data import preprocessors as t5_prep
import tensorflow.compat.v1 as tf


def tokenize(dataset,
             output_features,
             keys=('inputs', 'targets'),
             vocab_key='targets',
             **unused_kwargs):
  """Tokenize the given keys, using the vocabulary for vocab_key.

  Args:
    dataset: A tf.data.Dataset to process.
    output_features: a dict mapping feature name to seqio.Feature.
    keys: a list of strings
    vocab_key: a string

  Returns:
    A preprocessed tf.data.Dataset.
  """

  def my_fn(features):
    """Map function."""
    for k in keys:
      features[k + '_pretokenized'] = features[k]
      features[k] = output_features[vocab_key].vocabulary.encode_tf(features[k])
    return features

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def format_from_feature_dictionary(format_string, feature_dictionary):
  """Create strings based on a format string and a dictionary of tf.Tensors.

  If a key from feature_dictionary appears in curly-braces in format_string,
  then the associated value is inserted.  Non-string types are converted
  with tf.strings.as_string()

  All the referenced keys must have values which are tf.string tensors all
  with the same shape.

  Args:
    format_string: a string, e.g. "inputs: {inputs}, targets: {targets}"
    feature_dictionary: a dictionary string->Tensor

  Returns:
    a string Tensor
  """
  if not format_string:
    return ''
  to_join = []
  parts = [p for p in re.split(r'({\w*})', format_string) if p]
  any_tensors = False
  for part in parts:
    if part[0] == '{' and part[-1] == '}':
      t = feature_dictionary[part[1:-1]]
      if t.dtype != tf.string:
        t = tf.strings.as_string(t)
      to_join.append(t)
      any_tensors = True
    else:
      to_join.append(part)
  if not any_tensors:
    # In the case that the format string contains no references to tensors,
    #   we need to ensure that the output is the correct shape, so we
    #   join on an empty-string tensor of the correct shape.
    to_join = [
        tf.zeros(
            tf.shape(list(feature_dictionary.values())[0]), dtype=tf.string)
    ] + to_join
  return tf.strings.join(to_join)


def negate(dataset, keys, **unused_kwargs):
  """Negate one or more fields.

  Args:
    dataset: A tf.data.Dataset to process.
    keys: a list of strings

  Returns:
    A preprocessed tf.data.Dataset.
  """

  def my_fn(features):
    """Map function."""
    for k in keys:
      features[k] = -features[k]
    return features

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def concatenate(dataset,
                output_features,
                input_keys=('inputs', 'targets'),
                output_key='targets',
                keep_parts=False,
                **unused_kwargs):
  """Concatenate several fields into one.

  Args:
    dataset: A tf.data.Dataset to process.
    output_features: a dict mapping feature name to seqio.Feature.
    input_keys: a list of strings, the ordered keys of features to concatenate
    output_key: a string
    keep_parts: a boolean, whether to also include the original un-concatenated
      features.

  Returns:
    A preprocessed tf.data.Dataset.
  """
  del output_features

  def my_fn(features):
    """Map function."""
    parts = [features[k] for k in input_keys]
    if not keep_parts:
      for k in input_keys:
        del features[k]
    features[output_key] = tf.concat(parts, 0)
    return features

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@seqio.map_over_dataset
def format_options(example):
  """Formats options for FLAN tasks."""
  example['options_'] = tf.strings.reduce_join([
      'OPTIONS:\n- ',
      tf.strings.reduce_join(example['options'], separator='\n- ')
  ])
  return example


def shuffle_dataset(dataset: tf.data.Dataset, buffer_size=50000):
  """Random shuffles the dataset."""
  return dataset.shuffle(buffer_size)


@seqio.map_over_dataset
def reformat_passthrough(example, format_strings):
  """Reformat `example` and pass through existing features."""
  example.update({
      k: format_from_feature_dictionary(v, example)
      for k, v in format_strings.items()
  })
  return example


# A set of FLAN processors for `delimited_lm` tokenizations.
FLAN_TOKENIZE = [
    # This is similar to `tokenize`, but it passes through other features.
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    seqio.preprocessors.append_eos_after_trim,
]

# A set of FLAN processors for `lm` tokenizations.
FLAN_TOKENIZE_LM = [
    shuffle_dataset,
    tokenize,
    # negative ids mean no loss on input segment.
    functools.partial(negate, keys=['inputs']),
    functools.partial(
        concatenate,
        input_keys=['inputs', 'targets'],
        output_key='targets',
    ),
    seqio.CacheDatasetPlaceholder(),
    seqio.preprocessors.append_eos_after_trim,
]


def reformat_with_flan_dialog_prompt(example):
  return reformat_passthrough(
      example,
      format_strings={
          'inputs': '0 {inputs} X 1 FLAN',
          'targets': '{targets}',
      })


def reformat_with_dialog_prompt(example):
  return reformat_passthrough(
      example,
      format_strings={
          'inputs': '0 {inputs} X 1',
          'targets': '{targets}',
      })


def get_flan_formatter(inputs_pattern, targets_pattern):
  """Formats inputs and targets by patterns."""
  return [
      functools.partial(
          reformat_passthrough,
          format_strings={
              'inputs': inputs_pattern,
              'targets': targets_pattern,
          }),
      reformat_with_flan_dialog_prompt,
  ]


def get_dialog_formatter(inputs_pattern, targets_pattern):
  """Formats inputs and targets by patterns, without FLAN."""
  return [
      functools.partial(
          reformat_passthrough,
          format_strings={
              'inputs': inputs_pattern,
              'targets': targets_pattern,
          }),
      reformat_with_dialog_prompt,
  ]


def get_glm_formatter(inputs_pattern, targets_pattern):
  """Formats inputs and targets by patterns, without FLAN."""
  return [
      functools.partial(
          reformat_passthrough,
          format_strings={
              'inputs': inputs_pattern,
              'targets': targets_pattern,
          }),
      functools.partial(
          reformat_passthrough,
          format_strings={
              'inputs': '{inputs}',
              'targets': '{targets}',
          }),
  ]


# ============ Within a single task, apply multiple templates. =================


def example_batch_to_list(example_batch, num_templates):
  """Convert a single batch item in a dataset to a list of items.

  Say you have a dataset where each item is shape {question: (), answer: ()}.
  An example_batch will be a batched example with shape {question: (None,),
  answer: (None,)}.
  This will convert this example_batch to a list of examples, each with shape
  {question: (), answer: ()}.

  Args:
    example_batch: a single batch item in a dataset
    num_templates: the number of templates that are written, equal to batch size

  Returns:
    A list of items.
  """
  return [
      {k: v[i] for k, v in example_batch.items()} for i in range(num_templates)
  ]


def example_list_to_batch(example_list):
  """Reverse of the above example_batch_to_list function."""
  d = {}
  for k in example_list[0]:
    d[k] = tf.stack([example_list[i][k] for i in range(len(example_list))])
  return d


def reformat_single_example(example, patterns_list, i):
  """Formats an example into inputs and targets."""
  inputs_pattern, targets_pattern = patterns_list[i]
  format_strings = {'inputs': inputs_pattern, 'targets': targets_pattern}
  new_example = dict(example)
  for f_name, format_str in format_strings.items():
    new_example[f_name] = format_from_feature_dictionary(format_str, example)
  return new_example


def reformat_batched_example(example_batch, patterns_list):
  """Formats a batch of examples into inputs and targets."""
  example_list = example_batch_to_list(example_batch, len(patterns_list))
  reformatted_batch = [
      reformat_single_example(example, patterns_list, i)
      for i, example in enumerate(example_list)
  ]
  return example_list_to_batch(reformatted_batch)


def batch_apply_template(dataset, patterns_list):
  """Batch a dataset, apply the template to the batch, and then unbatch it."""
  apply_template_to_batch = functools.partial(
      reformat_batched_example, patterns_list=patterns_list)

  # First, strip the dataset of unbatchable features.
  dataset_batchable_features_only = remove_unbatchable_items_ds(
      dataset, training_keys=get_training_keys(patterns_list))

  # Batch the dataset, in preparation for applying apply_template_to_batch.
  dataset_batched = dataset_batchable_features_only.batch(
      len(patterns_list), drop_remainder=True)

  # Apply templating, then unbatch again.
  dataset_templated = dataset_batched.map(
      apply_template_to_batch, num_parallel_calls=tf.data.AUTOTUNE).unbatch()

  # Now that we are unbatched, merge back the unbatchable features.
  zipped_datasets = tf.data.Dataset.zip((dataset, dataset_templated))

  def merge_features(ex1, ex2):
    new_example = dict(ex1)
    for key, value in ex2.items():
      new_example[key] = value
    return new_example

  return zipped_datasets.map(
      merge_features, num_parallel_calls=tf.data.AUTOTUNE)


def remove_unbatchable_items_ex(ex, training_keys):
  return {k: v for k, v in ex.items() if k in training_keys}


def remove_unbatchable_items_ds(ds, training_keys):
  return ds.map(
      functools.partial(
          remove_unbatchable_items_ex, training_keys=training_keys))


def get_training_keys(patterns_list):
  """Get the feature keys that are actually needed for training."""

  def parse_brackets(format_string):
    parts = [p for p in re.split(r'({\w*})', format_string) if p]
    parts = [part[1:-1] for part in parts if part[0] == '{' and part[-1] == '}']
    return parts

  training_keys = set()
  for pattern in patterns_list:
    for s in pattern:
      keys = parse_brackets(s)
      for key in keys:
        training_keys.add(key)
  if 'options_' in training_keys:
    training_keys.add('options')
  return training_keys


def get_batch_flan_formatter(patterns_list):
  """This function applies several templates within a task."""
  return [
      functools.partial(batch_apply_template, patterns_list=patterns_list),
      reformat_with_flan_dialog_prompt,
  ]


def rank_classification_from_options(
    ds: tf.data.Dataset,
    glm_style: bool = False,
):
  """Prepares the example for rank classification eval."""

  def inputs_fn(example):
    options_key_name = 'options'
    if glm_style and 'glm_options' in example:
      options_key_name = 'glm_options'
    num_classes = tf.size(example[options_key_name])
    return tf.tile(tf.reshape(example['inputs'], [1]), [num_classes])

  def target_fn(example):
    options_key_name = 'options'
    if glm_style and 'glm_options' in example:
      options_key_name = 'glm_options'
    return example[options_key_name]

  def is_correct_fn(example):
    options_key_name = 'options'
    answers_key_name = 'answers'
    answer_key_name = 'answer'
    if glm_style:
      if 'glm_options' in example:
        options_key_name = 'glm_options'
      if 'glm_answers' in example:
        answers_key_name = 'glm_answers'
      if 'glm_answer' in example:
        answer_key_name = 'glm_answer'

    # Return multi-hot encoding if multiple correct answers.
    if answers_key_name in example:
      matrix = tf.equal(
          tf.expand_dims(example[options_key_name], 1),
          tf.expand_dims(example[answers_key_name], 0))
      return tf.math.reduce_any(matrix, axis=1)

    # Element-wise equality.
    return tf.equal(example[options_key_name],
                    tf.reshape(example[answer_key_name], []))

  return t5_prep.rank_classification(
      ds,
      inputs_fn,
      target_fn,
      is_correct_fn,
      mode='eval',
  )


GLM_RANK_CLASSIFICATION = functools.partial(
    rank_classification_from_options, glm_style=True)


@seqio.utils.map_over_dataset
def remove_trailing_spaces(example, features):
  """Removes trailing white spaces from `inputs`."""
  new_example = dict(example)
  for name in features:
    new_example[name] = tf.strings.strip(example[name])
  return new_example


@seqio.map_over_dataset
def get_fewshot_num_tokens(
    example, output_features: seqio.preprocessors.OutputFeaturesType):
  """Computes number of tokens for examples from FewshotDataSource."""
  for split in ['train', 'eval']:
    for key in ['inputs', 'targets']:
      if key not in output_features:
        raise ValueError(
            'Feature `%s` not in `output_features`. Cannot perform tokenization.'
            % key)
      vocab = output_features[key].vocabulary
      example[split][f'{key}_num_tokens'] = tf.reduce_sum(
          tf.ones_like(vocab.encode_tf(example[split][key]), dtype=tf.int32),
          axis=-1)
  return example


@seqio.map_over_dataset
def prune_fewshot_examples_by_length(example, max_input_length):
  """Prunes execessive exemplars by max input length."""
  inputs_num_tokens = example['train']['inputs_num_tokens']
  targets_num_tokens = example['train']['targets_num_tokens']
  total_num_tokens = inputs_num_tokens + targets_num_tokens
  total_num_tokens_cm = tf.cumsum(total_num_tokens)
  bool_mask = total_num_tokens_cm <= (
      max_input_length - example['eval']['inputs_num_tokens'])

  # Prunes execssive exemplars.
  for name in ['inputs', 'targets', 'inputs_num_tokens', 'targets_num_tokens']:
    example['train'][name] = tf.boolean_mask(example['train'][name], bool_mask)

  example['eval']['num_exemplars'] = tf.size(
      example['train']['inputs_num_tokens'])
  return example


@seqio.utils.map_over_dataset
def add_delimiter_after_x(ex, x_y_delimiter=' X '):
  new_ex = dict(ex)
  new_ex['inputs'] = tf.strings.join([ex['inputs'], x_y_delimiter])
  return new_ex
