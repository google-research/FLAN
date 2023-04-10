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

"""Seqio preprocessors."""
import functools
import re
from typing import Dict, Iterable, Mapping, Sequence, List

from flan.v2 import templates
import numpy as np
import seqio
import t5
from t5.data import preprocessors as t5_prep
import tensorflow as tf


# NB: We use the whitespace `stripped` version of these option enumerators
# as the targets for eval sets with answer options.
CHAR_OPTIONS = tf.constant(
    [f"({chr(x)})" for x in range(ord("A"),
                                  ord("Z") + 1)])


def tokenize(dataset,
             output_features,
             keys=("inputs", "targets"),
             vocab_key="targets",
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
      features[k + "_pretokenized"] = features[k]
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
    return ""
  to_join = []
  parts = [p for p in re.split(r"({\w*})", format_string) if p]
  any_tensors = False
  for part in parts:
    if part[0] == "{" and part[-1] == "}":
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
                input_keys=("inputs", "targets"),
                output_key="targets",
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
def format_options(example, use_char_options_format: bool = False):
  """Formats options."""
  if use_char_options_format:
    options_prefix = "OPTIONS:"
    separator = ""
    char_options = tf.constant(
        [f"\n({chr(x)}) " for x in range(ord("A"),
                                         ord("Z") + 1)])
    options = tf.reshape(
        tf.stack([char_options[:len(example["options"])], example["options"]],
                 axis=1), [-1])
  else:
    options_prefix = "OPTIONS:\n- "
    separator = "\n- "
    options = example["options"]

  example["options_"] = tf.strings.reduce_join(
      [options_prefix,
       tf.strings.reduce_join(options, separator=separator)])
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


# A set of processors for `delimited_lm` tokenizations.
FLAN_TOKENIZE = [
    # This is similar to `tokenize`, but it passes through other features.
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    seqio.preprocessors.append_eos_after_trim,
]

# A set of processors for `lm` tokenizations.
FLAN_TOKENIZE_LM = [
    tokenize,
    # negative ids mean no loss on input segment.
    functools.partial(negate, keys=["inputs"]),
    functools.partial(
        concatenate,
        input_keys=["inputs", "targets"],
        output_key="targets",
    ),
    seqio.CacheDatasetPlaceholder(),
]


def reformat_with_dialog_prompt(example):
  return reformat_passthrough(
      example,
      format_strings={
          "inputs": "0 {inputs} X 1",
          "targets": "{targets}",
      })


def get_formatter(inputs_pattern, targets_pattern):
  """Formats inputs and targets by patterns."""
  return [
      functools.partial(
          reformat_passthrough,
          format_strings={
              "inputs": inputs_pattern,
              "targets": targets_pattern,
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
  ret = [
      {k: v[i] for k, v in example_batch.items()} for i in range(num_templates)
  ]
  for idx, x in enumerate(ret):
    x.update({"_template_idx": idx})
  return ret


def example_list_to_batch(example_list):
  """Reverse of the above example_batch_to_list function."""
  d = {}
  for k in example_list[0]:
    d[k] = tf.stack([example_list[i][k] for i in range(len(example_list))])
  return d


def reformat_single_example(example, patterns_list, i):
  """Formats an example into inputs and targets."""
  if len(patterns_list[i]) == 2:
    inputs_pattern, targets_pattern = patterns_list[i]
  else:
    assert len(patterns_list[i]) == 3
    inputs_pattern, targets_pattern, final_input_pattern = patterns_list[i]
  format_strings = {"inputs": inputs_pattern, "targets": targets_pattern}
  new_example = dict(example)
  for f_name, format_str in format_strings.items():
    if "exemplar_inputs_0" in example and f_name == "inputs":
      exp_0 = {"input": example["exemplar_inputs_0"],
               "output": example["exemplar_targets_0"]}
      exp_0_str = format_from_feature_dictionary(inputs_pattern + targets_pattern, exp_0)
      exp_1 = {"input": example["exemplar_inputs_1"],
               "output": example["exemplar_targets_1"]}
      exp_1_str = format_from_feature_dictionary(inputs_pattern + targets_pattern, exp_1)

      #exp_0_str = tf.constant('exp0\n', dtype=tf.string)  # test
      #exp_1_str = tf.constant('exp1\n', dtype=tf.string)  # test

      inputs_format = "{exp_0_str}{exp_1_str}" + inputs_pattern
      inputs = {"exp_0_str": exp_0_str,
                "exp_1_str": exp_1_str,
                "input": example["input"]}

      inputs_with_ex = format_from_feature_dictionary(inputs_format, inputs)
      inputs_with_ex = tf.strings.regex_replace(inputs_with_ex, r" *$", "")

      final_input_pattern = final_input_pattern.replace("{{Definition}}", "{Definition}")
      final_input_pattern = final_input_pattern.replace("{{inputs}}", "{inputs_with_ex}")
      example["inputs_with_ex"] = inputs_with_ex
      example["final_suffix"] = tf.constant("", dtype=tf.string)

      new_example[f_name] = format_from_feature_dictionary(final_input_pattern, example)
    else:
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
  passthrough_keys = [
      "_template_idx", "_task_source", "_task_name",
      "exemplar_inputs_0", "exemplar_targets_0",
      "exemplar_inputs_1", "exemplar_targets_1",
      "Definition", "ex_explanation", "ex_input", "ex_output"]
  dataset_batchable_features_only = remove_unbatchable_items_ds(
      dataset, training_keys=get_training_keys(patterns_list, passthrough_keys))

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


def get_training_keys(patterns_list, add_keys: List[str]):
  """Get the feature keys that are actually needed for training."""

  def parse_brackets(format_string):
    parts = [p for p in re.split(r"({\w*})", format_string) if p]
    parts = [part[1:-1] for part in parts if part[0] == "{" and part[-1] == "}"]
    return parts

  training_keys = set()
  for pattern in patterns_list:
    for s in pattern:
      keys = parse_brackets(s)
      for key in keys:
        training_keys.add(key)
  if "options_" in training_keys:
    training_keys.add("options")
  if add_keys is not None:
    for k in add_keys:
      training_keys.add(k)
  return training_keys


def get_batch_formatter(patterns_list):
  """This function applies several templates within a task."""
  return [
      functools.partial(batch_apply_template, patterns_list=patterns_list),
  ]


def rank_classification_from_options(
    ds: tf.data.Dataset,
    glm_style: bool = False,
):
  """Prepares the example for rank classification eval."""

  def inputs_fn(example):
    options_key_name = "options"
    if glm_style and "glm_options" in example:
      options_key_name = "glm_options"
    num_classes = tf.size(example[options_key_name])
    return tf.tile(tf.reshape(example["inputs"], [1]), [num_classes])

  def target_fn(example):
    options_key_name = "options"
    if glm_style and "glm_options" in example:
      options_key_name = "glm_options"
    return example[options_key_name]

  def is_correct_fn(example):
    options_key_name = "options"
    answers_key_name = "answers"
    answer_key_name = "answer"
    if glm_style:
      if "glm_options" in example:
        options_key_name = "glm_options"
      if "glm_answers" in example:
        answers_key_name = "glm_answers"
      if "glm_answer" in example:
        answer_key_name = "glm_answer"

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
      mode="eval",
  )


GLM_RANK_CLASSIFICATION = functools.partial(
    rank_classification_from_options, glm_style=True)


# Add template metadata
@seqio.utils.map_over_dataset
def add_template_info(ex, template_type):
  new_ex = dict(ex)
  new_ex["_template_type"] = template_type
  return new_ex


# Add source and dataset metadata
@seqio.utils.map_over_dataset
def add_source_info(ex, task_name, task_source):
  new_ex = dict(ex)
  new_ex["_task_name"] = task_name
  new_ex["_task_source"] = task_source
  return new_ex


@seqio.utils.map_over_dataset
def remove_trailing_spaces(example, features):
  r"""Trims white spaces and newlines etc.

  in `example`.

  For example:
    ' \nHello \n  ' --> 'Hello'
    ' \nHello \n \t ' --> 'Hello'

  Args:
    example: input example.
    features: which features in example will be trimed.

  Returns:
    The new example after trimming.
  """
  new_example = dict(example)
  for name in features:
    new_example[name] = tf.strings.strip(example[name])
  return new_example


@seqio.utils.map_over_dataset
def remove_trailing_spaces_escape_newlines(example, features):
  r"""Removes trailing white spaces in `example`.

  Newlines etc won't be removed.

  For example:
    ' \nHello \n    ' --> ' \nHello \n'
    ' \nHello \n \t ' --> ' \nHello \n \t'

  Args:
    example: input example.
    features: which features in example will be trimed.

  Returns:
    The new example after trimming.
  """
  new_example = dict(example)
  for name in features:
    new_example[name] = tf.strings.regex_replace(example[name], r" *$", "")
  return new_example


@seqio.map_over_dataset
def get_fewshot_num_tokens(
    example, output_features: seqio.preprocessors.OutputFeaturesType):
  """Computes number of tokens for examples from FewshotDataSource."""
  for split in ["train", "eval"]:
    for key in ["inputs", "targets"]:
      if key not in output_features:
        raise ValueError(
            "Feature `%s` not in `output_features`. Cannot perform tokenization."
            % key)
      vocab = output_features[key].vocabulary
      example[split][f"{key}_num_tokens"] = tf.reduce_sum(
          tf.ones_like(vocab.encode_tf(example[split][key]), dtype=tf.int32),
          axis=-1)
  return example


@seqio.map_over_dataset
def prune_fewshot_examples_by_template_idx(example):
  """Prunes exemplars that do not have the same template as the input case."""
  fs_template_idx = example["train"]["_template_idx"]
  bool_mask = fs_template_idx == example["eval"]["_template_idx"]

  # Prunes exemplars.
  for name in ["inputs", "targets", "_template_idx"]:
    example["train"][name] = tf.boolean_mask(example["train"][name], bool_mask)

  example["eval"]["num_exemplars"] = tf.size(example["train"]["_template_idx"])
  return example


@seqio.map_over_dataset
def prune_fewshot_examples_by_length(example, max_input_length):
  """Prunes execessive exemplars by max input length."""
  inputs_num_tokens = example["train"]["inputs_num_tokens"]
  targets_num_tokens = example["train"]["targets_num_tokens"]
  total_num_tokens = inputs_num_tokens + targets_num_tokens
  total_num_tokens_cm = tf.cumsum(total_num_tokens)
  bool_mask = total_num_tokens_cm <= (
      max_input_length - example["eval"]["inputs_num_tokens"])

  # Prunes execssive exemplars.
  for name in ["inputs", "targets", "inputs_num_tokens", "targets_num_tokens"]:
    example["train"][name] = tf.boolean_mask(example["train"][name], bool_mask)

  example["eval"]["num_exemplars"] = tf.size(
      example["train"]["inputs_num_tokens"])
  return example


def filter_zero_shot_in_few_shot(dataset):
  """Remove cases that do not have any exemplar."""

  def my_fn(example):
    return tf.greater(example["eval"]["num_exemplars"], 0)

  return dataset.filter(my_fn)


@seqio.utils.map_over_dataset
def add_delimiter_after_x(ex, x_y_delimiter=" X "):
  new_ex = dict(ex)
  new_ex["inputs"] = tf.strings.join([ex["inputs"], x_y_delimiter])
  return new_ex


def numbered_items_str(items: tf.Tensor):
  """Returns a string Tensor of numbered items."""
  num_items = tf.shape(items)[0]
  number_list = tf.strings.as_string(tf.range(1, 1 + num_items, 1))
  numbered_items = tf.strings.join([number_list, items], separator=". ")
  return tf.strings.reduce_join(numbered_items, separator="\n")


@seqio.map_over_dataset
def format_options_non_deterministic(example):
  """Formats options in a variety of ways.

  We take array example["options"] as input, and outputs example["options_"]
  which contains a random format of the options. Note that this supports up to
  26 option items.
  For example:
  example["options"] = tf.constant([["An apple", "A banana", "A cashew"]])
  example["options_"] can be:
    "Select from below.
       + Eat apple;
       + Eat banana;
       + Eat cashew;"

  Note that we change the answer string too. If the options are:
    A. Eat apple; B. Eat banana;
  Then the answer will be changed from "Eat apple"/"Eat banana" to "A."/"B.".

  Args:
    example: dictionary of inputs. It must have an "options" entry.

  Returns:
    The same example dictionary but with the additional "options_" entry.
  """
  opt_start_string_candidates = templates.OPT_START_STRING_CANDIDATES
  opt_item_name_candidates = templates.OPT_ITEM_NAME_CANDIDATES
  opt_item_end_str_candidates = templates.OPT_ITEM_END_STR_CANDIDATES

  def _random_draw(candidates):
    # Need to use tf.random instead of np.random. Otherwise, there won"t be
    # randomness.
    idx = tf.random.uniform(
        shape=[], minval=0, maxval=len(candidates), dtype=tf.dtypes.int64)
    return tf.gather(tf.constant(candidates), idx)

  # Get the start_str.
  start_str = _random_draw(opt_start_string_candidates)

  # Get the item_name for each option.
  item_names = _random_draw(opt_item_name_candidates)

  # Get the end str for each option.
  opt_end_str = _random_draw(opt_item_end_str_candidates)

  options = example["options"]
  item_names = item_names[:len(options)]
  # Interleave item_names, options, and opt_end_str.
  options_ = tf.reshape(
      tf.stack([item_names, options, opt_end_str[:len(options)]], axis=1), [-1])

  example["options_"] = tf.strings.reduce_join([
      start_str,
      tf.strings.reduce_join(options_),
  ])

  # Potentially changed the answer string.
  answer = example["answer"]
  match_arr = tf.equal(tf.strings.strip(answer), tf.strings.strip(options))
  has_match = tf.math.reduce_any(match_arr)
  # Check if the item_names are unique. Examples:
  #   ['\n[a]. ', '\n[b]. ', '\n[c]. ', '\n[d]. ']
  #   ['\n1). ', '\n2). ', '\n3). ', '\n4). ']
  is_abc_style = tf.equal(
      len(tf.unique(tf.strings.strip(item_names))[0]), len(item_names))

  answer_maybe_abc_style = tf.cond(
      tf.math.logical_and(has_match, is_abc_style),
      lambda: tf.boolean_mask(tf.strings.strip(item_names), match_arr)[0],
      lambda: answer)
  example["answer"] = answer_maybe_abc_style

  return example


@seqio.map_over_dataset
def format_dialog_non_deterministic(example):
  """Formats dialog next turn prediction tasks.

  We take array example["dialog"] as input, and outputs example["dialog_"]
  which contains a random format of the dialog. Note that this supports up to
  50 dialog turns.
  For example:
  example["dialog"] = tf.constant([["Hi!", "What"s up?", "Not much."]])
  example["dialog_"] can be:
    "DIALOG:
       Person 1: Hi!
       Person 2: What"s up?
       Person 1: Not much."

  Args:
    example: dictionary of inputs. It must have an "dialog" entry.

  Returns:
    The same example dictionary but with the additional "dialog_" entry.
  """
  dialog_start_string_candidates = templates.DIALOG_START_STRING_CANDIDATES
  dialog_item_name_candidates = templates.DIALOG_ITEM_NAME_CANDIDATES
  dialog_item_end_str_candidates = templates.DIALOG_ITEM_END_STR_CANDIDATES

  def _random_draw(candidates):
    # Need to use tf.random instead of np.random. Otherwise, there won't be
    # randomness.
    idx = tf.random.uniform(
        shape=[], minval=0, maxval=len(candidates), dtype=tf.dtypes.int64)
    return tf.gather(tf.constant(candidates), idx)

  # Get the start_str.
  start_str = _random_draw(dialog_start_string_candidates)

  # Get the item_name for each dialog turn.
  item_names = _random_draw(dialog_item_name_candidates)

  # Get the end str for each dialog turn.
  opt_end_str = _random_draw(dialog_item_end_str_candidates)

  dialog = example["dialog"]
  # Interleave item_names, dialog turns, and opt_end_str.
  dialog = tf.reshape(
      tf.stack([item_names[:len(dialog)], dialog, opt_end_str[:len(dialog)]],
               axis=1), [-1])

  # Add final item name to indicate we need the next dialog turn:
  example["dialog_"] = tf.strings.reduce_join([
      start_str,
      tf.strings.reduce_join(dialog),
      item_names[len(dialog)],
  ])
  return example


@seqio.map_over_dataset
def format_dialog(example):
  """Formats 2 person dialog determinstically."""
  example["dialog_"] = tf.strings.reduce_join([
      "DIALOG:\n",
      tf.strings.reduce_join(example["dialog"], separator="\n- "),
      "\n- ",
  ])
  return example


def rank_classification(ds: tf.data.Dataset,
                        options_key: str = "options",
                        answer_key: str = "answer",
                        use_char_answer: bool = False) -> tf.data.Dataset:
  """Prepares the example for scoring eval.

  Args:
    ds: input dataset
    options_key: a dict key to access the list of classes the input dataset uses
    answer_key: a dict key to access the answer to each example.
    use_char_answer: a bool of whether to use (A), (B), (C),... as the target
      options.  Given a dataset with the following single example:  [{ "inputs":
      "this will be repeated twice.", "targets": "yes", "options":
      tf.constant(["no", "yes"]), "answer": "yes", }] the processed dataset will
      contain the following two examples because the number of classes is two:
      [{ "idx": [0, 0], "inputs": "this will be repeated twice.", "targets":
      "no", "is_correct": False, }, { "idx": [0, 1], "inputs": "this will be
      repeated twice.", "targets": "yes", "is_correct": True,  # because
      groundtruth is "yes" }].

  Returns:
    processed dataset
  """

  def inputs_fn(example):
    num_classes = tf.size(example[options_key])
    return tf.tile(tf.reshape(example["inputs"], [1]), [num_classes])

  def target_fn(example):
    if use_char_answer:
      return CHAR_OPTIONS[:len(example[options_key])]
    else:
      return example[options_key]

  def is_correct_fn(example):
    if use_char_answer:
      ex_options = CHAR_OPTIONS[:len(example[options_key])]
    else:
      ex_options = example[options_key]
    return tf.equal(ex_options, example[answer_key])

  return t5.data.preprocessors.rank_classification(
      ds,
      inputs_fn,
      target_fn,
      is_correct_fn,
      mode="eval",
  )


@seqio.map_over_dataset
def boolq(example: Mapping[str, tf.Tensor],
          use_char_answer: bool = False) -> Mapping[str, tf.Tensor]:
  """Processes the BoolQ from TFDS.

  Args:
    example: a mapping from a feature name to a dict
    use_char_answer: whether to set the answer as the string (False), or the
      option letter (True).  An input example  { "answer": False, "passage":
      "dummy passage.", "question": "dummy question?", "title": "dummy title", }
      is processed to  { "title": "dummy title", "passage": "dummy passage.",
      "question": "dummy question?", "options": ["no", "yes"], "answer": "no", }

  Returns:
    processed example dict
  """
  one_hot = tf.one_hot(tf.cast(example["answer"], tf.int32), 2)
  options = tf.constant(["no", "yes"])
  if use_char_answer:
    answer = tf.boolean_mask(CHAR_OPTIONS[:len(one_hot)], one_hot)[0]
  else:
    answer = tf.boolean_mask(options, one_hot)[0]
  return {
      "title": example["title"],
      "passage": example["passage"],
      "question": example["question"],
      "options": options,
      "answer": answer,
  }


@seqio.map_over_dataset
def rte(
    example: Mapping[str, tf.Tensor],
    use_legacy_template: bool = False,
    use_char_answer: bool = False,
) -> Mapping[str, tf.Tensor]:
  """Processes the RTE (SuperGLUE version) from TFDS.

  Args:
    example: a mapping from a feature name to a dict
    use_legacy_template: whether to use the PaLM template with lower performance
    use_char_answer: whether to set the answer as the string (False), or the
      option letter (True).  An input example  { "hypothesis": "dummy
      hypothesis", "idx": 42, "label": 1, "premise": "dummy premise", } is
      processed to  { "premise": "dummy premise", "hypothesis": "dummy
      hypothesis", "options": ["yes", "no"], "answer": "no", }

  Returns:
    processed example dict
  """
  one_hot = tf.one_hot(tf.cast(example["label"], tf.int32), 2)
  if use_legacy_template:
    # following gpt-3 convention
    options = tf.constant(["true", "false"])
  else:
    options = tf.constant(["yes", "no"])

  if use_char_answer:
    answer = tf.boolean_mask(CHAR_OPTIONS[:len(one_hot)], one_hot)[0]
  else:
    answer = tf.boolean_mask(options, one_hot)[0]
  return {
      "premise": example["premise"],
      "hypothesis": example["hypothesis"],
      "options": options,
      "answer": answer,
  }


@seqio.map_over_dataset
def arc(example: Mapping[str, tf.Tensor],
        use_char_answer: bool = False) -> Mapping[str, tf.Tensor]:
  """Process the ARC from TFDS.

  Args:
    example: a mapping from a feature name to a dict
    use_char_answer: whether to set the answer as the string (False), or the
      option letter (True).  An input example  { "answerKey": 1, "choices": {
      "label": [0, 1, 2, 3], "text": [ "a balance", "a metric ruler", "a
      graduated cylinder", "a thermometer" ], }, "id": "Mercury_7081655",
      "question": "Which tool should be used to measure the stem length of a
      plant?" } is processed to  { "answer": "a metric ruler", "options": [ "a
      balance", "a metric ruler", "a graduated cylinder", "a thermometer" ],
      "question": "Which tool should be used to measure the stem length of a
      plant?" }

  Returns:
    processed example dict
  """
  num_labels = len(example["choices"]["text"])
  one_hot = tf.one_hot(tf.cast(example["answerKey"], tf.int32), num_labels)
  options = example["choices"]["text"]
  if use_char_answer:
    answer = tf.boolean_mask(CHAR_OPTIONS[:len(one_hot)], one_hot)[0]
  else:
    answer = tf.boolean_mask(options, one_hot)[0]
  return {
      "question": example["question"],
      "options": options,
      "answer": answer,
  }


@seqio.map_over_dataset
def anli(
    example: Mapping[str, tf.Tensor],
    use_legacy_template: bool = False,
    use_char_answer: bool = False,
) -> Mapping[str, tf.Tensor]:
  """Processes ANLI R1, R2, R3 from the TFDS source.

  Args:
    example: a mapping from a feature name to a dict
    use_legacy_template: whether to use the PaLM template with lower performance
    use_char_answer: whether to set the answer as the string (False), or the
      option letter (True).  An input example  { "context": "Lofar is a Telugu
      film directed by Puri Jagannadh.", "hypothesis": " Varun Tej had billing
      over Disha Patani in Lofar.", "label": 1, "uid":
      b"8e3ba01d-5bd2-43b4-a154-3eabd4e79c2c" } is processed to  { "answer": "it
      is not possible to tell", "hypothesis": "Varun Tej had billing over Disha
      Patani in Lofar.", "options": ["yes", "it is not possible to tell", "no"],
      "premise": "Lofar is a Telugu film directed by Puri Jagannadh." }

  Returns:
    processed example dict
  """
  one_hot = tf.one_hot(tf.cast(example["label"], tf.int32), 3)
  if use_legacy_template:
    # GPT-3 convention
    options = tf.constant(["true", "neither", "false"])
  else:
    options = tf.constant(["yes", "it is not possible to tell", "no"])

  if use_char_answer:
    answer = tf.boolean_mask(CHAR_OPTIONS[:len(one_hot)], one_hot)[0]
  else:
    answer = tf.boolean_mask(options, one_hot)[0]
  return {
      "premise": example["context"],
      "hypothesis": example["hypothesis"],
      "options": options,
      "answer": answer,
  }


@seqio.map_over_dataset
def simple_cot_tsv(example):
  """Processes a simple tsv file with chain of thought."""
  question = tf.strings.split(example, sep="\t")[0]
  question = tf.strings.regex_replace(question, r"\\n", "\n")
  question = tf.strings.regex_replace(question, r"\n", "\n")
  answer = tf.strings.split(example, sep="\t")[1]
  chain_of_thought = tf.strings.split(example, sep="\t")[2]
  chain_of_thought = tf.strings.regex_replace(chain_of_thought, r"\\n", "\n")
  chain_of_thought = tf.strings.regex_replace(chain_of_thought, r"\n", "\n")
  return {
      "question": question,
      "answer": answer,
      "chain_of_thought": chain_of_thought,
  }


@seqio.map_over_dataset
def simple_tsv(example):
  # pyformat: disable
  r"""Processes a simple tsv file.

  Args:
    example: a string, with question and answer separated by a tab"

  An input example

      "Each pack of dvds costs 76 dollars. If there is a discount of 25 dollars on each pack. How much do you have to pay to buy each pack?\t51.0"
  is processed to

  {
      "question": "Each pack of dvds costs 76 dollars. If there is a discount of 25 dollars on each pack. How much do you have to pay to buy each pack?",
      "answer": "51.0",
  }

  Returns:
    processed example dict
  """
  # pyformat: enable
  question = tf.strings.split(example, sep="\t")[0]
  question = tf.strings.regex_replace(question, r"\\n", "\n")
  question = tf.strings.regex_replace(question, r"\n", "\n")
  answer = tf.strings.split(example, sep="\t")[1]
  return {
      "question": question,
      "answer": answer,
  }


@seqio.map_over_dataset
def strategyqa(example):
  r"""Processes StrategyQA.

  Args:
    example: a string containing "Yes or no: {question}\t{answer}"  An input
      example  "Yes or no: I like apple, so I probably like apple juice,
      right?\tyes" is processed to  { "question": "I like apple, so I probably
      like apple juice, right?", "answer": "yes", }

  Returns:
    processed example dict
  """
  question = tf.strings.split(example, sep="\t")[0]
  question = tf.strings.split(question, sep="Yes or no: ")[1]
  answer = tf.strings.split(example, sep="\t")[1]
  return {
      "question": question,
      "answer": answer,
  }


def filter_unified_qa_science_inst(dataset):
  """Remove cases that do not have exactly 4 options."""

  def my_fn(example):
    input_str = example["input"]
    has_option_d = len(tf.strings.split(input_str, sep="(D)")) > 1
    has_option_e = len(tf.strings.split(input_str, sep="(E)")) > 1
    return has_option_d and (not has_option_e)

  return dataset.filter(my_fn)


@seqio.map_over_dataset
def unified_qa_science_inst(example):
  r"""Processes unified_qa, ai2_science_middle.

  Args:
    example: a mapping from a feature name to a dict  An input example  {
      "input": "What is the largest insect in the world? \\n " "(A) Giant wētā.
      (B) Ant. (C) Lion. (D) Flea Beetle.", "output": "Giant wētā.", } is
      processed to  { "question": "What is the largest insect in the world?",
      "options": ["Giant wētā.", "Ant.", "Lion.", "Flea Beetle."], "answer":
      "Giant wētā.", }

  Returns:
    processed example dict
  """
  entire_input = example["input"]
  entire_input = tf.strings.regex_replace(entire_input, r"\\n", "\n")
  entire_input = tf.strings.regex_replace(entire_input, r"\n", "\n")
  entire_output = example["output"]
  entire_output = tf.strings.regex_replace(entire_output, r"\\n", "\n")
  entire_output = tf.strings.regex_replace(entire_output, r"\n", "\n")

  question = tf.strings.strip(tf.strings.split(entire_input, sep="(A)")[0])
  option_a = tf.strings.strip(
      tf.strings.split(tf.strings.split(entire_input, sep="(A)")[1], "(B)")[0])
  option_b = tf.strings.strip(
      tf.strings.split(tf.strings.split(entire_input, sep="(B)")[1], "(C)")[0])
  option_c = tf.strings.strip(
      tf.strings.split(tf.strings.split(entire_input, sep="(C)")[1], "(D)")[0])
  option_d = tf.strings.strip(tf.strings.split(entire_input, sep="(D)")[1])
  options = tf.concat([
      tf.expand_dims(option_a, axis=0),
      tf.expand_dims(option_b, axis=0),
      tf.expand_dims(option_c, axis=0),
      tf.expand_dims(option_d, axis=0)
  ],
                      axis=0)

  return {
      "question": question,
      "options": options,
      "answer": entire_output,
  }


@seqio.map_over_dataset
def hellaswag(example: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
  """Processes HellaSwag from the TFDS source.

  Args:
    example: a mapping from a feature name to a dict  An input example  {
      "activity_label": "Capoeira", "context": "People are dancing on a street
      while clapping their hands. A man " "is bent over in a yellow shirt
      holding sticks. a man", "endings": [ "is standing on the sidewalk watching
      them dance.", "starts doing a flip on the stage.", "is saying something to
      the camera.", "plays with the sticks by tapping them on the ground." ],
      "label": 3, "split_type": "indomain" } is processed to  {
      "activity_label": "Capoeira", "answer": "plays with the sticks by tapping
      them on the ground.", "context": "People are dancing on a street while
      clapping their hands. A man " "is bent over in a yellow shirt holding
      sticks. a man", "options": [ "is standing on the sidewalk watching them
      dance.", "starts doing a flip on the stage.", "is saying something to the
      camera.", "plays with the sticks by tapping them on the ground." }

  Returns:
    processed example dict
  """
  num_labels = len(example["endings"])
  one_hot = tf.one_hot(example["label"], num_labels)
  options = example["endings"]
  return {
      "activity_label": example["activity_label"],
      "context": example["context"],
      "options": options,
      "answer": tf.boolean_mask(options, one_hot)[0],
  }


@seqio.map_over_dataset
def arithmetic_addition(example):
  # pyformat: disable
  r"""Processes arithmetic addition data.

  Args:
    example: a string containing ""

  An input example
    "I am a perfect calculator and I can't...Example 4:\n{x1} + {x2}\nStarting from rightmost column:\t1536\t3"
  is processed to

  {
      "question": "I am a perfect calculator and I can't...Example 4:\n{x1} + {x2}\nStarting from rightmost column:",
      "answer": "1536",
      "num_digits": "3",
  }

  Returns:
    processed example dict
  """
  # pyformat: enable
  question, answer, num_digits = tf.io.decode_csv(
      example,
      record_defaults=[""] * 3,
      field_delim="\t",
      use_quote_delim=False)
  question = tf.strings.regex_replace(question, r"\\n", "\n")
  question = tf.strings.regex_replace(question, r"\n", "\n")
  output = dict(
      zip(["question", "answer", "num_digits"], (question, answer, num_digits)))
  return output


@seqio.map_over_dataset
def trivia_qa(example: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
  """Processes TriviaQA example from the TFDS source.

  Args:
    example: a mapping from a feature name to a dict  An input example {
      "answer": { "aliases": ["Torquemada (disambiguation)", "Torquemada"],
      "matched_wiki_entity_name": "", "normalized_aliases": ["torquemada",
      "torquemada disambiguation"], "value": "Torquemada", }, "question": "In
      1483, who was appointed the first grand inquisitor of the " "Spanish
      Inquisition?", } is processed to  { "answer": "Torquemada", "answers":
      ["Torquemada (disambiguation)", "Torquemada"], "question": "In 1483, who
      was appointed the first grand inquisitor of the " "Spanish Inquisition?",
      }

  Returns:
    processed example dict
  """
  return {
      "question": example["question"],
      "answer": example["answer"]["value"],
      "answers": example["answer"]["aliases"],
  }


@seqio.map_over_dataset
def tydiqa(example: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
  """Processes TriviaQA example from the TFDS source.

  Args:
    example: a mapping from a feature name to a dict

  Returns:
    processed example dict
  """
  answers = example["answers"]["text"]
  question = tf.strings.strip(example["question"])
  context = tf.strings.strip(example["context"])
  return {
      "question": question,
      "context": context,
      "answer": answers[0],
      "answers": answers,
  }


@seqio.map_over_dataset
def nq_open(example: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
  """Processes natural_questions_open example from the TFDS source.

  Args:
    example: a mapping from a feature name to a dict  An input example {
      "answer":  ["English", "Welsh", "Irish", "German"], "question": "where
      does the last name rice come from", } is processed to  { "answer":
      "English, Welsh, Irish, German", "answers":  ["English", "Welsh", "Irish",
      "German"], "question": "where does the last name rice come from", }

  Returns:
    processed example dict
  """
  return {
      "question": example["question"],
      "answer": tf.strings.reduce_join(example["answer"], separator=", "),
      "answers": example["answer"],
      "first_answer": example["answer"][0],
  }


@seqio.map_over_dataset
def wiki_dialog(
    example: Mapping[str, tf.Tensor],
    random_turn: bool = True,
) -> Mapping[str, tf.Tensor]:
  # pyformat: disable
  """Processes Wiki Dialog example from the TFDS source.

  Args:
    example: a mapping from a feature name to a dict
    random_turn: Whether to sample a random answer turn, or always use the first
      for determinsim and testing.

  An input example
  {
      "utterances": np.array([
          b"I am your assistant, answering questions about Kesha."
          b"who sang tik tok and we are who we are",
          b"KTik Tok and We R Who We R are songs by the American singer Kesha.",
          b"When was the release?",
          b"In July 2009, Tik Tok was offered as a free download on Kesha's Myspace page for over a month before its official sale release.",
          b"How about We Are?",
          b"Kesha's We R Who We R was released October 22, 2010.",
          b"Where did the song chart?",
          b"We R Who We R debuted at number one on the Billboard Hot 100, making it the 17th song in the chart's history to do so.",
          b"Did Kesha have any more?",
          b"Tik Tok topped the Billboard Hot 100 chart for 9 consecutive weeks."
      ]),
  }
  is processed to

  {
      "dialog": np.array([
          b"who sang tik tok and we are who we are",
          b"KTik Tok and We R Who We R are songs by the American singer Kesha.",
          b"When was the release?",
          b"In July 2009, Tik Tok was offered as a free download on Kesha's Myspace page for over a month before its official sale release.",
          b"How about We Are?"
      ]),
      "answer": b"Kesha's We R Who We R was released October 22, 2010."
  }

  Returns:
    processed example dict
  """
  # pyformat: enable
  dialog = example["utterances"][1:]
  if random_turn:
    maxval = tf.math.minimum(tf.cast(len(dialog) / 2, tf.dtypes.int64), 20)
    idx = tf.random.uniform(
        shape=[], minval=0, maxval=maxval, dtype=tf.dtypes.int64) * 2 + 1
  else:
    idx = 1
  return {
      "dialog": dialog[:idx],
      "answer": dialog[idx],
      "answers": tf.convert_to_tensor([dialog[idx]]),
  }


@seqio.map_over_dataset
def task_master(
    example: Mapping[str, tf.Tensor],
    random_turn: bool = True,
) -> Mapping[str, tf.Tensor]:
  # pyformat: disable
  """Processes a Task Master example from the TFDS source.

  Args:
    example: a mapping from a feature name to a dict
    random_turn: Whether to sample a random answer turn, or always use the first
      for determinsim and testing.

  An input example
  {
      "text": [
          b'I want to schedule an appointment for a car repair at Intelligent Auto Solutions.',
          b'Sure, what is you name and number?',
          b"I'm Sarah and my number is 555-5555.",
          b'What seems to be the issue?', b"I'm having engine trouble.",
          b'What type of car do you own?', b'I have a 2005 Honda Odyssey',
          b'Can you tell me more about the issue?',
          b'Yes, my car keeps stalling in traffic and having trouble starting back up again.',
          b'Can you drop your car off tomorrow at 3:00PM? They have an appointment then.',
          b'I really need my car sooner.  We are going to a wedding tomorrow and I need it fixed ASAP!',
          b'Well they have a late appointment at 8:00 PM tonight.',
          b'Yes, that will have to work. ',
          b'I will add your appointment to the schedule.', b'Thank you.',
          b'So, you have an appointment booked for tonight at 8PM at Intelligent Auto Solutions. The fee will be $55 for the initial inspection.',
          b'Okay.',
          b'Great. Your appointment is confirmed and they will see you at 8PM.',
          b'Thanks a lot.', b"You're welcome.", b'Bye.', b'Bye.'
      ]
  }
  is processed to

  {
      "dialog": [
          b'I want to schedule an appointment for a car repair at Intelligent Auto Solutions.',
          b'Sure, what is you name and number?',
          b"I'm Sarah and my number is 555-5555.",
          b'What seems to be the issue?', b"I'm having engine trouble.",
          b'What type of car do you own?', b'I have a 2005 Honda Odyssey',
          b'Can you tell me more about the issue?',
          b'Yes, my car keeps stalling in traffic and having trouble starting back up again.',
      ],
      "answer": b"Yes, my car keeps stalling in traffic and having trouble starting back up again."
  }

  Returns:
    processed example dict
  """
  # pyformat: enable
  dialog = example["text"]
  if random_turn:
    maxval = tf.math.minimum(tf.cast(len(dialog), tf.dtypes.int64), 20)
    idx = tf.random.uniform(
        shape=[], minval=1, maxval=maxval, dtype=tf.dtypes.int64)
  else:
    idx = 1
  return {
      "dialog": dialog[:idx],
      "answer": dialog[idx],
      "answers": tf.convert_to_tensor([dialog[idx]]),
  }


def filter_qrecc(dataset):
  def filter_func(example):
    return example["turn_id"] > 6
  return dataset.filter(filter_func)


@seqio.map_over_dataset
def qrecc(
    example: Mapping[str, tf.Tensor],
    random_turn: bool = True,
) -> Mapping[str, tf.Tensor]:
  # pyformat: disable
  """Processes QReCC example from the TFDS source.

  Args:
    example: a mapping from a feature name to a dict
    random_turn: Whether to sample a random answer turn, or always use the first
      for determinsim and testing.

  An input example
  {
      "history_with_truth_answer": np.array([
          b"who sang tik tok and we are who we are",
          b"KTik Tok and We R Who We R are songs by the American singer Kesha.",
          b"When was the release?",
          b"In July 2009, Tik Tok was offered as a free download on Kesha's Myspace page for over a month before its official sale release.",
          b"How about We Are?",
          b"Kesha's We R Who We R was released October 22, 2010.",
          b"Where did the song chart?",
          b"We R Who We R debuted at number one on the Billboard Hot 100, making it the 17th song in the chart's history to do so.",
          b"Did Kesha have any more?",
          b"Tik Tok topped the Billboard Hot 100 chart for 9 consecutive weeks."
      ]),
  }
  is processed to

  {
      "dialog": np.array([
          b"who sang tik tok and we are who we are",
          b"KTik Tok and We R Who We R are songs by the American singer Kesha.",
          b"When was the release?",
          b"In July 2009, Tik Tok was offered as a free download on Kesha's Myspace page for over a month before its official sale release.",
          b"How about We Are?"
      ]),
      "answer": b"Kesha's We R Who We R was released October 22, 2010."
  }

  Returns:
    processed example dict
  """
  # pyformat: enable
  dialog = example["context"]
  if random_turn:
    maxval = tf.math.minimum(tf.cast(len(dialog) / 2, tf.dtypes.int64), 20)
    idx = tf.random.uniform(
        shape=[], minval=0, maxval=maxval, dtype=tf.dtypes.int64)
    idx = int(idx * 2 + 1)
  else:
    idx = 1
  return {
      "dialog": dialog[:idx],
      "answer": dialog[idx],
      "answers": tf.convert_to_tensor([dialog[idx]]),
  }


@seqio.map_over_dataset
def t0(example: tf.Tensor,
       multiple_choice: bool,
       use_char_answer: bool = False) -> Mapping[str, tf.Tensor]:
  """Processes T0/P3.

  Args:
    example: a Tensor representing a line in a tsv file.
    multiple_choice: Whether it is a multiple choice type example
    use_char_answer: whether to set the answer as the string (False), or the
      option letter (True).  An input example  { "inputs_pretokenized": ...
      "targets_pretokenized": ... "answer_choices": ... }  is processed to  {
      "question": "which is the third letter in the alphabet?", "options":
      ["answer a", "answer b", "answer c", "answer d], "answer": "answer c" }

  Returns:
    processed example dict
  """
  query = tf.strings.strip(
      tf.strings.regex_replace(example["inputs_pretokenized"], r"\\n", " "))
  query = tf.strings.strip(
      tf.strings.regex_replace(example["inputs_pretokenized"], r"\n", " "))
  answer = tf.strings.strip(
      tf.strings.regex_replace(example["targets_pretokenized"], r"\\n", ""))
  answer = tf.strings.strip(
      tf.strings.regex_replace(example["targets_pretokenized"], r"\n", ""))
  if multiple_choice:
    fn = lambda x: tf.strings.strip(tf.strings.regex_replace(x, r"\\n", ""))
    options = tf.map_fn(fn, example["answer_choices"])
    fn = lambda x: tf.strings.strip(tf.strings.regex_replace(x, r"\n", ""))
    options = tf.map_fn(fn, example["answer_choices"])
    answer_key = tf.where(tf.equal(options, answer))[0][0]
    one_hot = tf.one_hot(answer_key, len(options))
    if use_char_answer:
      answer = tf.boolean_mask(CHAR_OPTIONS[:len(one_hot)], one_hot)[0]
    else:
      answer = tf.boolean_mask(options, one_hot)[0]
    out_tensors = {
        "question": query,
        "options": options,
        "answer": answer,
    }
  else:
    out_tensors = {"question": query, "answer": answer}
  return out_tensors


@seqio.map_over_dataset
def drrepair(example: tf.Tensor) -> Mapping[str, tf.Tensor]:
  r"""Processes DrRepair.

  Args:
    example: a Tensor representing a line in a tsv file.  An input example
      tf.Tensor(<incorrect_code>, <correct_code>)  is processed to  {
      "question": "import os;\\nos.path.jjin('path1', 'path2')", "answer":
      "import os;\nos.path.join('path1', 'path2')" }

  Returns:
    processed example dict
  """
  # replace \\n by \n
  example = tf.strings.regex_replace(example, r"```", "")
  example = tf.strings.regex_replace(example, r"\\n", "\n")
  example = tf.strings.split(example, sep="\t")
  return {
      "question": example[0],
      "answer": example[1],
  }


@seqio.map_over_dataset
def dmcc(example: tf.Tensor) -> Mapping[str, tf.Tensor]:
  # pyformat: disable
  """Processes DeepMind CodeContests: https://github.com/deepmind/code_contests.

  Args:
    example: a TFTensor representing a coding problem.

  An input example

  {
      "inputs": "[code]Write code that adds two numbers.[BEGIN]"
      "targets": "def add(a,b): return a+b[DONE]."
  }

  is processed to

  {
      "question": "Write code that adds two numbers.",
      "answer": "def add(a,b): return a+b"
  }

  Returns:
    processed example dict
  """
  # pyformat: enable
  ex_inputs = tf.strings.regex_replace(example["inputs"], r"\[code\]", "")
  ex_inputs = tf.strings.regex_replace(ex_inputs, r"\[BEGIN\]", "")
  ex_targets = tf.strings.regex_replace(example["targets"], r"\[code\]", "")
  ex_targets = tf.strings.regex_replace(ex_targets, r"\[DONE\]", "")
  ex_targets = tf.strings.join([ex_targets, b"\n\n\n\n"])
  return {
      "question": ex_inputs,
      "answer": ex_targets,
  }


@seqio.map_over_dataset
def niv2_few_shot_exemplar_lookup_fn(
    example,
    exemplar_input_lookup,
    exemplar_targets_lookup,
):
  """Lookup positive example fields and populate the dataset example."""
  task_key = tf.strings.split(example["task_name"], sep=".")[0]
  example["exemplar_inputs"] = tf.strings.split(
      exemplar_input_lookup.lookup(task_key),
      sep='\nflanv2-separator\n')
  example["exemplar_targets"] = tf.strings.split(
      exemplar_targets_lookup.lookup(task_key),
      sep='\nflanv2-separator\n')

  n = len(example["exemplar_inputs"])
  sample_size = tf.constant(2)  # 2 exemplars.
  idx = tf.random.shuffle(tf.range(n))[:sample_size]
  exemplar_inputs = tf.gather(example["exemplar_inputs"], idx)
  exemplar_targets = tf.gather(example["exemplar_targets"], idx)
  example["exemplar_inputs_0"] = exemplar_inputs[0]
  example["exemplar_targets_0"] = exemplar_targets[0]
  example["exemplar_inputs_1"] = exemplar_inputs[1]
  example["exemplar_targets_1"] = exemplar_targets[1]
  return example


@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
def tf_capitalize(tensor: tf.Tensor) ->...:
  """Python function for capitalization."""

  def np_func(text: np.ndarray) -> np.ndarray:
    func = lambda s: s.decode().capitalize().encode()
    applyall = np.vectorize(func)
    return applyall(text)

  return tf.numpy_function(np_func, [tensor], tf.string)


@seqio.map_over_dataset
def capitalize(example: tf.Tensor,
               text_features: Iterable[str]) -> Mapping[str, tf.Tensor]:
  """Capitalizes text features."""
  for feature in text_features:
    example[feature] = tf_capitalize(example[feature])
  return example


@seqio.map_over_dataset
def add_constant_text_features(
    example: Dict[str, tf.Tensor],
    text_features: Mapping[str, Sequence[str]]) -> Dict[str, tf.Tensor]:
  """Adds constant text features to the example."""
  for key, texts in text_features.items():
    example[key] = tf.constant(texts, dtype=tf.string)
  return example


def filter_non_strings(dataset, field):
  """Filter examples that have empty `field`."""

  def my_fn(example):
    dtype_str = (example[field].dtype == tf.string)
    non_empty_str = tf.greater(tf.size(example[field]), 0)
    return tf.logical_and(dtype_str, non_empty_str)

  return dataset.filter(my_fn)


def filter_long_strings(dataset, field, length=384):
  """Filter examples whose `field` are too long."""

  def my_fn(example):
    return tf.less(tf.size(example[field]), length)

  return dataset.filter(my_fn)


@seqio.map_over_dataset
def sparse_to_dense(example, field):
  example[field] = tf.reshape(tf.sparse.to_dense(example[field]), shape=())
  return example


@seqio.map_over_dataset
def strip_the_answer_is_in_cot(example):
  """Remove `answer is` part in CoT."""
  cot = example["cot"]
  cot = tf.strings.split(cot, sep=" The answer is ")[0]
  cot = tf.strings.split(cot, sep=" So the answer is ")[0]
  example["cot"] = cot
  return example


@seqio.map_over_dataset
def strip_field(example, field_key="inputs"):
  """Remove `answer is` part in CoT."""
  example[field_key] = tf.strings.strip(example[field_key])
  return example
