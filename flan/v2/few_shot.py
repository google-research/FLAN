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

"""Utilities for creating few-shot learning tasks."""
import enum
import functools
from typing import Optional

from flan.v2 import preprocessors as prep
import seqio
import tensorflow as tf


class ShotConfig(enum.Enum):
  """Determines how many few-shot exemplars to feature in the input prompt."""
  ZERO = 0  # Provide no few-shot exemplars.
  ONE = 1  # Provide exactly one few-shot exemplar.
  TWO = 2  # Provide exactly two few-shot exemplars.
  THREE = 3  # Provide exactly three few-shot exemplars.
  FIVE = 5  # Provide exactly five few-shot exemplars.
  TEN = 10
  MULTI = 'multi'  # Provide task-specific number of few-shot exemplars.

  @property
  def name_suffix(self) -> str:
    """A suffix that can be added to a Task or Mixture name."""
    if self == ShotConfig.ZERO:
      return ''
    return f'_{self.name.lower()}_shot'


def register_few_shot_version_of_task(
    base_task_name: str,
    new_task_name: str,
    num_shots: int,
    x_y_delimiter: str = '\n\n',
    inputs_prefix: str = '',
    targets_prefix: str = '',
    example_separator: str = '\n\n\n',
    final_suffix: str = '',
    input_pattern: str = '{{inputs}}{final_suffix}',
    prune_exemplars: bool = False,
    max_input_length: Optional[int] = None,
    prune_based_on_template_idx: Optional[bool] = False,
    strip_targets: Optional[bool] = False,
):
  """Registers a few-shot version of a Task."""
  task = seqio.TaskRegistry.get(base_task_name)

  # The list of preprocessors to run on individual exemplars.
  single_ex_preprocessors = list(task.preprocessors)

  def remove_preprocessors_if_present(to_remove):
    """Removes single-example preprocessors if they are present."""
    num_to_remove = len(to_remove)
    if single_ex_preprocessors[-num_to_remove:] == to_remove:
      for _ in range(num_to_remove):
        single_ex_preprocessors.pop()
      return True  # Indicate that preprocessors were present and removed.
    return False

  # We don't want to tokenize individual exemplars. We want to postpone
  # tokenization until after we have formed few-shot examples. So, remove the
  # tokenization steps from `single_ex_preprocessors`.
  flan_tokenize_removed = remove_preprocessors_if_present(prep.FLAN_TOKENIZE)
  flan_tokenize_lm_removed = remove_preprocessors_if_present(
      prep.FLAN_TOKENIZE_LM)

  if not (flan_tokenize_removed or flan_tokenize_lm_removed):
    raise ValueError(f'Error in {base_task_name}. We expect FLAN tasks to have'
                     ' FLAN_TOKENIZE as final in preprocessing steps.')

  # Don't expand each exemplar into multiple options. We will expand to multiple
  # options for the entire few-shot example.
  rank_classification_removed = remove_preprocessors_if_present(
      [prep.rank_classification_from_options])

  glm_rank_classification_removed = remove_preprocessors_if_present(
      [prep.GLM_RANK_CLASSIFICATION])

  # There should be a delimiter between the x and y of each example. Added here.
  @seqio.utils.map_over_dataset
  def add_delimiter_after_x(ex):
    new_ex = dict(ex)
    new_ex['inputs'] = tf.strings.join([ex['inputs'], x_y_delimiter])
    return new_ex

  single_ex_preprocessors.append(add_delimiter_after_x)

  # Form few-shot examples.
  few_shot_data_source = seqio.experimental.FewshotDataSource(
      original_source=task.source,
      num_shots=num_shots,
      train_preprocessors=single_ex_preprocessors,
      eval_preprocessors=single_ex_preprocessors,
      train_split='train',
      train_feature_keys=('inputs', 'targets', '_template_idx'),
                          # '_template_type', '_task_source', '_task_name'),
  )
  # These are the preprocessors we run *after* we have formed few-shot examples.
  # Note that we re-introduce the tokenization steps here.
  full_ex_preprocessors = []

  if prune_based_on_template_idx:
    full_ex_preprocessors.append(prep.prune_fewshot_examples_by_template_idx)

  if prune_exemplars:
    # Prunes excessive exemplars according to the max input length.
    if not max_input_length:
      raise ValueError(
          'To prune exemplars, `max_input_length` needs to be provided: %s.' %
          max_input_length)
    full_ex_preprocessors.extend([
        prep.get_fewshot_num_tokens,
        functools.partial(
            prep.prune_fewshot_examples_by_length,
            max_input_length=max_input_length)
    ])

  if prune_based_on_template_idx or prune_exemplars:
    full_ex_preprocessors.append(prep.filter_zero_shot_in_few_shot)

  full_ex_preprocessors.append(
      functools.partial(
          seqio.experimental.fewshot_preprocessor,
          inputs_prefix=inputs_prefix,
          targets_prefix=targets_prefix,
          example_separator=example_separator))

  full_ex_preprocessors.append(
      functools.partial(
          prep.remove_trailing_spaces_escape_newlines, features=['inputs']))

  if strip_targets:
    full_ex_preprocessors.append(
        functools.partial(prep.remove_trailing_spaces, features=['targets']))

  full_ex_preprocessors.append(
      functools.partial(
          prep.reformat_passthrough,
          format_strings={
              # equivalent to: 'inputs': f'{{inputs}}{final_suffix}', (default)
              'inputs': input_pattern.format(**{'final_suffix': final_suffix}),
              'targets': '{targets}',
          }))

  if rank_classification_removed:
    full_ex_preprocessors.append(prep.rank_classification_from_options)

  if glm_rank_classification_removed:
    full_ex_preprocessors.append(prep.GLM_RANK_CLASSIFICATION)

  if flan_tokenize_removed:
    full_ex_preprocessors.extend(prep.FLAN_TOKENIZE)

  if flan_tokenize_lm_removed:
    full_ex_preprocessors.extend(prep.FLAN_TOKENIZE_LM)

  # pylint: disable=protected-access
  if task.source._num_input_examples:
    few_shot_data_source._num_input_examples = {
        k: v - num_shots for k, v in task.source._num_input_examples.items()
    }
  # pylint: enable=protected-access
  seqio.TaskRegistry.add(
      name=new_task_name,
      source=few_shot_data_source,
      output_features=task.output_features,
      preprocessors=full_ex_preprocessors,
      postprocess_fn=task.postprocess_fn,
      metric_fns=task.metric_fns)
