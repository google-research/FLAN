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

"""Utilities for creating few-shot learning tasks."""
import enum
import functools
from typing import Optional

import seqio
import tensorflow as tf

from flan import preprocessors as flan_prep


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


def register_few_shot_version_of_task(base_task_name: str,
                                      new_task_name: str,
                                      num_shots: int,
                                      x_y_delimiter: str = ' X ',
                                      inputs_prefix: str = '0 ',
                                      targets_prefix: str = '1 ',
                                      example_separator: str = ' X ',
                                      final_suffix: str = ' FLAN',
                                      prune_exemplars: bool = False,
                                      max_input_length: Optional[int] = None):
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
  if not remove_preprocessors_if_present(flan_prep.FLAN_TOKENIZE):
    raise ValueError('We expect FLAN tasks to have FLAN_TOKENIZE as final '
                     'preprocessing steps.')

  # Don't expand each exemplar into multiple options. We will expand to multiple
  # options for the entire few-shot example.
  rank_classification_removed = remove_preprocessors_if_present(
      [flan_prep.rank_classification_from_options])

  glm_rank_classification_removed = remove_preprocessors_if_present(
      [flan_prep.GLM_RANK_CLASSIFICATION])

  # Don't put the dialog prompt around each exemplar. We will put it around the
  # entire few-shot example.
  flan_dialog_prompt_removed = remove_preprocessors_if_present(
      [flan_prep.reformat_with_flan_dialog_prompt])

  dialog_prompt_removed = remove_preprocessors_if_present(
      [flan_prep.reformat_with_dialog_prompt])

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
      train_feature_keys=('inputs', 'targets'),
  )
  # These are the preprocessors we run *after* we have formed few-shot examples.
  # Note that we re-introduce the tokenization steps here.
  full_ex_preprocessors = []

  if prune_exemplars:
    # Prunes excessive exemplars according to the max input length.
    if not max_input_length:
      raise ValueError(
          'To prune exemplars, `max_input_length` needs to be provided: %s.' %
          max_input_length)
    full_ex_preprocessors.extend([
        flan_prep.get_fewshot_num_tokens,
        functools.partial(
            flan_prep.prune_fewshot_examples_by_length,
            max_input_length=max_input_length)
    ])

  full_ex_preprocessors.append(
      functools.partial(
          seqio.experimental.fewshot_preprocessor,
          inputs_prefix=inputs_prefix,
          targets_prefix=targets_prefix,
          example_separator=example_separator))

  full_ex_preprocessors.append(
      functools.partial(flan_prep.remove_trailing_spaces, features=['inputs']))

  if dialog_prompt_removed:
    # Do nothing. fewshot_preprocessor already added the needed dialog markers.
    pass

  if flan_dialog_prompt_removed:
    full_ex_preprocessors.append(
        functools.partial(
            flan_prep.reformat_passthrough,
            format_strings={
                'inputs': f'{{inputs}}{final_suffix}',
                'targets': '{targets}',
            }))

  if rank_classification_removed:
    full_ex_preprocessors.append(flan_prep.rank_classification_from_options)

  if glm_rank_classification_removed:
    full_ex_preprocessors.append(flan_prep.GLM_RANK_CLASSIFICATION)

  full_ex_preprocessors.extend(flan_prep.FLAN_TOKENIZE)
  seqio.TaskRegistry.add(
      name=new_task_name,
      source=few_shot_data_source,
      output_features=task.output_features,
      preprocessors=full_ex_preprocessors,
      postprocess_fn=task.postprocess_fn,
      metric_fns=task.metric_fns)
