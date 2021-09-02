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

"""Utility functions for FLAN."""
import abc
import re
from typing import Optional

import numpy as np

from flan import templates


def is_classification(flan_pattern_name: str):
  """Returns if the task is a classification task."""
  # ReCoRD task has variable length options, so it is not called options in
  # the input pattern. But it is classification.
  if flan_pattern_name == 'record':
    return True
  input_patterns = [p[0] for p in templates.PATTERNS[flan_pattern_name]]
  return np.any(['{options_}' in pattern for pattern in input_patterns])


class SeqioTaskName(metaclass=abc.ABCMeta):
  """Abstract class for seqio task name."""

  @abc.abstractclassmethod
  def get(cls, *args):
    """Returns task name."""
    raise NotImplementedError

  @abc.abstractclassmethod
  def parse(cls, task_name: str):
    """Returns task name."""
    raise NotImplementedError

  @abc.abstractclassmethod
  def match(cls, task_name: str) -> Optional[re.Match]:
    """Returns the match object if `task_name` matches the name pattern."""
    raise NotImplementedError


class ZeroshotEvalTaskName(SeqioTaskName):
  """Task name for zeroshot eval."""

  @classmethod
  def get(cls, t_name: str, template_id: int) -> str:
    return f'{t_name}_type_{template_id}'

  @classmethod
  def parse(cls, task_name):
    match = cls.match(task_name)
    return match[1], int(match[2])

  @classmethod
  def match(cls, task_name) -> Optional[re.Match]:
    return re.fullmatch(r'^(.+)_type_(\d+)$', task_name)


class ZeroshotScoreEvalTaskName(SeqioTaskName):
  """Task name for zeroshot scoring eval."""

  @classmethod
  def get(cls, t_name: str, template_id: int) -> str:
    return f'{t_name}_type_{template_id}_scoring_eval'

  @classmethod
  def parse(cls, task_name):
    match = cls.match(task_name)
    return match[1], int(match[2])

  @classmethod
  def match(cls, task_name) -> Optional[re.Match]:
    return re.fullmatch(r'^(.+)_type_(\d+)_scoring_eval$', task_name)


class ZeroshotScoreEvalNoOptionTaskName(SeqioTaskName):
  """Task name for zeroshot scoring eval without options."""

  @classmethod
  def get(cls, t_name: str, template_id: int) -> str:
    return f'{t_name}_type_{template_id}_score_eval_no_options'

  @classmethod
  def parse(cls, task_name):
    match = cls.match(task_name)
    return match[1], int(match[2])

  @classmethod
  def match(cls, task_name) -> Optional[re.Match]:
    return re.fullmatch(r'^(.+)_type_(\d+)_score_eval_no_options$', task_name)


class ZeroshotScoreFLANNoOptionTaskName(SeqioTaskName):
  """Task name for zeroshot scoring eval without options."""

  @classmethod
  def get(cls, t_name: str, template_id: int) -> str:
    return f'{t_name}_type_{template_id}_score_flan_no_options'

  @classmethod
  def parse(cls, task_name):
    match = cls.match(task_name)
    return match[1], int(match[2])

  @classmethod
  def match(cls, task_name) -> Optional[re.Match]:
    return re.fullmatch(r'^(.+)_type_(\d+)_score_flan_no_options$', task_name)


class AllPromptsTaskName(SeqioTaskName):
  """Task name for the training job realized from all prompts."""

  @classmethod
  def get(cls, t_name: str) -> str:
    return f'{t_name}_all_prompts'

  @classmethod
  def parse(cls, task_name):
    match = cls.match(task_name)
    return match[1]

  @classmethod
  def match(cls, task_name) -> Optional[re.Match]:
    return re.fullmatch(r'^(.+)_all_prompts', task_name)


class ZeroshotTemplatedTaskName(SeqioTaskName):
  """Zeroshot task name with number of realized templates."""

  @classmethod
  def get(cls, t_name: str, num_templates: int) -> str:
    return f'{t_name}_{num_templates}templates'

  @classmethod
  def parse(cls, task_name):
    match = cls.match(task_name)
    return match[1], int(match[2])

  @classmethod
  def match(cls, task_name) -> Optional[re.Match]:
    return re.fullmatch(r'^(.+)_(\d+)templates$', task_name)


class XshotTemplatedTaskName(SeqioTaskName):
  """Zeroshot task name with number of realized templates."""

  @classmethod
  def get(cls, t_name: str, num_templates: int, num_shot: str) -> str:
    return f'{t_name}_{num_templates}templates_{num_shot}_shot'

  @classmethod
  def parse(cls, task_name):
    match = cls.match(task_name)
    return match[1], int(match[2]), match[3]

  @classmethod
  def match(cls, task_name) -> Optional[re.Match]:
    return re.fullmatch(r'^(.+)_(\d+)templates_([a-z]+)_shot$', task_name)


def remove_input_patterns_options(input_pattern: str) -> str:
  """Remove options from the input pattern."""
  no_options_pattern = input_pattern.replace('{options_}', '')
  no_options_pattern = no_options_pattern.replace('{options_str}', '').strip()
  return no_options_pattern


def t_name_to_flan_pattern_name(t_name: str) -> str:
  """Converts `t_name` to flan `PATTERN` key.

  Some seqio tasks use the same flan patterns.
  Args:
    t_name: Task config name.

  Returns:
    a key for `PATTERNS`.
  """
  if 'para_crawl' in t_name:
    return 'para_crawl'
  elif 'wmt16_translate' in t_name:
    return 'wmt16_translate'
  elif t_name in {'arc_challenge', 'arc_easy'}:
    return 'arc'
  elif t_name in {'anli_r1', 'anli_r2', 'anli_r3'}:
    return 'anli'
  elif t_name in {'mnli_matched', 'mnli_mismatched'}:
    return 'mnli'
  return t_name


def get_eval_dir_basename(task: str, split: str) -> str:
  """Returns the basename for eval directory.

  Args:
    task: a seqio eval task name.
    split: split name.
  """
  return f'eval_{task}_{split}'
