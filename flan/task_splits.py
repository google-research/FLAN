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

"""Task clusters and splits for FLAN Meena."""

import collections
import random
from typing import List, Mapping, Optional, Sequence, Set

import numpy as np
import seqio

from flan import few_shot
from flan import tasks as flan_tasks  # pylint: disable=unused-import

ShotConfig = few_shot.ShotConfig

# Number of different intra-cluster splits to generate.
_NUM_INTRA_CLUSTER_SPLITS = 10

# Number of tasks in the training set for ablation study.
_NUM_TRAIN_TASKS_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

# Number of clusters in the training set for ablation study.
_NUM_TRAIN_CLUSTERS_LIST = range(1, 8)
_LIMIT_NUM_TRAIN_TASKS = 15
_TASKS_PER_CLUSTER_LIST = [1, 2, 3, 4]

# Define task clusters using abbreviated task names. These abbreviated names
# correspond to keys in templates.py:PATTERNS
_DEFAULT_TASK_CLUSTERS_ABBREV = collections.OrderedDict([
    ('summarization', [
        'aeslc',
        'cnn_dailymail',
        'gigaword',
        'multi_news',
        'newsroom',
        'samsum',
        'xsum',
        'ag_news_subset',
        'opinion_abstracts_rotten_tomatoes',
        'opinion_abstracts_idebate',
        'wiki_lingua_english_en',
    ]), ('structure_to_text', [
        'web_nlg_en',
        'common_gen',
        'e2e_nlg',
        'dart',
    ]),
    ('reading_comprehension', [
        'squad_v1',
        'squad_v2',
        'drop',
        'multirc',
        'openbookqa',
        'bool_q',
    ]),
    ('open_domain_qa', [
        'trivia_qa',
        'natural_questions',
        'arc_challenge',
        'arc_easy',
    ]), ('paraphrase', [
        'glue_mrpc',
        'glue_qqp',
        'paws_wiki',
        'stsb',
    ]),
    ('sentiment', [
        'imdb_reviews',
        'sentiment140',
        'yelp_polarity_reviews',
        'sst2',
    ]), ('text_formatting', [
        'true_case',
        'fix_punct',
        'word_segment',
    ]), ('common_sense', [
        'copa',
        'hellaswag',
        'story_cloze',
        'piqa',
    ]),
    ('translation', [
        'para_crawl_enes',
        'wmt14_enfr',
        'wmt16_translate_deen',
        'wmt16_translate_tren',
        'wmt16_translate_csen',
        'wmt16_translate_fien',
        'wmt16_translate_roen',
        'wmt16_translate_ruen',
    ]), ('coreference', [
        'definite_pronoun_resolution',
        'winogrande',
        'wsc',
    ]),
    ('entailment', [
        'anli_r1',
        'anli_r2',
        'anli_r3',
        'cb',
        'rte',
        'mnli_matched',
        'mnli_mismatched',
        'qnli',
        'wnli',
        'snli',
    ]), ('math', [
        'math_dataset',
    ]), ('conversational_qa', [
        'quac',
        'coqa',
    ]), ('word_disambiguation', [
        'wic',
    ]), ('linguistic_acceptability', [
        'cola',
    ]), ('question_classification', [
        'trec',
    ]), ('read_comp_and_common_sense', [
        'record',
        'cosmos_qa',
    ])
])

_SUPERGLUE_TASKS = frozenset(
    {'bool_q', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc'})


def full_task_name(abbrev_name: str, num_templates: int,
                   shot_config: ShotConfig) -> str:
  """Converts abbreviated task name into full TaskRegistry name."""
  return f'{abbrev_name}_{num_templates}templates{shot_config.name_suffix}'


def is_superglue_task(task_name: str) -> bool:
  """Checks whether a name produced by full_task_name is a SuperGLUE task."""
  for abbrev_name in _SUPERGLUE_TASKS:
    if task_name.startswith(f'{abbrev_name}_'):
      return True
  return False


# Each value in this dictionary is a list of tasks that can be collapsed into a
# single task (because they share the same format and instructions). Note that
# we use abbreviated task names (matching _DEFAULT_TASK_CLUSTERS_ABBREV).
_COLLAPSIBLE_TASKS_ABBREV = collections.OrderedDict([
    ('opinion_abstracts',
     ['opinion_abstracts_rotten_tomatoes', 'opinion_abstracts_idebate']),
    ('squad', ['squad_v1', 'squad_v2']),
    ('ai2_arc_with_ir', ['ai2_arc_with_ir_challenge', 'ai2_arc_with_ir_easy']),
    ('arc', ['arc_challenge', 'arc_easy']),
    ('anli', ['anli_r1', 'anli_r2', 'anli_r3']),
    ('mnli', ['mnli_matched', 'mnli_mismatched']),
])


def _get_default_task_collapse_map(
    num_templates: int, shot_config: ShotConfig) -> Mapping[str, str]:
  """Returns a map from a task to its collapsed task."""
  collapse_map = collections.OrderedDict()
  for collapsed_task, tasks_to_collapse in _COLLAPSIBLE_TASKS_ABBREV.items():
    for task in tasks_to_collapse:
      task_full_name = full_task_name(task, num_templates, shot_config)
      collapsed_task_full_name = full_task_name(collapsed_task, num_templates,
                                                shot_config)
      collapse_map[task_full_name] = collapsed_task_full_name
  return collapse_map


def collapse_related_tasks(tasks: Sequence[str],
                           collapse_map: Mapping[str, str]) -> List[str]:
  """Given a list of tasks, collapses related ones into single tasks."""
  collapsed_tasks = set()
  non_collapsed_tasks = []
  for task in tasks:
    if task in collapse_map:
      collapsed_tasks.add(collapse_map[task])
    else:
      non_collapsed_tasks.append(task)
  return non_collapsed_tasks + sorted(collapsed_tasks)


def expand_related_tasks(tasks: Set[str],
                         expand_map: Mapping[str, Sequence[str]]) -> Set[str]:
  """The inverse of `collapse_related_tasks`.

  Args:
    tasks: a list of tasks to expand.
    expand_map: map from a collapsed task name to a list of expanded task names.

  Returns:
    A list of expanded tasks.
  """
  expanded_tasks = set()
  for task in tasks:
    if task in expand_map:
      for full_task in expand_map[task]:
        expanded_tasks.add(full_task)
    else:
      expanded_tasks.add(task)
  return expanded_tasks


class TaskSplit:
  """Splits tasks into train and test tasks."""

  def __init__(self,
               name: str,
               train_tasks: Set[str],
               test_tasks: Set[str],
               handle_overlap: str = 'error'):
    """Creates a task split with no train-test overlap.

    Args:
      name: name of the split.
      train_tasks: a set of tasks to train on. Each string should be the name of
        a task in seqio.TaskRegistry.
      test_tasks: a set of tasks to evaluate on. Each string should be the name
        of a task in seqio.TaskRegistry.
      handle_overlap: what to do if overlap is detected between train and test
        tasks. Can be one of {'error', 'remove', 'allow'}. If 'error', will
        raise an error. If 'remove', will prune the overlapping tasks from the
        train tasks. If 'allow', will do nothing.
    """
    # Make copies, since we may modify them.
    train_tasks = set(train_tasks)
    test_tasks = set(test_tasks)

    # Check for train-test overlap.
    overlap = train_tasks & test_tasks
    if overlap:
      if handle_overlap == 'error':
        raise ValueError(f'Train-test overlap: {overlap}')
      elif handle_overlap == 'remove':
        for task in overlap:
          train_tasks.remove(task)
      elif handle_overlap == 'allow':
        pass
      else:
        raise ValueError(
            f'Unsupported value for `handle_overlap`: {handle_overlap}')

    # Make sure there are no empty splits.
    if not train_tasks:
      raise ValueError('train_tasks cannot be empty.')
    if not test_tasks:
      raise ValueError('test_tasks cannot be empty.')

    self.name = name
    self.train_tasks = sorted(train_tasks)
    self.test_tasks = sorted(test_tasks)

  def visualize_by_cluster(self, task_clusters: Mapping[str, Sequence[str]]):
    """Visualizes a split by clusters."""
    task_to_cluster = {}
    for cluster, tasks in task_clusters.items():
      for task in tasks:
        task_to_cluster[task] = cluster

    def group_by_cluster(tasks):
      grouped = collections.defaultdict(list)
      for task in tasks:
        grouped[task_to_cluster[task]].append(task)
      return grouped

    train_clusters = group_by_cluster(self.train_tasks)
    test_clusters = group_by_cluster(self.test_tasks)

    for cluster in set(train_clusters) | set(test_clusters):
      print(f'===== {cluster} =====')
      print(f'  TRAIN: {train_clusters[cluster]}')
      print(f'  TEST:  {test_clusters[cluster]}')

  def __repr__(self) -> str:
    return self.name

  @property
  def train_mixture_name(self) -> str:
    return self.name + '_train'

  @property
  def eval_mixture_name(self) -> str:
    return self.name + '_eval'

  def to_dict(self):
    return {
        'name': self.name,
        'train_tasks': self.train_tasks,
        'test_tasks': self.test_tasks,
    }

  def __eq__(self, o) -> bool:
    return (self.name == o.name and self.train_tasks == o.train_tasks and
            self.test_tasks == o.test_tasks)

  def __ne__(self, o) -> bool:
    return not self.__eq__(o)


def _get_default_task_clusters(
    num_templates: int,
    shot_config: ShotConfig,
    exclude_missing_tasks: bool = False,
) -> collections.OrderedDict:
  """Returns a dict of task clusters.

  Args:
    num_templates: number of templates per task.
    shot_config: specifies how many shots.
    exclude_missing_tasks: if True, any task that has not been included in
      templates.py:PATTERNS will be excluded from the returned clusters. If
        False, this function will throw an error for any requested task that has
        not been included in templates.py:PATTERNS.

  Returns:
    An OrderedDict, where each key is a cluster name and each value is a list of
    task names. Note that a task may appear in multiple clusters.
  """
  # Convert abbreviated task names to full names.
  task_clusters = collections.OrderedDict()

  for cluster_name, abbrev_task_names in _DEFAULT_TASK_CLUSTERS_ABBREV.items():
    full_names = []

    for abbrev_name in abbrev_task_names:
      full_name = full_task_name(abbrev_name, num_templates, shot_config)
      try:
        seqio.TaskRegistry.get(full_name)
      except ValueError:
        if exclude_missing_tasks:
          continue
        else:
          raise ValueError(f'No task defined for {full_name}.')
      full_names.append(full_name)

      if not full_names:
        continue  # If all tasks in a cluster are missing, exclude the cluster.
      task_clusters[cluster_name] = full_names

  return task_clusters


def generate_all_overlap_split(
    num_templates: int = 10,
    task_clusters: Optional[collections.OrderedDict] = None,
    shot_config: ShotConfig = ShotConfig.ZERO) -> TaskSplit:
  """Generates a task split where train and test tasks are the same."""
  if not task_clusters:
    task_clusters = _get_default_task_clusters(num_templates, shot_config)
  tasks = set()
  for cluster_tasks in task_clusters.values():
    tasks.update(cluster_tasks)
  return TaskSplit(
      name=f'flan_all_overlap_{num_templates}templates{shot_config.name_suffix}',
      train_tasks=tasks,
      test_tasks=tasks,
      handle_overlap='allow')


def generate_superglue_num_templates_ablation(
    shot_config: ShotConfig = ShotConfig.ZERO) -> List[TaskSplit]:
  """Task split with superglue held-out, ablation on number of templates."""

  task_splits = []
  for num_templates in flan_tasks.NUM_TEMPLATES_LIST:
    task_clusters = _get_default_task_clusters(num_templates, shot_config)
    all_tasks = []
    for cluster in task_clusters.values():
      all_tasks += cluster

    train_tasks = set()
    test_tasks = set()
    for task in all_tasks:
      if is_superglue_task(task):
        test_tasks.add(task)
      else:
        train_tasks.add(task)

    task_split = TaskSplit(
        name=f'flan_superglue_split_{num_templates}templates{shot_config.name_suffix}',
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        handle_overlap='error')
    task_splits.append(task_split)

  return task_splits


def generate_superglue_num_tasks_ablation(
    num_templates: int = 10,
    shot_config: ShotConfig = ShotConfig.ZERO) -> List[TaskSplit]:
  """A task split with superglue as test tasks and all others for training."""

  task_clusters = _get_default_task_clusters(num_templates, shot_config)

  task_splits = []

  for num_train_tasks in _NUM_TRAIN_TASKS_LIST:

    train_tasks = set()
    test_tasks = set()

    all_tasks = []
    for cluster in task_clusters.values():
      all_tasks += cluster
    # Fixed random seed reduces randomness in different num_train_tasks values.
    random.Random(42).shuffle(all_tasks)

    for task in all_tasks:
      if is_superglue_task(task):
        test_tasks.add(task)
      else:
        if len(train_tasks) < num_train_tasks:
          train_tasks.add(task)

    task_split = TaskSplit(
        name=f'flan_superglue_split{shot_config.name_suffix}'
        f'_{num_templates}templates'
        f'_{num_train_tasks}train_tasks',
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        handle_overlap='error')
    task_splits.append(task_split)

  return task_splits


def generate_inter_ablation(
    shot_config: ShotConfig = ShotConfig.ZERO) -> List[TaskSplit]:
  """Ablation study on number of task clusters for training."""

  task_splits = []

  # Up to 8 train clusters, ordered by number of tasks.
  train_clusters_ordered = [
      'summarization',
      'translation',
      'reading_comprehension',
      'sentiment',
      'structure_to_text',
      'coreference',
      'conversational_qa',
      'paraphrase',
  ]
  test_clusters_names = {'entailment', 'common_sense', 'open_domain_qa'}

  for num_templates in flan_tasks.NUM_TEMPLATES_LIST:

    all_task_clusters = _get_default_task_clusters(num_templates, shot_config)

    test_clusters = {
        k: v for k, v in all_task_clusters.items() if k in test_clusters_names
    }
    test_tasks = set()
    for tasks in test_clusters.values():
      test_tasks.update(tasks)

    for num_train_clusters in _NUM_TRAIN_CLUSTERS_LIST:

      # Get train tasks.
      train_clusters = {
          k: v
          for k, v in all_task_clusters.items()
          if k in train_clusters_ordered[:num_train_clusters]
      }
      largest_cluster_size = max([len(v) for v in train_clusters.values()])
      train_tasks = []
      for i in range(largest_cluster_size):
        for tasks in train_clusters.values():
          if len(tasks) > i:
            train_tasks.append(tasks[i])

      # Task split with num_train_clusters clusters.
      task_split = TaskSplit(
          name=f'flan_diversity_split{num_train_clusters}' +
          f'_{num_templates}templates' + shot_config.name_suffix,
          train_tasks=set(train_tasks),
          test_tasks=test_tasks,
          handle_overlap='error')

      # Task split with num_train_clusters clusters, constant number of tasks.
      limited_task_split = TaskSplit(
          name=f'flan_diversity_split{num_train_clusters}' +
          f'_{num_templates}templates' +
          f'_{_LIMIT_NUM_TRAIN_TASKS}task_limit' + shot_config.name_suffix,
          train_tasks=set(train_tasks[:_LIMIT_NUM_TRAIN_TASKS]),
          test_tasks=test_tasks,
          handle_overlap='error')

      for split in [task_split, limited_task_split]:
        task_splits.append(split)

    # Hold number of clusters constant, vary number of tasks per cluster.
    for tasks_per_cluster in _TASKS_PER_CLUSTER_LIST:

      # Get train tasks.
      train_clusters = {
          k: v
          for k, v in all_task_clusters.items()
          if k in train_clusters_ordered[:num_train_clusters]
      }
      train_tasks = set()
      for tasks in train_clusters.values():
        train_tasks.update(tasks[:tasks_per_cluster])

      # Add the task split.
      task_split = TaskSplit(
          name='flan_inter_split' + f'_{num_templates}templates' +
          f'_{tasks_per_cluster}tpc'  # tpc = tasks_per_cluster
          + shot_config.name_suffix,
          train_tasks=train_tasks,
          test_tasks=test_tasks,
          handle_overlap='error')
      task_splits.append(task_split)

  return task_splits


def generate_intra_cluster_splits(
    num_templates: int = 10,
    task_clusters: Optional[collections.OrderedDict] = None,
    task_collapse_map: Optional[Mapping[str, str]] = None,
    shot_config: ShotConfig = ShotConfig.ZERO) -> List[TaskSplit]:
  """Generates intra-cluster splits of tasks."""
  if not task_clusters:
    task_clusters = _get_default_task_clusters(num_templates, shot_config)
  if not task_collapse_map:
    task_collapse_map = _get_default_task_collapse_map(num_templates,
                                                       shot_config)

  # Compute the inverse of the task_collapse_map.
  task_expand_map = collections.defaultdict(list)
  for task, collapsed_task in task_collapse_map.items():
    task_expand_map[collapsed_task].append(task)

  if not isinstance(task_clusters, collections.OrderedDict):
    # The ordering of the keys is important for reproducibility.
    raise TypeError('task_clusters must be an OrderedDict.')

  # For the purpose of intra-cluster splitting, temporarily collapse related
  # tasks into single tasks, so that related tasks don't end up split across
  # train and test.
  task_clusters = {
      cluster_name: collapse_related_tasks(cluster_tasks, task_collapse_map)
      for cluster_name, cluster_tasks in task_clusters.items()
  }

  task_splits = []

  # For a given intra-cluster split (split_idx), and a given task cluster
  # (cluster_idx), choose 1 task (out of num_cluster_tasks total) to holdout.
  def select_holdout_task_idx(split_idx, cluster_idx, num_cluster_tasks):
    # Use a fixed random seed, so that this function is totally deterministic.
    rand = np.random.RandomState(seed=cluster_idx)
    # A random permutation of the integers [0, 1, 2, ..., num_cluster_tasks - 1]
    holdout_indices = rand.permutation(num_cluster_tasks)
    # Each split holds out a different task. When all tasks have been held out
    # once, loop back to the beginning of the random permutation.
    return holdout_indices[split_idx % num_cluster_tasks]

  for split_idx in range(_NUM_INTRA_CLUSTER_SPLITS):
    all_train_tasks = set()
    all_test_tasks = set()

    for cluster_idx, cluster_tasks in enumerate(task_clusters.values()):
      if len(cluster_tasks) <= 1:
        # If a cluster has <= 1 tasks, put them all in train.
        all_train_tasks.update(cluster_tasks)
      else:
        # Otherwise, hold out one task to be a test task.
        test_idx = select_holdout_task_idx(split_idx, cluster_idx,
                                           len(cluster_tasks))
        train_tasks = list(cluster_tasks)
        test_task = train_tasks.pop(test_idx)
        all_train_tasks.update(train_tasks)
        all_test_tasks.add(test_task)

    # After splitting is complete, re-expand any collapsed tasks.
    all_train_tasks = expand_related_tasks(all_train_tasks, task_expand_map)
    all_test_tasks = expand_related_tasks(all_test_tasks, task_expand_map)

    # Because some tasks are a member of multiple clusters, a task could be held
    # out from one cluster but not others, causing it to appear in both train
    # and test. We therefore pass handle_overlap='remove' to prune such tasks
    # from train, leaving them only in test.
    task_split = TaskSplit(
        name=f'flan_intra_cluster_split_{split_idx}' +
        f'_{num_templates}templates' + shot_config.name_suffix,
        train_tasks=all_train_tasks,
        test_tasks=all_test_tasks,
        handle_overlap='remove')
    task_splits.append(task_split)

  return task_splits


def generate_inter_cluster_splits(
    num_templates: int = 10,
    task_clusters: Optional[collections.OrderedDict] = None,
    shot_config: ShotConfig = ShotConfig.ZERO) -> List[TaskSplit]:
  """Generates inter-cluster splits of tasks."""
  if not task_clusters:
    task_clusters = _get_default_task_clusters(num_templates, shot_config)

  task_splits = []
  for test_cluster_idx, (test_cluster_name,
                         test_tasks) in enumerate(task_clusters.items()):
    train_tasks = set()
    for cluster_name, cluster_tasks in task_clusters.items():
      if test_cluster_name == cluster_name:
        continue

      # Special rule for reading comp and common sense tasks.
      if test_cluster_name in {'reading_comprehension', 'common_sense'}:
        if cluster_name == 'read_comp_and_common_sense':
          continue
      if test_cluster_name == 'read_comp_and_common_sense':
        if cluster_name in {'reading_comprehension', 'common_sense'}:
          continue

      if test_cluster_name == 'paraphrase' and cluster_name == 'entailment':
        continue
      if test_cluster_name == 'entailment' and cluster_name == 'paraphrase':
        continue

      train_tasks.update(cluster_tasks)

    # Because some tasks are a member of multiple clusters, a task could be in
    # both the test cluster and a train cluster. We therefore pass
    # handle_overlap='remove' to prune such tasks from train, leaving them only
    # in test.
    task_split = TaskSplit(
        name=f'flan_inter_cluster_split_{test_cluster_idx}' +
        f'_{num_templates}templates' + shot_config.name_suffix,
        train_tasks=train_tasks,
        test_tasks=set(test_tasks),
        handle_overlap='remove')
    task_splits.append(task_split)

  return task_splits
