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

"""Add Mixtures to the registry."""

import functools
import seqio

from flan import few_shot
from flan import task_splits
from flan import tasks  # pylint: disable=unused-import
from flan import templates  # pylint: disable=unused-import

mixing_rate_3k = functools.partial(seqio.mixing_rate_num_examples, maximum=3000)

all_splits = []

for shot_config in few_shot.ShotConfig:
  # Add inter cluster splits.
  all_splits += task_splits.generate_inter_cluster_splits(
      shot_config=shot_config)

  # Add intra cluster splits.
  all_splits += task_splits.generate_intra_cluster_splits(
      shot_config=shot_config)

  # Add all overlap split.
  all_splits += [
      task_splits.generate_all_overlap_split(shot_config=shot_config)
  ]

  # Add superglue ablation on number of templates.
  all_splits += task_splits.generate_superglue_num_templates_ablation(
      shot_config=shot_config)

  # Add superglue ablation on number of tasks.
  all_splits += task_splits.generate_superglue_num_tasks_ablation(
      shot_config=shot_config)

  # Add diversity ablation on number of clusters in finetuning.
  all_splits += task_splits.generate_inter_ablation(shot_config=shot_config)

for split in all_splits:
  seqio.MixtureRegistry.add(
      name=split.train_mixture_name,
      tasks=split.train_tasks,
      default_rate=mixing_rate_3k)
  seqio.MixtureRegistry.add(
      name=split.eval_mixture_name,
      tasks=split.test_tasks,
      default_rate=seqio.mixing_rate_num_examples)
