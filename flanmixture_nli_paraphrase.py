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

"""Add Mixtures to the registry. I've created a new file"""

import functools
import seqio

from flan import few_shot
from flan import task_splits
from flan import tasks  # pylint: disable=unused-import
from flan import templates  # pylint: disable=unused-import

shot_config = task_splits.ShotConfig.ZERO
task_clusters =  task_splits._get_default_task_clusters(10, shot_config)

test_tasks = []
train_tasks = []

for cluster in task_clusters:
    if cluster in ['paraphrase', 'entailment']:
        for task in task_clusters[cluster]:
            test_tasks.append(task)
    else:
        for task in task_clusters[cluster]:
            train_tasks.append(task)

seqio.MixtureRegistry.add(
  name='flan_split_paraphrase_entailment_train',
  tasks=train_tasks,
  default_rate=mixing_rate_3k)
seqio.MixtureRegistry.add(
  name='flan_split_paraphrase_entailment_test',
  tasks=test_tasks,
  default_rate=seqio.mixing_rate_num_examples)
