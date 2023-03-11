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

"""Sub-mixtures and the final mixture."""

from flan.v2 import constants
from flan.v2 import constants_t0
from flan.v2 import mixtures_utils
from flan.v2 import tasks  # pylint: disable=unused-import

import seqio

seqio.add_global_cache_dirs(constants.CACHE_DIRS)


DEFAULT_MIXTURE_MAX_EXAMPLES = {
    'FLAN': 30000,
    'T0': 20000,
    'CoT': 100000,
    'NIv2': 5000,
    'Dialog': 200000,
}
DEFAULT_MIXTURE_MAX_EXAMPLES['CoT-II'] = DEFAULT_MIXTURE_MAX_EXAMPLES['CoT']
DEFAULT_MIXTURE_MAX_EXAMPLES['Dialog-II'] = DEFAULT_MIXTURE_MAX_EXAMPLES[
    'Dialog']

DEFAULT_MIXTURE_RATES = {
    'FLAN': 4,
    'T0': 3,
    'CoT': 1,
    'NIv2': 1,
    'Dialog': 1,
    'CoT-II': 0.3,
    'Dialog-II': 0.3,
}
# pylint: disable=g-long-lambda
DEFAULT_MIXTURE_TASK_FILTERS = {
    'FLAN':
        None,
    'T0':
        lambda x: (constants_t0.T0_TASK_PREFIX not in x) or ('_score_eval' in x)
        or constants_t0.T0_TRAIN_TASK_METADATA[x]['in_flan'],
    'CoT':
        None,
    'NIv2':
        lambda x: False,  # Always passes
    'Dialog':
        lambda x: x not in ['wiki_dialog', 'task_master', 'qrecc'],
    'Dialog-II':
        lambda x: x not in [
            'wiki_dialog_input_inversion', 'task_master_input_inversion',
            'qrecc_input_inversion'
        ],
}
DEFAULT_MIXTURE_TASK_FILTERS['CoT-II'] = DEFAULT_MIXTURE_TASK_FILTERS['CoT']
# pylint: enable=g-long-lambda

# ----------------------- Sub-Mixture Definitions ----------------------- #
for task in ['FLAN', 'T0', 'CoT', 'Dialog', 'NIv2']:
  for (setting, zs_ratio, opt_ratio) in [('ZSOpt', 1, 1), ('ZSNoOpt', 1, 0),
                                         ('FSOpt', 0, 1), ('FSNoOpt', 0, 0)]:

    if task in ['CoT', 'Dialog']:
      tasks = [task, f'{task}-II']
      rates = {task: 1, f'{task}-II': 0.3}
    else:
      tasks = [task]
      rates = {task: 1}

    # CoT, NIv2, and Dialog do not have NoOpt versions.
    if task in ["CoT", "Dialog", "NIv2"] and "NoOpt" in setting:
        continue

    mixture_name = f'{task.lower()}_{setting.lower()}'
    mixtures_utils.generate_mixture_suites(
        submixtures=tasks,
        submix_ex_caps=DEFAULT_MIXTURE_MAX_EXAMPLES,
        submix_rates=rates,
        ratio_zero_shot=zs_ratio,
        ratio_answer_opts=opt_ratio,
        override_mix_name=mixture_name,
        task_suffixes=constants.TRAIN_TASK_SUFFIXES,
    )
