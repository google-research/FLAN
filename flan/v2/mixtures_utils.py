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

"""Utilities that help create mixtures."""

import collections
import functools
import typing

from flan.v2 import constants
from flan.v2 import constants_t0
from flan.v2 import task_configs

import seqio

TRAIN_TASK_SUFFIXES = constants.TRAIN_TASK_SUFFIXES
UNIVERSAL_MIX_PREFIX = 'palmflan'

# pylint: disable=protected-access
DEFAULT_MIXTURE_TASKS = {
    'FLAN': list(task_configs.FLAN_V0_TASK_CONFIGS.keys()),
    'T0': list(task_configs.T0_TASK_CONFIGS),
    'CoT': list(task_configs.COT_TASK_CONFIGS),
    'Dialog': list(task_configs.DIALOG_TASK_CONFIGS),
    'NIv2': list(task_configs.NIV2_TASK_CONFIGS),
    'CoT-II': list(task_configs.COT_II_TASK_CONFIGS),
    'Dialog-II': list(task_configs.DIALOG_II_TASK_CONFIGS),
}
# pylint: enable=protected-access

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
        None,
    'Dialog':
        lambda x: x not in ['wiki_dialog', 'qrecc'],
    'Dialog-II':
        lambda x: x not in
        ['wiki_dialog_input_inversion', 'qrecc_input_inversion'],
}
DEFAULT_MIXTURE_TASK_FILTERS['CoT-II'] = DEFAULT_MIXTURE_TASK_FILTERS['CoT']
# pylint: enable=g-long-lambda

ZS_OPT_TEMPLATES = [
    '{t_name}_template_0to10_non_deter_opt_zero_shot{suffix}',
    '{t_name}_template_0to10_zero_shot{suffix}',
]
ZS_NOOPT_TEMPLATES = [
    '{t_name}_template_0to10_no_opt_zero_shot{suffix}',
]
FS_OPT_TEMPLATES = [
    '{t_name}_template_mix_five_shot{suffix}',
    '{t_name}_template_0to10_x_shot{suffix}',
]
FS_NOOPT_TEMPLATES = [
    '{t_name}_template_0to10_no_opt_x_shot{suffix}',
]


def register_mixture(mix_prefix: str,
                     mix_ex_cap: int,
                     task_suffix: str,
                     tasks_obj: typing.Union[typing.Dict[str, typing.Any],
                                             typing.List[str]],
                     templates: typing.List[str],
                     filter_fn: typing.Optional[typing.Callable[[str], bool]],
                     use_all_templates: bool = True) -> str:
  """Selects appropriate task-template IDs and creates a mixture.

  Args:
    mix_prefix: Name prefix of the the new mixture.
    mix_ex_cap: Maximum number of examples per dataset in this mixture.
    task_suffix: Name suffix of the mixture, based on the model type.
    tasks_obj: A list or dictionary where the elements or keys (respectively)
      are the names of relevant tasks.
    templates: A list of templates to apply to the tasks.
    filter_fn: A function that takes in task names enumerated by `tasks_obj` and
      returns True if the task should NOT be added to this mixture.
    use_all_templates: If True, use all templates in the `templates` list,
      otherwise, take the first one for each `task_obj` task name that appears
      in `ALL_REGISTERED_TASK_NAMES`.

  Returns:
    The registered seqio mixture name.
  """

  final_mix_name = mix_prefix + task_suffix
  if check_mix_exists(final_mix_name):
    return final_mix_name

  task_template_ids = []
  task_names = tasks_obj.keys() if isinstance(tasks_obj, dict) else tasks_obj
  for task_name in task_names:
    if filter_fn is not None and filter_fn(task_name):
      continue

    # pylint: disable=g-complex-comprehension
    formatted_templates = [
        template.format(**{
            't_name': task_name,
            'suffix': task_suffix
        }) for template in templates
    ]
    # pylint: enable=g-complex-comprehension

    if use_all_templates:
      task_template_ids.extend(formatted_templates)
    else:
      if task_name in task_configs.NON_DETER_TASK_NAMES:
        task_template_ids.append(formatted_templates[0])
      else:
        task_template_ids.append(formatted_templates[1])

  mix = seqio.MixtureRegistry.add(
      name=final_mix_name,
      tasks=task_template_ids,
      default_rate=functools.partial(
          seqio.mixing_rate_num_examples, maximum=mix_ex_cap),
  )
  return mix.name


def register_mixture_of_mixtures(
    new_mix_name: str,
    submix_rates: typing.List[typing.Tuple[str, float]],
) -> str:
  """Registers a mixture of mixtures, at the given rates.

  Args:
    new_mix_name: Mixture name.
    submix_rates: List of submixes to include and their respective rates.

  Returns:
    The registered seqio mixture name.
  """
  if check_mix_exists(new_mix_name):
    return new_mix_name
  mix = seqio.MixtureRegistry.add(
      name=new_mix_name,
      tasks=submix_rates,
  )
  return mix.name


def check_mix_exists(mix_name: str):
  # NB: It's important to check each time, without saving variable.
  return mix_name in seqio.MixtureRegistry.names()


def register_submixture_variants(
    submix_key: str,
    submix_ex_caps: typing.Dict[str, int],
    ratio_zero_shot: float,
    ratio_answer_opts: float,
    tsuffix: str,
    mixture_name: str,
) -> str:
  """Creates specified permutations of mixtures for [ZS, FS] x [Opt, No-Opt]."""

  def generate_submix_prefix(submix_key, zero_shot: bool, opt: bool):
    zs_token = 'zs' if zero_shot else 'fs'
    opt_token = 'opt' if opt else 'noopt'
    return f'{UNIVERSAL_MIX_PREFIX}_{submix_key.lower()}_{zs_token}_{opt_token}'

  submix_task_names = DEFAULT_MIXTURE_TASKS[submix_key]

  # ZS Opt
  submix_zs_opt_prefix = generate_submix_prefix(
      submix_key, zero_shot=True, opt=True)
  zs_opt_mix_name = register_mixture(
      mix_prefix=submix_zs_opt_prefix,
      mix_ex_cap=submix_ex_caps[submix_key],
      task_suffix=tsuffix,
      tasks_obj=submix_task_names,
      templates=ZS_OPT_TEMPLATES,
      filter_fn=DEFAULT_MIXTURE_TASK_FILTERS[submix_key],
      use_all_templates=False,
  )

  # ZS NoOpt
  submix_zs_noopt_prefix = generate_submix_prefix(
      submix_key, zero_shot=True, opt=False)
  zs_noopt_mix_name = register_mixture(
      mix_prefix=submix_zs_noopt_prefix,
      mix_ex_cap=submix_ex_caps[submix_key],
      task_suffix=tsuffix,
      tasks_obj=submix_task_names,
      templates=ZS_NOOPT_TEMPLATES,
      filter_fn=DEFAULT_MIXTURE_TASK_FILTERS[submix_key],
  )

  # FS Opt
  submix_fs_opt_prefix = generate_submix_prefix(
      submix_key, zero_shot=False, opt=True)
  fs_opt_mix_name = register_mixture(
      mix_prefix=submix_fs_opt_prefix,
      mix_ex_cap=submix_ex_caps[submix_key],
      task_suffix=tsuffix,
      tasks_obj=submix_task_names,
      templates=FS_OPT_TEMPLATES,
      filter_fn=DEFAULT_MIXTURE_TASK_FILTERS[submix_key],
  )

  # FS NoOpt
  submix_fs_noopt_prefix = generate_submix_prefix(
      submix_key, zero_shot=False, opt=False)
  fs_noopt_mix_name = register_mixture(
      mix_prefix=submix_fs_noopt_prefix,
      mix_ex_cap=submix_ex_caps[submix_key],
      task_suffix=tsuffix,
      tasks_obj=submix_task_names,
      templates=FS_NOOPT_TEMPLATES,
      filter_fn=DEFAULT_MIXTURE_TASK_FILTERS[submix_key],
  )

  # Merge ZS Opt, FS Opt, ZS NoOpt, FS NoOpt for this submix_key into 1 mix
  combination_scores = [
      ratio_zero_shot * ratio_answer_opts,  # ZS Opt
      (1 - ratio_zero_shot) * ratio_answer_opts,  # FS Opt
      ratio_zero_shot * (1 - ratio_answer_opts),  # ZS NoOpt
      (1 - ratio_zero_shot) * (1 - ratio_answer_opts)  # FS NoOpt
  ]

  all_submix_keys = [
      zs_opt_mix_name,
      fs_opt_mix_name,
      zs_noopt_mix_name,
      fs_noopt_mix_name,
  ]
  selected_submixes = [
      (k, s) for k, s in zip(all_submix_keys, combination_scores) if s > 0.0
  ]

  return register_mixture_of_mixtures(
      new_mix_name=mixture_name,
      submix_rates=selected_submixes,
  )


# pylint: disable=dangerous-default-value
def generate_mixture_suites(
    submixtures: typing.List[str],
    submix_ex_caps: typing.Dict[str, int],
    submix_rates: typing.Dict[str, float],
    ratio_zero_shot: float,
    ratio_answer_opts: float,
    override_mix_name: typing.Optional[str] = None,
    task_suffixes: typing.List[str] = TRAIN_TASK_SUFFIXES,
):
  """Creates a top-level Palm+Flan mixture from all specifications.

  This function creates a high-level mixture from submixtures.
  For instance, if all submixtures are included and ratio_zero_shot=0.5 and
  ratio_answer_opts=0.5 then we want to create and merge several combinations
  of submixtures, like this:

  For each submix in submixtures:
    submix = Merge([ZS-Opt, ZS-NoOpt, FS-Opt, FS-NoOpt])
    where each of these are example capped by submix_ex_caps[submix]

  Next we merge these submixtures together at `submix_rates`:
    Merge([(submix1, rate1), (submix2, rate2), (submix3, rate3), ...])

  Args:
    submixtures: List of submixtures names. Subset of `DEFAULT_MIXTURE_TASKS`:
      'FLAN', 'T0', 'CoT', 'Dialog', 'NIv2'
    submix_ex_caps: Dict submixture name to dataset example cap.
    submix_rates: Dict submixture name to top-level rate for that submixture
    ratio_zero_shot: ratio of zero shot to few shot. 1.0 is all zero shot. 0.0
      is all few shot.
    ratio_answer_opts: ratio of answer options to no answer options. 1.0 is all
      answer options. 0.0 is no answer options.
    override_mix_name: Optional override name for the final top-level mixture.
      Appends the `suffix` for each included.
    task_suffixes: List of task suffix mixtures you'd like to include
  """
  # Check inputs are valid.
  # pylint: disable=superfluous-parens
  if not all(
      [submix_key in DEFAULT_MIXTURE_TASKS for submix_key in submixtures]):
    raise ValueError(
        f'Submix_keys do not correspond to {DEFAULT_MIXTURE_TASKS.keys()}.')
  if not all([submix_key in submix_ex_caps for submix_key in submixtures]):
    raise ValueError(f'submix_ex_caps do not correspond to {submixtures}.')
  if not all([submix_key in submix_rates for submix_key in submixtures]):
    raise ValueError(f'submix_rates do not correspond to {submixtures}.')
  if not (0.0 <= ratio_zero_shot <= 1.0):
    raise ValueError('ratio_zero_shot needs to be between 0 and 1, inclusive.')
  if not (0.0 <= ratio_answer_opts <= 1.0):
    raise ValueError(
        'ratio_answer_opts needs to be between 0 and 1, inclusive.')

  # Create all submixtures
  for tsuffix in task_suffixes:
    submix_key_to_submix_name = {}
    for submix_key in submixtures:

      automatic_submix_name = (
          f'{UNIVERSAL_MIX_PREFIX}_{submix_key.lower()}' +
          f'_zs({ratio_zero_shot})_opts({ratio_answer_opts})' + tsuffix)
      submixture_variant_name = register_submixture_variants(
          submix_key, submix_ex_caps, ratio_zero_shot, ratio_answer_opts,
          tsuffix, automatic_submix_name)
      submix_key_to_submix_name[submix_key] = submixture_variant_name

    ordered_task_keys = sorted([k for k in submix_key_to_submix_name.keys()])
    task_str = '_'.join([
        f'{k.lower()}({submix_rates[k]},{submix_ex_caps[k]})'
        for k in ordered_task_keys
    ])
    automatic_mix_name = (f'{UNIVERSAL_MIX_PREFIX}_{task_str}' +
                          f'_zs({ratio_zero_shot})_opts({ratio_answer_opts})' +
                          tsuffix)
    final_mix_name = f'{override_mix_name}{tsuffix}' if override_mix_name else automatic_mix_name
    # Combine submixes into one final mixture.
    register_mixture_of_mixtures(
        new_mix_name=final_mix_name,
        submix_rates=[
            (v, submix_rates[k]) for k, v in submix_key_to_submix_name.items()
        ],
    )


# Utilities used to "slice" NIv2 mixture
def get_niv2_task_id_to_task_names(niv2: seqio.Mixture):
  task_names = [t.name for t in niv2.tasks]
  task_id_to_task_names = collections.defaultdict(list)
  for task_name in task_names:
    task_id = int(task_name.split('_')[0][4:])
    task_id_to_task_names[task_id].append(task_name)
  return task_id_to_task_names


def get_niv2_subsampled_task_names(task_id_to_task_names, task_slice):
  subsampled_task_names = []
  sorted_task_ids = sorted(list(task_id_to_task_names.keys()))
  for task_id in sorted_task_ids[task_slice]:
    subsampled_task_names += task_id_to_task_names[task_id]
  return subsampled_task_names
