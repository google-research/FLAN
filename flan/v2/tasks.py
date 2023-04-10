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

"""Register all tasks in task_configs.py."""
import copy
import random
import functools
import json
import os
import tensorflow as tf
from typing import List, Tuple

from flan.v2 import constants
from flan.v2 import few_shot
from flan.v2 import preprocessors as prep
from flan.v2 import task_configs
from flan.v2 import templates
from flan.v2 import utils
import seqio

# All tasks will be defined for each set of features.
ShotConfig = few_shot.ShotConfig


# Load all Natural Instruction V2 exemplars into memory.
_niv2_few_shot_exemplars = []
for _part_idx in range(10):
  _niv2_few_shot_exemplar_file = os.path.join(
      os.path.dirname(__file__), 'niv2_few_shot_data',
      'niv2_exemplars.jsonl-{:05d}-of-00010'.format(_part_idx))
  _niv2_few_shot_exemplars.extend([
      json.loads(x) for x in open(_niv2_few_shot_exemplar_file, 'r').readlines()])

def _flatten(list_of_str):
  return '\nflanv2-separator\n'.join([str(x) for x in list_of_str])
_niv2_task_names = [x['task'].split('.')[0] for x in _niv2_few_shot_exemplars]
_niv2_exemplar_inputs = [_flatten([y['input'] for y in x['sample']]) for x in _niv2_few_shot_exemplars]
_niv2_exemplar_targets = [_flatten([y['output'] for y in x['sample']]) for x in _niv2_few_shot_exemplars]
_niv2_exemplar_inputs_lookup = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(_niv2_task_names), tf.constant(_niv2_exemplar_inputs)),
    default_value="")
_niv2_exemplar_targets_lookup = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(_niv2_task_names), tf.constant(_niv2_exemplar_targets)),
    default_value="")


# Add zero-shot tasks in task_configs.TASK_CONFIGS.
def register_zero_shot_task(zero_shot_name: str,
                            zero_shot_config: task_configs.TaskConfig,
                            patterns: List[Tuple[str, str]],
                            template_type: str=None):
  if len(patterns) == 1:
    formatter = prep.get_formatter(patterns[0][0], patterns[0][1])
  else:
    # This batch formatter applies many prompts to a single task.
    formatter = prep.get_batch_formatter(patterns)

  add_template_metadata_fn = functools.partial(prep.add_template_info, template_type=template_type)
  for suffix, output_features in constants.TRAIN_TASK_SUFFIXES_AND_FEATURES:
    seqio.TaskRegistry.add(
        zero_shot_name + suffix,
        source=zero_shot_config.source,
        preprocessors=zero_shot_config.preprocessors + [add_template_metadata_fn] + formatter +
        prep.FLAN_TOKENIZE,
        postprocess_fn=zero_shot_config.postprocess_fn,
        output_features=output_features,
        metric_fns=zero_shot_config.metric_fns)


# Add zero-shot tasks in task_configs.TASK_CONFIGS.
def register_niv2_few_shot_task(
    few_shot_name: str,
    zero_shot_config: task_configs.TaskConfig,
    patterns: List[Tuple[str, str, str]],
    template_type: str=None):
  add_exemplar_features_fn = functools.partial(
      prep.niv2_few_shot_exemplar_lookup_fn,
      exemplar_input_lookup=_niv2_exemplar_inputs_lookup,
      exemplar_targets_lookup=_niv2_exemplar_targets_lookup)
  add_template_metadata_fn = functools.partial(prep.add_template_info, template_type=template_type)
  for suffix, output_features in constants.TRAIN_TASK_SUFFIXES_AND_FEATURES:
    seqio.TaskRegistry.add(
        few_shot_name + suffix,
        source=zero_shot_config.source,
        preprocessors=zero_shot_config.preprocessors +
        [add_template_metadata_fn, add_exemplar_features_fn] +
        prep.get_batch_formatter(patterns) + prep.FLAN_TOKENIZE,
        postprocess_fn=zero_shot_config.postprocess_fn,
        output_features=output_features,
        metric_fns=zero_shot_config.metric_fns)


for t_name, config in task_configs.ALL_CANDIDATE_TASK_CONFIGS.items():
  flan_pattern_name = utils.t_name_to_flan_pattern_name(t_name)
  patterns_list = templates.PATTERNS[flan_pattern_name]

  selected_patterns = patterns_list[0:1]
  zero_shot_task_name = f"{t_name}_template_0_zero_shot"
  register_zero_shot_task(zero_shot_task_name, config, selected_patterns, "zs_opt")

  selected_patterns = patterns_list
  zero_shot_task_name = f"{t_name}_template_0to10_zero_shot"
  register_zero_shot_task(zero_shot_task_name, config, selected_patterns, "zs_opt")

  # Add tasks that have no answer options provided in their templates at all.
  no_opt_patterns_list = templates.PATTERNS_NO_OPTIONS[flan_pattern_name]
  no_opt_selected_patterns = no_opt_patterns_list
  zero_shot_task_name = f"{t_name}_template_0to10_no_opt_zero_shot"
  register_zero_shot_task(zero_shot_task_name, config, no_opt_selected_patterns, "zs_noopt")

  # Adding tasks with non-deterministic option strings.
  # The option string in each input sequence will have a random format.
  # For example:
  #   Choices: (1). Eat apple; (2). Eat banana; (3). Eat cashew;
  #   Options: A). Eat apple B). Eat banana C). Eat cashew
  # These t_names include:
  #   bool_q, rte, wsc, wsc273, wic, arc_challenge, arc_easy, multirc
  #   ag_news_subset, anli_r1, anli_r2, anli_r3, sentiment140, story_cloze
  #   imdb_reviews, paws_wiki, definite_pronoun_resolution, glue_mrpc, glue_qqp
  #   copa, winogrande, yelp_polarity_reviews, cosmos_qa, cb, cola, sst2
  #   mnli_matched, mnli_mismatched, qnli, wnli, snli, trec, stsb, piqa
  #   openbookqa, hellaswag, unified_qa_science_inst, T0 subtasks, ...
  if prep.format_options in config.preprocessors:
    config_non_deter_opt = copy.deepcopy(config)
    utils.inplace_modify_preprocessors(
        config_non_deter_opt.preprocessors,
        {prep.format_options: prep.format_options_non_deterministic})
    selected_patterns = patterns_list
    zero_shot_task_name = f"{t_name}_template_0to10_non_deter_opt_zero_shot"
    register_zero_shot_task(zero_shot_task_name, config_non_deter_opt,
                            selected_patterns, "zs_opt")

  # Adding tasks with non-deterministic dialog strings.
  # The dialog string in each input sequence will have a random format.
  # For example:
  #   2 Person dialog: (1). Hi; (2). How are you?; (1). Great ty;
  # These t_names include:
  #   wiki_dialog, task_master, qrecc
  if prep.format_dialog in config.preprocessors:
    config_non_deter_dialog = copy.deepcopy(config)
    utils.inplace_modify_preprocessors(
        config_non_deter_dialog.preprocessors,
        {prep.format_dialog: prep.format_dialog_non_deterministic})
    selected_patterns = patterns_list
    zero_shot_task_name = f"{t_name}_template_0to10_non_deter_opt_zero_shot"
    register_zero_shot_task(zero_shot_task_name, config_non_deter_dialog,
                            selected_patterns, "zs_opt")


# ========================== #
# == Few-shot tasks below == #

NON_NIV2_TASK_CONFIGS = {
    k: v
    for k, v in task_configs.ALL_CANDIDATE_TASK_CONFIGS.items()
    if k not in task_configs.NIV2_TASK_CONFIGS
}
for t_name, config in NON_NIV2_TASK_CONFIGS.items():
  flan_pattern_name = utils.t_name_to_flan_pattern_name(t_name)

  # Few-shot with and without options strings.
  for pattern_dict, pattern_name in [(templates.FEWSHOT_PATTERNS, ""),
                                     (templates.FEWSHOT_PATTERNS_NO_OPTIONS,
                                      "_no_opt")]:
    if flan_pattern_name not in pattern_dict:
      continue

    template_type = "fs_opt" if pattern_name == "" else "fs_noopt"
    add_template_metadata_fn = functools.partial(prep.add_template_info, template_type=template_type)

    # Usually, template_0 is the simplest template (without a lot of
    # instructions. So we do more shots with template_0.
    for pattern_i, num_shots in [(0, ShotConfig.FIVE), (1, ShotConfig.TWO),
                                 (2, ShotConfig.ONE)]:
      fewshot_base_task_name = f"{t_name}_template_{pattern_i}" + pattern_name
      fewshot_pattern = pattern_dict[flan_pattern_name][pattern_i]
      input_pattern = fewshot_pattern.inputs
      target_pattern = fewshot_pattern.targets
      few_shot_kwargs = fewshot_pattern.few_shot_kwargs
      for task_suffix, task_output_features in constants.TRAIN_TASK_SUFFIXES_AND_FEATURES:
        seqio.TaskRegistry.add(
            fewshot_base_task_name + task_suffix,
            source=config.source,
            preprocessors=config.preprocessors + [add_template_metadata_fn] +
            prep.get_formatter(input_pattern, target_pattern) +
            prep.FLAN_TOKENIZE,
            postprocess_fn=config.postprocess_fn,
            output_features=task_output_features,
            metric_fns=config.metric_fns)
        # Task names:
        # f'{t_name}_template_0_five_shot{suffix}'
        # f'{t_name}_template_1_two_shot{suffix}'
        # f'{t_name}_template_2_one_shot{suffix}'
        # f'{t_name}_template_0_no_opt_five_shot{suffix}'
        # f'{t_name}_template_1_no_opt_two_shot{suffix}'
        # f'{t_name}_template_2_no_opt_one_shot{suffix}'
        few_shot.register_few_shot_version_of_task(
            base_task_name=fewshot_base_task_name + task_suffix,
            new_task_name=(fewshot_base_task_name + num_shots.name_suffix +
                           task_suffix),
            num_shots=num_shots.value,
            prune_exemplars=True,
            max_input_length=constants.FEW_SHOT_MAX_LEN,
            **few_shot_kwargs)

    # Build few-shot with mixed templates.
    num_shots = ShotConfig.FIVE
    fewshot_base_task_name = f"{t_name}_template_mix" + pattern_name

    mix_patterns = []
    for few_shot_pattern in pattern_dict[flan_pattern_name]:
      if few_shot_pattern.in_template_mix:
        mix_patterns.append((few_shot_pattern.combined_inputs_w_target_prefix,
                             few_shot_pattern.targets))

    if len(mix_patterns) == 1:
      mix_formatter = prep.get_formatter(mix_patterns[0][0], mix_patterns[0][1])
    elif len(mix_patterns) > 1:
      mix_formatter = prep.get_batch_formatter(mix_patterns)
    else:
      continue

    for task_suffix, task_output_features in constants.TRAIN_TASK_SUFFIXES_AND_FEATURES:
      seqio.TaskRegistry.add(
          fewshot_base_task_name + task_suffix,
          source=config.source,
          preprocessors=config.preprocessors + [add_template_metadata_fn] + mix_formatter +
          prep.FLAN_TOKENIZE,
          postprocess_fn=config.postprocess_fn,
          output_features=task_output_features,
          metric_fns=config.metric_fns)
      # Task names:
      # f'{t_name}_template_mix_five_shot{suffix}'
      # f'{t_name}_template_mix_no_opt_five_shot{suffix}'
      few_shot.register_few_shot_version_of_task(
          base_task_name=fewshot_base_task_name + task_suffix,
          new_task_name=(fewshot_base_task_name + num_shots.name_suffix +
                         task_suffix),
          num_shots=num_shots.value,
          prune_exemplars=True,
          max_input_length=constants.FEW_SHOT_MAX_LEN,
          x_y_delimiter="",
          inputs_prefix="",
          targets_prefix="",
          example_separator=random.choice(["\n", "\n\n", "\n\n\n"]),
          # We always use the 2nd template's final_suffix and input_pattern.
          final_suffix=pattern_dict[flan_pattern_name][1].final_suffix,
          input_pattern=pattern_dict[flan_pattern_name][1].input_pattern)

    # Build few-shot with non-deterministic templates.
    # The #. of shots is non-deterministic too.
    # We have 10 templates/task. So in average 2.5 shots/task.
    num_shots_int = 25
    fewshot_base_task_name = f"{t_name}_template_0to10" + pattern_name

    all_patterns = []
    for few_shot_pattern in pattern_dict[flan_pattern_name]:
      all_patterns.append((few_shot_pattern.combined_inputs_w_target_prefix,
                           few_shot_pattern.combined_targets_wo_target_prefix))
    all_formatter = prep.get_batch_formatter(all_patterns)

    for task_suffix, task_output_features in constants.TRAIN_TASK_SUFFIXES_AND_FEATURES:
      seqio.TaskRegistry.add(
          fewshot_base_task_name + task_suffix,
          source=config.source,
          preprocessors=config.preprocessors + [add_template_metadata_fn] + all_formatter +
          prep.FLAN_TOKENIZE,
          postprocess_fn=config.postprocess_fn,
          output_features=task_output_features,
          metric_fns=config.metric_fns)
      # Task names:
      # f'{t_name}_template_0to10_x_shot{suffix}'
      # f'{t_name}_template_0to10_no_opt_x_shot{suffix}'
      few_shot.register_few_shot_version_of_task(
          base_task_name=fewshot_base_task_name + task_suffix,
          new_task_name=(fewshot_base_task_name + "_x_shot" + task_suffix),
          num_shots=num_shots_int,
          prune_exemplars=True,
          max_input_length=constants.FEW_SHOT_MAX_LEN,
          prune_based_on_template_idx=True,
          x_y_delimiter="",
          inputs_prefix="",
          targets_prefix="",
          example_separator="",
          strip_targets=True)

for t_name, config in task_configs.NIV2_TASK_CONFIGS.items():
  flan_pattern_name = utils.t_name_to_flan_pattern_name(t_name)
  mixed_templates = templates.INLINE_FS_PATTERNS[flan_pattern_name]
  x_shot_templates = templates.FEWSHOT_PATTERNS[flan_pattern_name]

  # Task names:
  # f'{t_name}_template_mix_five_shot{suffix}'
  # f'{t_name}_template_mix_no_opt_five_shot{suffix}'
  # The `_no_opt` version is not really used.
  # This is not necessarily five shot. We name it five_shot so that it matches
  # other few-shot task names. This makes it easy to use.
  register_zero_shot_task(f"{t_name}_template_mix_five_shot", config,
                          mixed_templates, "fs_opt")
  register_zero_shot_task(f"{t_name}_template_mix_no_opt_five_shot", config,
                          mixed_templates, "fs_noopt")

  few_shot_patterns = []
  for few_shot_pattern in x_shot_templates:
    few_shot_patterns.append((few_shot_pattern.combined_inputs_w_target_prefix,
                              few_shot_pattern.combined_targets_wo_target_prefix,
                              few_shot_pattern.input_pattern))

  for opt_type_name, template_type in [
      ("", "fs_opt"),
      ("_no_opt", "fs_noopt"),
  ]:
    few_shot_task_name = f"{t_name}_template_0to10{opt_type_name}_x_shot"
    register_niv2_few_shot_task(few_shot_task_name, config,
                                few_shot_patterns, template_type)
