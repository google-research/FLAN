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


# Add zero-shot tasks in task_configs.TASK_CONFIGS.
def register_zero_shot_task(zero_shot_name: str,
                            zero_shot_config: task_configs.TaskConfig,
                            patterns: List[Tuple[str, str]]):
  if len(patterns) == 1:
    formatter = prep.get_formatter(patterns[0][0], patterns[0][1])
  else:
    # This batch formatter applies many prompts to a single task.
    formatter = prep.get_batch_formatter(patterns)
  for suffix, output_features in constants.TRAIN_TASK_SUFFIXES_AND_FEATURES:
    seqio.TaskRegistry.add(
        zero_shot_name + suffix,
        source=zero_shot_config.source,
        preprocessors=zero_shot_config.preprocessors + formatter +
        prep.FLAN_TOKENIZE,
        postprocess_fn=zero_shot_config.postprocess_fn,
        output_features=output_features,
        metric_fns=zero_shot_config.metric_fns)


for t_name, config in task_configs.ALL_CANDIDATE_TASK_CONFIGS.items():
  flan_pattern_name = utils.t_name_to_flan_pattern_name(t_name)
  patterns_list = templates.PATTERNS[flan_pattern_name]

  selected_patterns = patterns_list[0:1]
  zero_shot_task_name = f"{t_name}_template_0_zero_shot"
  register_zero_shot_task(zero_shot_task_name, config, selected_patterns)

  selected_patterns = patterns_list
  zero_shot_task_name = f"{t_name}_template_0to10_zero_shot"
  register_zero_shot_task(zero_shot_task_name, config, selected_patterns)

  # Add tasks that have no answer options provided in their templates at all.
  no_opt_patterns_list = templates.PATTERNS_NO_OPTIONS[flan_pattern_name]
  no_opt_selected_patterns = no_opt_patterns_list
  zero_shot_task_name = f"{t_name}_template_0to10_no_opt_zero_shot"
  register_zero_shot_task(zero_shot_task_name, config, no_opt_selected_patterns)

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
                            selected_patterns)

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
                            selected_patterns)


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
            preprocessors=config.preprocessors +
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
          preprocessors=config.preprocessors + mix_formatter +
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
          preprocessors=config.preprocessors + all_formatter +
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
                          mixed_templates)
  register_zero_shot_task(f"{t_name}_template_mix_no_opt_five_shot", config,
                          mixed_templates)

  # Build few-shot with non-deterministic templates.
  # The #. of shots is non-deterministic too.
  # We have 10 templates/task. So in average 2.5 shots/task.
  num_shots_int = 25
  fewshot_base_task_name = f"{t_name}_template_0to10"

  all_patterns = []
  for few_shot_pattern in x_shot_templates:
    all_patterns.append((few_shot_pattern.combined_inputs_w_target_prefix,
                         few_shot_pattern.combined_targets_wo_target_prefix))
  all_formatter = prep.get_batch_formatter(all_patterns)

  for task_suffix, task_output_features in constants.TRAIN_TASK_SUFFIXES_AND_FEATURES:
    seqio.TaskRegistry.add(
        fewshot_base_task_name + task_suffix,
        source=config.source,
        preprocessors=config.preprocessors + all_formatter + prep.FLAN_TOKENIZE,
        postprocess_fn=config.postprocess_fn,
        output_features=task_output_features,
        metric_fns=config.metric_fns)
    # Task names:
    # f'{t_name}_template_0to10_x_shot{suffix}'
    # f'{t_name}_template_0to10_no_opt_x_shot{suffix}'
    # The `_no_opt` version is not really used.
    for opt_type_name in ["", "_no_opt"]:
      few_shot.register_few_shot_version_of_task(
          base_task_name=fewshot_base_task_name + task_suffix,
          new_task_name=(fewshot_base_task_name + opt_type_name + "_x_shot" +
                         task_suffix),
          num_shots=num_shots_int,
          prune_exemplars=True,
          max_input_length=constants.FEW_SHOT_MAX_LEN,
          prune_based_on_template_idx=True,
          x_y_delimiter="",
          inputs_prefix="",
          targets_prefix="",
          example_separator="",
          strip_targets=True,
          # We always use the 0th template's final_suffix and input_pattern.
          final_suffix=x_shot_templates[0].final_suffix,
          input_pattern=x_shot_templates[0].input_pattern)
