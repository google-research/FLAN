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

"""Configurations of all SeqIO tasks."""

import functools
import os

from flan.v2 import constants
from flan.v2 import constants_niv2
from flan.v2 import constants_t0
from flan.v2 import postprocessors as post
from flan.v2 import preprocessors as prep
from flan.v2 import task_configs_v1
from flan.v2 import utils
import frozendict

import seqio
from t5.evaluation import metrics as t5_metrics
import tensorflow as tf


DEFAULT_OUTPUT_FEATURES = constants.DEFAULT_OUTPUT_FEATURES
NATINST_META_DATA = constants_niv2.NATINST_META_DATA
TaskConfig = task_configs_v1.TaskConfig

FLAN_V0_TASK_CONFIGS = utils.reset_split_maxes_on_flan_v0_configs(
    task_configs_v1.TASK_CONFIGS)
COT_TASK_CONFIGS = {}
DIALOG_TASK_CONFIGS = {}
T0_TASK_CONFIGS = {}
NIV2_TASK_CONFIGS = {}
COT_II_TASK_CONFIGS = {}
DIALOG_II_TASK_CONFIGS = {}


# ========================= lambada ===========================
@seqio.map_over_dataset
def _process_lambada(example):
  example = tf.strings.strip(example["passage"])
  answer = tf.strings.split(example, sep=" ")[-1]
  except_last_word = tf.strings.substr(
      example, 0,
      tf.strings.length(example) - tf.strings.length(answer) - 1)
  return {
      "sentence": except_last_word,
      "answer": answer,
  }


lambada_prep_fn = functools.partial(prep.add_source_info,
      task_name="lambada:1.0.0", task_source="Flan2021")
FLAN_V0_TASK_CONFIGS["lambada"] = TaskConfig(
    source=seqio.TfdsDataSource(tfds_name="lambada:1.0.0", splits=["train"]),
    preprocessors=[_process_lambada, lambada_prep_fn],
    postprocess_fn=post.take_first_word,
    metric_fns=[t5_metrics.accuracy],
)


# ========================= CoT ===========================
# pylint: disable=line-too-long
# cot_type = "cot" if annotations are high-quality and "stream" if annotations
# are low-quality.
COT_DATA_PATH = os.path.join(os.path.dirname(__file__), "cot_data")
for dataset_name, cot_type, nlines in [
    ("gsm8k", "cot", 7473),
    ("strategyqa", "cot", 2061),
    ("creak", "cot", 6915),
    ("qasc", "cot", 1084),
    ("esnli", "cot", 36174),
    ("ecqa", "cot", 7112),
    ("sensemaking", "cot", 6070),
    ("aqua", "stream", 2728),
    ("qed", "stream", 5154),
]:
  cot_prep_fn = functools.partial(prep.add_source_info,
      task_name=f"{cot_type}_{dataset_name}", task_source="CoT")
  COT_TASK_CONFIGS[f"{cot_type}_{dataset_name}"] = TaskConfig(
      source=seqio.TextLineDataSource(
          {"train": os.path.join(COT_DATA_PATH, f"{dataset_name}_train.tsv")},
          num_input_examples={"train": nlines}),
      preprocessors=[prep.simple_cot_tsv, cot_prep_fn],
      postprocess_fn=None,
      metric_fns=[],
  )
  # '_input_inversion' will get mapped to an inverted template
  cot_ii_tname = f"{cot_type}_input_inversion_{dataset_name}"
  COT_II_TASK_CONFIGS[cot_ii_tname] = COT_TASK_CONFIGS[f"{cot_type}_{dataset_name}"]
  cot_ii_prep_fn = functools.partial(prep.add_source_info,
      task_name=f"{cot_type}_{dataset_name}_ii", task_source="CoT")
  COT_II_TASK_CONFIGS[cot_ii_tname].preprocessors = COT_II_TASK_CONFIGS[cot_ii_tname].preprocessors[:-1] + [cot_ii_prep_fn]

# ============ unified_qa, ai2_science_middle with instructions ==============
uqsi_prep_fn = functools.partial(prep.add_source_info,
    task_name=f"unified_qa_science_inst", task_source="Flan2021")
FLAN_V0_TASK_CONFIGS["unified_qa_science_inst"] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name="unified_qa/ai2_science_middle:1.0.0",
        # There are instances in this task that give weird errors,
        # so we use only part of the data.
        splits=["train"]),
    preprocessors=[
        prep.filter_unified_qa_science_inst,
        prep.unified_qa_science_inst,
        uqsi_prep_fn,
        prep.format_options,
    ],
    postprocess_fn=post.take_first_line,
    metric_fns=[t5_metrics.accuracy],
)

# ============================ Wiki Dialog ==============================
wikidialog_prep_fn = functools.partial(prep.add_source_info,
    task_name=f"wiki_dialog", task_source="Dialog")
DIALOG_TASK_CONFIGS["wiki_dialog"] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name="wiki_dialog:1.0.0", splits=["train"]),
    preprocessors=[prep.wiki_dialog, prep.format_dialog, wikidialog_prep_fn],
    postprocess_fn=post.take_first_line,
    metric_fns=[t5_metrics.accuracy],
)
# '_input_inversion' will get mapped to an inverted template
wd_ii_tname = "wiki_dialog_input_inversion"
DIALOG_II_TASK_CONFIGS[wd_ii_tname] = DIALOG_TASK_CONFIGS["wiki_dialog"]
wikidialog_prep_fn = functools.partial(prep.add_source_info,
    task_name=f"wiki_dialog_ii", task_source="Dialog")
DIALOG_II_TASK_CONFIGS[wd_ii_tname].preprocessors = DIALOG_II_TASK_CONFIGS[wd_ii_tname].preprocessors[:-1] + [wikidialog_prep_fn]


# ================================== QReCC ====================================
qrecc_prep_fn = functools.partial(prep.add_source_info,
    task_name=f"qrecc", task_source="Dialog")
DIALOG_TASK_CONFIGS["qrecc"] = TaskConfig(
    source=seqio.TfdsDataSource(tfds_name="q_re_cc:1.0.0", splits=["train"]),
    preprocessors=[prep.filter_qrecc, prep.qrecc, prep.format_dialog, qrecc_prep_fn],
    postprocess_fn=post.take_first_line,
    metric_fns=[t5_metrics.accuracy],
)
# '_input_inversion' will get mapped to an inverted template
qrecc_ii_tname = "qrecc_input_inversion"
DIALOG_II_TASK_CONFIGS[qrecc_ii_tname] = DIALOG_TASK_CONFIGS["qrecc"]
wikidialog_prep_fn = functools.partial(prep.add_source_info,
    task_name=f"qrecc_ii", task_source="Dialog")
DIALOG_II_TASK_CONFIGS[qrecc_ii_tname].preprocessors = DIALOG_II_TASK_CONFIGS[qrecc_ii_tname].preprocessors[:-1] + [qrecc_prep_fn]

# ========================= T0 (P3) Training Sets ===========================
for task_name in constants_t0.T0_TRAIN_TASK_SPLITS:
  subtask_id = task_name.split(":")[-1]
  if constants_t0.T0_TRAIN_TASK_METADATA[task_name]["in_flan"]:
    continue
  # Do not process T0 variants with negative examples.
  # We still keep the variants of these sets in a different format.
  if "_score_eval" in subtask_id:
    continue
  elif constants_t0.T0_TRAIN_TASK_METADATA[task_name][
      "task_type"] == "t0_question_answer":
    preprocessors = [functools.partial(prep.t0, multiple_choice=False)]
    if constants_t0.T0_TRAIN_TASK_METADATA[task_name]["seq_len"]["max"] == 1:
      postprocessors = functools.partial(post.take_first_word)
    else:
      postprocessors = functools.partial(post.take_first_line)
  elif constants_t0.T0_TRAIN_TASK_METADATA[task_name]["task_type"] in [
      "t0_multiple_choice", "t0_multiple_choice_separated_options"
  ]:
    preprocessors = [functools.partial(prep.t0, multiple_choice=True)]
    postprocessors = None
    # Only include non-deterministic options if they aren't already hard-coded.
    if constants_t0.T0_TRAIN_TASK_METADATA[task_name][
        "task_type"] == "t0_multiple_choice_separated_options":
      preprocessors.append(prep.format_options)

  t0_metadata_prep = functools.partial(prep.add_source_info,
    task_name=subtask_id, task_source="P3")
  T0_TASK_CONFIGS[task_name] = TaskConfig(
      source=seqio.TfdsDataSource(
          tfds_name=f"huggingface:bigscience__p3/{subtask_id}",
        #   tfds_name=f"bigscience__p3/{subtask_id}",
          splits=["train"]),
      preprocessors=preprocessors + [t0_metadata_prep],
      postprocess_fn=postprocessors,
      metric_fns=[t5_metrics.accuracy],
  )

# ====================== Natural Instructions v2.5 ======================
# Prepare lookup table for positive example info
niv2_keys, niv2_inputs, niv2_outputs, niv2_exps = [], [], [], []
for niv2_tname, niv2_info in NATINST_META_DATA.items():
  niv2_keys.append(niv2_tname.split(".")[0])
  niv2_inputs.append(niv2_info["positive_examples"]["input"])
  niv2_outputs.append(niv2_info["positive_examples"]["output"])
  niv2_exps.append(niv2_info["positive_examples"]["explanation"])
niv2_posex_input_lookup = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(niv2_keys), tf.constant(niv2_inputs)),
    default_value="")
niv2_posex_output_lookup = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(niv2_keys), tf.constant(niv2_outputs)),
    default_value="")
niv2_posex_exp_lookup = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(niv2_keys), tf.constant(niv2_exps)),
    default_value="")

NIV2_MMLU_TASK_KEYS = tf.constant([
    k.split(".")[0]
    for k in NATINST_META_DATA.keys()
    if int(k.split("_")[0].replace("task", "")) in list(range(685, 738))
])


def filter_mmlu_fn(dataset):
  """Filter MMLU datasets."""

  def filter_func(example):
    task_key = tf.strings.split(example["task_name"], sep=".")[0]
    return not tf.reduce_any(tf.equal(task_key, NIV2_MMLU_TASK_KEYS))

  return dataset.filter(filter_func)


@seqio.map_over_dataset
def lookup_posex_fn(example):
  """Lookup positive example fields and populate the dataset example."""
  task_key = tf.strings.split(example["task_name"], sep=".")[0]
  example["ex_input"] = niv2_posex_input_lookup.lookup(task_key)
  example["ex_output"] = niv2_posex_output_lookup.lookup(task_key)
  example["ex_explanation"] = niv2_posex_exp_lookup.lookup(task_key)
  example["Definition"] = example["definition"]
  example["_task_name"] = example["task_name"]
  example["_task_source"] = "NIv2"
  return example


# Natural Instructions TFDS:
NIV2_TASK_CONFIGS["tfds_natural_instructions"] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name="natural_instructions:1.0.1",
        tfds_data_dir=None,
        splits=["train"]),
    preprocessors=[
        filter_mmlu_fn,
        lookup_posex_fn,
    ],
    postprocess_fn=None,
    metric_fns=[],
)


# =========== Freeze task configs ========== #
FLAN_V0_TASK_CONFIGS = frozendict.frozendict(FLAN_V0_TASK_CONFIGS)
COT_TASK_CONFIGS = frozendict.frozendict(COT_TASK_CONFIGS)
DIALOG_TASK_CONFIGS = frozendict.frozendict(DIALOG_TASK_CONFIGS)
T0_TASK_CONFIGS = frozendict.frozendict(T0_TASK_CONFIGS)
NIV2_TASK_CONFIGS = frozendict.frozendict(NIV2_TASK_CONFIGS)
COT_II_TASK_CONFIGS = frozendict.frozendict(COT_II_TASK_CONFIGS)
DIALOG_II_TASK_CONFIGS = frozendict.frozendict(DIALOG_II_TASK_CONFIGS)


# =========== Define Non-Deterministic Tasks for Mixtures_utils.py ========== #
ALL_CANDIDATE_TASK_CONFIGS = {}
ALL_CANDIDATE_TASK_CONFIGS.update(FLAN_V0_TASK_CONFIGS)
ALL_CANDIDATE_TASK_CONFIGS.update(COT_TASK_CONFIGS)
ALL_CANDIDATE_TASK_CONFIGS.update(DIALOG_TASK_CONFIGS)
ALL_CANDIDATE_TASK_CONFIGS.update(T0_TASK_CONFIGS)
ALL_CANDIDATE_TASK_CONFIGS.update(NIV2_TASK_CONFIGS)
ALL_CANDIDATE_TASK_CONFIGS.update(COT_II_TASK_CONFIGS)
ALL_CANDIDATE_TASK_CONFIGS.update(DIALOG_II_TASK_CONFIGS)
ALL_CANDIDATE_TASK_CONFIGS = frozendict.frozendict(ALL_CANDIDATE_TASK_CONFIGS)

NON_DETER_TASK_NAMES = []
for t_name, config in ALL_CANDIDATE_TASK_CONFIGS.items():
  if prep.format_dialog in config.preprocessors or (prep.format_options
                                                    in config.preprocessors):
    # Tasks that have prep.format_dialog and prep.format_options
    # preprocessors will have non-deterministic versions.
    NON_DETER_TASK_NAMES.append(t_name)
