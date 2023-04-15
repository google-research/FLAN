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
import copy

from flan.v2 import constants
from flan.v2 import constants_niv2
from flan.v2 import constants_xp3
from flan.v2 import constants_t0
from flan.v2 import postprocessors as post
from flan.v2 import preprocessors as prep
from flan.v2 import task_configs_v1
from flan.v2 import utils
import frozendict

import seqio
import datasets
from t5.evaluation import metrics as t5_metrics
import tensorflow as tf
import re
from tqdm import tqdm
import json


DEFAULT_OUTPUT_FEATURES = constants.DEFAULT_OUTPUT_FEATURES
TaskConfig = task_configs_v1.TaskConfig
DEFAULT_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"
DEFAULT_VOCAB = seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH)

XP3_TASK_CONFIGS = {}

# Helper functions from our current xp3 script
# Not sure if they will be needed


def feature_to_spec(feature, length=False):
    if isinstance(feature, datasets.ClassLabel):
        return tf.TensorSpec(shape=() if not length else (None if length == -1 else length,), dtype=tf.int64)
    elif isinstance(feature, datasets.Value):
        return tf.TensorSpec(
            shape=() if not length else (None if length == -1 else length,), dtype=getattr(tf.dtypes, feature.dtype)
        )
    elif hasattr(feature, "dtype") and hasattr(feature, "shape"):
        return tf.TensorSpec(shape=feature.shape, dtype=feature.dtype)
    elif isinstance(feature, datasets.Sequence):
        return feature_to_spec(feature.feature, length=feature.length)
    elif isinstance(feature, list):
        return [feature_to_spec(f, length=length) for f in feature]
    elif isinstance(feature, dict):
        return {k: feature_to_spec(v, length=length) for k, v in feature.items()}
    else:
        raise ValueError(f"Unparseable feature type {type(feature)}")


def hf_dataset_to_tf_dataset(dataset):
    return tf.data.Dataset.from_generator(
        dataset.__iter__, output_signature={
            k: feature_to_spec(v) for k, v in dataset.features.items()}
    )


def get_tf_dataset(split, shuffle_files, dataset_name, subset_name, split_mapping, seed):
    # HF datasets does not support file-level shuffling
    del shuffle_files, seed
    print("we have reached the end of this func")
    dataset = datasets.load_dataset(dataset_name, subset_name)
    dataset = dataset[split_mapping[split]]
    print("we are now moving to the end of this func")
    # dataset = utils.apply_template(dataset, template)
    return hf_dataset_to_tf_dataset(dataset)


def task_clean(text):
    # Clean the text according to allowed characters for a task name
    return re.sub(r"[^\w\d\._]+", "_", text)


def get_task_name(dataset_name, subset_name):
    return task_clean(dataset_name + (f"_{subset_name}_" if subset_name is not None else "_"))


# ========================= XP3 Training Sets ===========================
for task in constants_xp3.XP3_TRAIN_TASKS_SPLIT:
    subtask_id = task.split(':')[-1]
    subset_name = task['subset_name']
    ds_name = task['dataset_name']
    task_name = task['task_name']
    if constants_t0.T0_TRAIN_TASK_METADATA[task]["in_flan"]:
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
                                         task_name=subtask_id, task_source="xP3")
    XP3_TASK_CONFIGS[task] = TaskConfig(
        source=seqio.TfdsDataSource(
            tfds_name=f"huggingface:{ds_name}/{subset_name}",
            #   tfds_name=f"bigscience__p3/{subtask_id}",
            splits=["train"]),
        preprocessors=preprocessors + [t0_metadata_prep],
        postprocess_fn=postprocessors,
        metric_fns=[t5_metrics.accuracy],
    )


# =========== Freeze task configs ========== #
XP3_TASK_CONFIGS = frozendict.frozendict(XP3_TASK_CONFIGS)
