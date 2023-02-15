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

"""Define light-weight seqio tasks for FLAN."""
import collections
import dataclasses
import functools
from typing import Any, Callable, List, Optional, Tuple

import seqio
from t5.data import glue_utils
from t5.data import postprocessors as t5_post
from t5.evaluation import metrics as t5_metrics
import tensorflow.compat.v1 as tf

from flan import baseline_templates
from flan import few_shot
from flan import metrics as gm_metrics
from flan import postprocessors
from flan import preprocessors
from flan import templates
from flan import utils

ShotConfig = few_shot.ShotConfig

# This is a placeholder, for the paper we used an internal vocabulary and model.
VOCAB_FILE = 'gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model'
FLAN_VOCABULARY = seqio.SentencePieceVocabulary(VOCAB_FILE)

FLAN_OUTPUT_FEATURES = {
    'inputs':
        seqio.Feature(
            vocabulary=FLAN_VOCABULARY, add_eos=True, required=False),
    'targets':
        seqio.Feature(vocabulary=FLAN_VOCABULARY, add_eos=True)
}
FLAN_OUTPUT_FEATURES_LM = {
    'targets': seqio.Feature(vocabulary=FLAN_VOCABULARY, add_eos=True)
}


TASK_CONFIGS = {}


@dataclasses.dataclass
class _TaskConfig:
  source: seqio.DataSource
  preprocessors: List[Callable[..., tf.data.Dataset]]
  postprocess_fn: Optional[Callable[..., Any]]
  metric_fns: List[seqio.MetricFnCallable]
  num_multi_shots: int = 1
  # TODO(kguu): we should manually decide `num_multi_shots` for every task.


NUM_TRAIN_EXAMPLES = 30000
NUM_VAL_EXAMPLES = 200
SPLITS_DICT = {
    'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
    'validation': f'validation[:{NUM_VAL_EXAMPLES}]',
    'test': 'test',
}

PARACRAWL_SPLITS_DICT = {
    'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
    'validation': f'train[-{1000+NUM_VAL_EXAMPLES}:-1000]',
    'test': 'train[-1000:]',
}

WMT16_SPLITS_DICT = {
    'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
    'validation': f'validation[:{NUM_VAL_EXAMPLES}]',
    'test': 'test',
}

NUM_VAL_EXAMPLES_WSC = 50
WSC_SPLITS_DICT = {
    'train': f'train[:-{NUM_VAL_EXAMPLES_WSC}]',
    'validation': f'train[-{NUM_VAL_EXAMPLES_WSC}:]',
    'test': 'validation',
}

# Number of templates per task for ablation study.
NUM_TEMPLATES_LIST = [1, 2, 4, 7, 10]


def enumerate_items(items_list):
  num_items = tf.shape(items_list)[0]
  number_list = tf.strings.as_string(tf.range(1, 1 + num_items, 1))
  numbered_items = tf.strings.join([number_list, items_list], separator='. ')
  numbered_items_str = tf.strings.reduce_join(numbered_items, separator='\n')
  return numbered_items_str


# =============================== BoolQ ========================================
@seqio.map_over_dataset
def _process_boolq(example):
  one_hot = tf.one_hot(tf.cast(example['answer'], tf.int32), 2)
  options = tf.constant(['no', 'yes'])
  return {
      'title': example['title'],
      'text': example['passage'],
      'question': example['question'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['bool_q'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='bool_q:1.0.0',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_boolq,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_super_glue_metric('boolq'),
)


# =============================== RTE ========================================
@seqio.map_over_dataset
def _process_rte(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['yes', 'no'])
  glm_options = tf.constant(['true', 'false'])
  return {
      'premise': example['premise'],
      'hypothesis': example['hypothesis'],
      'options': options,
      'glm_options': glm_options,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


TASK_CONFIGS['rte'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='super_glue/rte:1.0.2',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_rte,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_super_glue_metric('rte'),
)


# =============================== Wsc ========================================
@seqio.map_over_dataset
def _process_wsc(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['no', 'yes'])
  return {
      'context': example['text'],
      'text1': example['span1_text'],
      'text2': example['span2_text'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['wsc'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='super_glue/wsc:1.0.2',
        splits=WSC_SPLITS_DICT,
    ),
    preprocessors=[
        _process_wsc,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    # Metric function same as in t5/data/tasks.py
    metric_fns=[t5_metrics.accuracy],
)


# =============================== WSC273 =======================================
@seqio.map_over_dataset
def _process_wsc273(example):
  """WSC 273 dataset."""
  prefix = tf.strings.strip(
      tf.strings.substr(example['text'], 0, example['pronoun_start']))
  suffix = tf.strings.strip(
      tf.strings.substr(example['text'], example['pronoun_end'], -1))
  entities = tf.stack(
      [example['option1_normalized'], example['option2_normalized']])
  options_list_endings = tf.fill(tf.shape(entities), value=suffix)
  options = tf.strings.join([entities, options_list_endings], separator=' ')
  one_hot = tf.one_hot(example['label'], 2)
  return {
      'context': prefix,
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['wsc273'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='wsc273:1.0.0',  # Only the test split is available.
    ),
    preprocessors=[
        _process_wsc273,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)


# =============================== Wic ========================================
@seqio.map_over_dataset
def _process_wic(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['different meanings', 'the same meaning'])
  return {
      'sentence1': example['sentence1'],
      'sentence2': example['sentence2'],
      'word': example['word'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['wic'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='super_glue/wic:1.0.2',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_wic,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_super_glue_metric('wic'),
)


# =============================== Natural Questions ============================
@seqio.map_over_dataset
def _process_natural_questions(example):
  return {
      'question': example['question'] + '?',
      'answer': example['answer'][0],
      'answers': example['answer'],
  }


TASK_CONFIGS['natural_questions'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='natural_questions_open:1.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_natural_questions,
    ],
    postprocess_fn=t5_post.qa,
    # Metric function same as in kera/python/t5/tasks.py
    metric_fns=[t5_metrics.trivia_qa],
)


# =============================== ReCoRD ============================
@seqio.map_over_dataset
def _process_record(example):
  """Processing function for ReCoRD dataset."""

  query_left = tf.strings.strip(
      tf.strings.split(
          example['query'], '@placeholder', result_type='RaggedTensor')[0][0])

  # query_right needs to be appended to all options and answers.
  query_right = tf.strings.split(
      example['query'], '@placeholder', result_type='RaggedTensor')[0][1]

  # Append query_right to options.
  entities = example['entities']
  options_list_endings = tf.fill(tf.shape(entities), value=query_right)
  options = tf.strings.join([entities, options_list_endings], separator='')

  # Append query_right to answers.
  answers = example['answers']
  answers_list_endings = tf.fill(tf.shape(answers), value=query_right)
  answers = tf.strings.join([answers, answers_list_endings])

  # Because options is variable length, make it into a string.
  options_str = tf.strings.reduce_join(
      ['OPTIONS:\n- ',
       tf.strings.reduce_join(options, separator='\n- ')])

  # Remove the "@highlights".
  passage = tf.strings.split(
      example['passage'], '\n@highlight', result_type='RaggedTensor')[0][0]

  return {
      'answer': answers[0],
      'passage': passage,
      'query': query_left,
      'answers': answers,
      'options_str': options_str,
      'options': options,
  }


TASK_CONFIGS['record'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='super_glue/record:1.0.2',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_record,
    ],
    postprocess_fn=t5_post.qa,
    metric_fns=glue_utils.get_super_glue_metric('record'),
)


# ============================== trivia_qa =====================================
@seqio.map_over_dataset
def _process_trivia_qa(example):
  return {
      'question': example['question'],
      'answer': example['answer']['normalized_value'],
      'answers': example['answer']['normalized_aliases'],
  }


def _filter_trivia_qa(dataset):

  def my_fn(example):
    return 'value' in example['answer']

  return dataset.filter(my_fn)


# Using default 'rc' configuration. 'rc.nocontext' has the same examples.
TASK_CONFIGS['trivia_qa'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='trivia_qa/rc:1.1.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _filter_trivia_qa,
        _process_trivia_qa,
    ],
    postprocess_fn=t5_post.qa,
    metric_fns=[t5_metrics.trivia_qa],
)


# =============================== Arc ==========================================
@seqio.map_over_dataset
def _process_arc(example):
  return {
      'question': example['question'],
      'options': example['choices']['text'],
      'answer': example['choices']['text'][example['answerKey']],
      'label': int(example['answerKey']),
  }


def _filter_arc(dataset):

  def my_fn(example):
    return tf.equal(tf.shape(example['options'])[0], 4)

  return dataset.filter(my_fn)


for config_name in ['Challenge', 'Easy']:
  TASK_CONFIGS[f'arc_{config_name.lower()}'] = _TaskConfig(
      source=seqio.TfdsDataSource(
          tfds_name=f'ai2_arc/ARC-{config_name}:1.0.0',
          splits={
              'train': f'train[:-{NUM_VAL_EXAMPLES}]',
              'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
              'test': 'test',
          }),
      preprocessors=[
          _process_arc,
          _filter_arc,
          preprocessors.format_options,
      ],
      postprocess_fn=None,
      metric_fns=[t5_metrics.accuracy],
  )


# =============================== Math Dataset =================================
@seqio.map_over_dataset
def _process_math_dataset(example):
  return {
      'question': example['question'],
      'answer': example['answer'],
  }


# There are other math datasets, but it may not be super helpful to add them in.
TASK_CONFIGS['math_dataset'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='math_dataset/algebra__linear_1d:1.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test',
        }),
    preprocessors=[
        _process_math_dataset,
    ],
    postprocess_fn=None,
    # There is only one correct answer to a math question.
    metric_fns=[t5_metrics.accuracy],
)


# ==================================== aeslc ===================================
@seqio.map_over_dataset
def _process_aeslc(example):
  return {
      'body': tf.strings.regex_replace(example['email_body'], r'\n', ' '),
      'subject': tf.strings.regex_replace(example['subject_line'], r'\n', ''),
  }


def _filter_aeslc(dataset):

  def my_fn(example):
    text = tf.reduce_join([example['email_body'], example['subject_line']])
    text = tf.strings.lower(tf.strings.regex_replace(text, r'\n', ' '))
    long_enough = tf.strings.length(text) > 0
    # If you want to filter out uses of "enron"
    # no_enron = tf.math.logical_not(
    # tf.strings.regex_full_match(text, r'.*enron.*'))
    return long_enough

  return dataset.filter(my_fn)


TASK_CONFIGS['aeslc'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='aeslc:1.0.0',
        splits={
            'train': 'train',
            'validation': f'validation[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test'
        }),
    preprocessors=[
        _filter_aeslc,
        _process_aeslc,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.rouge],
)


# ============================== CNN Dailymail =================================
@seqio.map_over_dataset
def _process_cnn_dailymail(example):
  article = example['article']
  for sep in ['2013 . ', '2014 . ', '2012 . ', ') -- ']:
    article = tf.strings.split(
        article, sep=sep, result_type='RaggedTensor')[-1][0]
  return {
      'text': article,
      'highlights': example['highlights'],
  }


TASK_CONFIGS['cnn_dailymail'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='cnn_dailymail:3.1.0', splits=SPLITS_DICT),
    preprocessors=[
        _process_cnn_dailymail,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.rouge],
)


# ============================== Gigaword ======================================
@seqio.map_over_dataset
def _process_gigaword(example):
  return {
      'text': example['document'],
      'summary': example['summary'],
  }


def _filter_gigaword(dataset):

  def my_fn(example):
    text = tf.reduce_join([example['document'], example['summary']])
    no_unk = tf.logical_not(tf.strings.regex_full_match(text, '.*UNK.*'))
    no_hashtag = tf.logical_not(tf.strings.regex_full_match(text, '.*#.*'))
    return no_unk and no_hashtag

  return dataset.filter(my_fn)


TASK_CONFIGS['gigaword'] = _TaskConfig(
    source=seqio.TfdsDataSource(tfds_name='gigaword:1.2.0', splits=SPLITS_DICT),
    preprocessors=[
        _filter_gigaword,
        _process_gigaword,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.rouge],
)


# ============================== Multi News ====================================
@seqio.map_over_dataset
def _process_multi_news(example):
  return {
      'text': example['document'],
      'summary': example['summary'],
  }


def _filter_multi_news(dataset):

  def my_fn(example):
    return 'document' in example

  return dataset.filter(my_fn)


TASK_CONFIGS['multi_news'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='multi_news:1.0.0', splits=SPLITS_DICT),
    preprocessors=[
        _filter_multi_news,
        _process_multi_news,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.rouge],
)


# ============================== Newsroom ====================================
@seqio.map_over_dataset
def _process_newsroom(example):
  return {
      'text': example['text'],
      'title': example['title'],
      'summary': example['summary'],
  }


TASK_CONFIGS['newsroom'] = _TaskConfig(
    source=seqio.TfdsDataSource(tfds_name='newsroom:1.0.0', splits=SPLITS_DICT),
    preprocessors=[
        _process_newsroom,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.rouge],
)


# =================== Opinion Abstracts Rotten Tomatoes ========================
@seqio.map_over_dataset
def _process_opinion_abstracts_rotten_tomatoes(example):
  return {
      'critic_consensus':
          example['_critic_consensus'],
      'movie':  # Should be title case...
          tf.regex_replace(example['_movie_name'], '_', ' '),
      'first_review':
          example['_critics']['value'][0],
      'numbered_reviews':
          enumerate_items(example['_critics']['value'][:10]),
  }


TASK_CONFIGS['opinion_abstracts_rotten_tomatoes'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='opinion_abstracts/rotten_tomatoes:1.0.0',
        splits={
            'train': 'train[:-600]',
            'validation': 'train[-600:-500]',
            'test': 'train[-500:]',
        }),
    preprocessors=[
        _process_opinion_abstracts_rotten_tomatoes,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.rouge],
)


# ======================= Opinion Abstracts idebate ============================
@seqio.map_over_dataset
def _process_opinion_abstracts_idebate(example):
  return {
      'claim':
          example['_claim'],
      'debate_name':
          example['_debate_name'],
      'argument_sentences':
          enumerate_items(example['_argument_sentences']['value'][:10]),
  }


TASK_CONFIGS['opinion_abstracts_idebate'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='opinion_abstracts/idebate:1.0.0',
        splits={
            'train': 'train[:-600]',
            'validation': 'train[-600:-500]',
            'test': 'train[-500:]',
        }),
    preprocessors=[
        _process_opinion_abstracts_idebate,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.rouge],
)


# ================================= CoQA =======================================
@seqio.map_over_dataset
def _process_coqa(example):
  return {
      'text': example['story'],
      'numbered_questions': enumerate_items(example['questions']),
      'numbered_answers': enumerate_items(example['answers']['input_text']),
  }


TASK_CONFIGS['coqa'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='coqa:1.0.0',
        splits={
            'train': 'train[:-100]',
            'validation': 'train[-100:]',
            'test': 'test',
        }),
    preprocessors=[
        _process_coqa,
    ],
    postprocess_fn=None,
    # Squad, according to the paper: https://arxiv.org/pdf/1808.07042.pdf
    metric_fns=[t5_metrics.squad],
)


# ================================ samsum ======================================
@seqio.map_over_dataset
def _process_samsum(example):
  dialogue = tf.strings.regex_replace(example['dialogue'], '\r\n', '\n')
  dialogue = tf.strings.regex_replace(dialogue, '<.*>', ' ')
  return {
      'summary': example['summary'],
      'dialogue': dialogue,
  }


TASK_CONFIGS['samsum'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='samsum:1.0.0',
        splits={
            'train': 'train',
            'validation': f'validation[:{NUM_VAL_EXAMPLES}]',
            'test': 'test',
        }),
    preprocessors=[
        _process_samsum,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.rouge],
)


# ============================== xsum ==========================================
@seqio.map_over_dataset
def _process_xsum(example):
  return {
      'text': example['document'],
      'summary': example['summary'],
  }


TASK_CONFIGS['xsum'] = _TaskConfig(
    source=seqio.TfdsDataSource(tfds_name='huggingface:xsum', splits=SPLITS_DICT),
    preprocessors=[
        _process_xsum,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.rouge],
)


# ============================== squad_v1 ======================================
@seqio.map_over_dataset
def _process_squad_v1(example):
  return {
      'title': tf.regex_replace(example['title'], '_', ' '),
      'context': example['context'],
      'question': example['question'],
      # Use the first answer as the training target.
      'answer': example['answers']['text'][0],
      # All answers are used for evaluation.
      'answers': example['answers']['text'],
  }


TASK_CONFIGS['squad_v1'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='squad/v1.1:3.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_squad_v1,
    ],
    # This postprocessor will populate `answers` as the ground truth.
    postprocess_fn=t5_post.qa,
    metric_fns=[t5_metrics.squad],
)

# ============================== squad_v2 ======================================
_UNANSWERABLE_RESPONSE = 'unanswerable'


@seqio.map_over_dataset
def _process_squad_v2(example):
  """Squad v2 processing: select from multiple answers when unanswerable."""
  if example['is_impossible']:
    answer = tf.constant(_UNANSWERABLE_RESPONSE)
    answers = tf.constant([_UNANSWERABLE_RESPONSE])
  else:
    answer = example['answers']['text'][0]
    answers = example['answers']['text']
  return {
      'title': tf.regex_replace(example['title'], '_', ' '),
      'context': example['context'],
      'question': example['question'],
      'answer': answer,
      # All answers are used for evaluation.
      'answers': answers,
  }


TASK_CONFIGS['squad_v2'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='squad/v2.0:3.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_squad_v2,
    ],
    postprocess_fn=t5_post.qa,
    metric_fns=[t5_metrics.squad],
)


# ============================== drop ==========================================
@seqio.map_over_dataset
def _process_drop(example):
  return {
      'context': example['passage'],
      'question': example['question'],
      'answer': example['answer'],
      'answers': tf.reshape(example['answer'], [1]),
  }


TASK_CONFIGS['drop'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='drop:2.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'dev',
        }),
    preprocessors=[
        _process_drop,
    ],
    postprocess_fn=t5_post.qa,
    # Uses F1 and EM according to the leaderboard.
    metric_fns=[t5_metrics.squad],
)


# ============================== quac ==========================================
@seqio.map_over_dataset
def _process_quac(example):
  return {
      'title': example['title'],
      'background': example['background'],
      'context': example['context'],
      'question': example['question'],
      'answer': example['orig_answer']['text'],
  }


TASK_CONFIGS['quac'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='quac:1.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_quac,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.trivia_qa],
)


# ================================ multirc =====================================
@seqio.map_over_dataset
def _process_multirc(example):
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 2)
  options = tf.constant(['no', 'yes'])
  glm_options = tf.constant(['False', 'True'])
  return {
      'paragraph': example['paragraph'],
      'question': example['question'],
      'response': example['answer'],
      'options': options,
      'glm_options': glm_options,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
      'idx/paragraph': example['idx']['paragraph'],
      'idx/question': example['idx']['question'],
      'idx/answer': example['idx']['question'],
  }


# Copied from t5_post.multirc, but changed label_classes to ["no", "yes"]
def flan_post_multirc(string_label, example=None, is_target=False):
  """Returns dict containing the class with the question index for grouping."""
  res = {
      'value':
          t5_post.string_label_to_class_id(
              string_label, example=example, label_classes=('no', 'yes'))
  }
  # Add the group, if present, since the model outputs will not have it.
  if is_target and 'idx/question' in example:
    res['group'] = example['idx/question']
  return res


TASK_CONFIGS['multirc'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='super_glue/multirc:1.0.2',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_multirc,
        preprocessors.format_options,
    ],
    postprocess_fn=flan_post_multirc,
    metric_fns=glue_utils.get_super_glue_metric('multirc'),
)


# ============================== ag news =======================================
@seqio.map_over_dataset
def _process_ag_news_subset(example):
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 4)
  options = tf.constant(['World', 'Sports', 'Business', 'Science/Tech'])
  return {
      'title': example['title'],
      'text': example['description'],
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


def _filter_ag_news_subset(dataset):

  def my_fn(example):
    return 'title' in example

  return dataset.filter(my_fn)


TASK_CONFIGS['ag_news_subset'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='ag_news_subset:1.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test'
        }),
    preprocessors=[
        _filter_ag_news_subset,
        _process_ag_news_subset,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    # This is a topic classification dataset
    metric_fns=[t5_metrics.accuracy],
)


# ============================== anli ==========================================
@seqio.map_over_dataset
def _process_anli(example):
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 3)
  options = tf.constant(['Yes', 'It\'s impossible to say', 'No'])
  glm_options = tf.constant(['true', 'neither', 'false'])
  return {
      'context': example['context'],
      'hypothesis': example['hypothesis'],
      'options': options,
      'glm_options': glm_options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


for config_name in ['r1', 'r2', 'r3']:
  # `r1` only has 15k examples, so `train[:30000]` throws an error
  t_set = 'train' if config_name == 'r1' else f'train[:{NUM_TRAIN_EXAMPLES}]'
  TASK_CONFIGS[f'anli_{config_name}'] = _TaskConfig(
      source=seqio.TfdsDataSource(
          tfds_name=f'anli/{config_name}:0.1.0',
          splits={
              'train': t_set,
              'validation': f'validation[:{NUM_VAL_EXAMPLES}]',
              'test': 'test',
          }),
      preprocessors=[
          _process_anli,
          preprocessors.format_options,
      ],
      postprocess_fn=None,
      # Metric function from nlp/unicorn/rainbow/ext5/classification_tasks.py
      metric_fns=[t5_metrics.accuracy],
  )


# ============================== sentiment140 ==================================
@seqio.map_over_dataset
def _process_sentiment140(example):
  """Process sentiment140 from 5 classes into neg/pos."""
  is_positive = tf.logical_or(
      tf.math.equal(example['polarity'], 3),
      tf.math.equal(example['polarity'], 4))
  label = tf.cast(is_positive, tf.int32)
  one_hot = tf.one_hot(label, 2)
  # Prior work uses two classes
  # (https://www.aclweb.org/anthology/C14-1008.pdf,
  # https://arxiv.org/pdf/1404.2188.pdf)
  options = tf.constant(['negative', 'positive'])
  return {
      'text': example['text'],
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


def _filter_sentiment140(dataset):

  def my_fn(example):
    return tf.math.reduce_any([
        tf.math.equal(example['label'], 0),  # negative
        tf.math.equal(example['label'], 1),  # slightly negative
        tf.math.equal(example['label'], 3),  # positive
        tf.math.equal(example['label'], 4)
    ])  # slightly positive

  return dataset.filter(my_fn)


TASK_CONFIGS['sentiment140'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='sentiment140:1.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test',
        }),
    preprocessors=[
        _process_sentiment140,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)


# ============================== story_cloze ===================================
@seqio.map_over_dataset
def _process_story_cloze(example):
  label = tf.cast(tf.add(example['label'], -1), tf.int32)
  one_hot = tf.one_hot(label, 2)
  options = example['endings']
  return {
      'context': example['context'],
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


# The other config, `2018` does not have correct labels in tfds.
TASK_CONFIGS['story_cloze'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='story_cloze/2016:1.0.0',
        splits={
            'train': f'validation[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'validation[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test',
        }),
    preprocessors=[
        _process_story_cloze,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)


# ============================== imdb_reviews ==================================
@seqio.map_over_dataset
def _process_imdb_reviews(example):
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 2)
  options = tf.constant(['negative', 'positive'])
  return {
      'text': tf.regex_replace(example['text'], '<br />', '\n'),
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


# The other configs are `byte` and `subwords8k`/`subwords32k` (restricted
# vocabulary). We don't need them.
TASK_CONFIGS['imdb_reviews'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='imdb_reviews/plain_text:1.0.0',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test'
        }),
    preprocessors=[
        _process_imdb_reviews,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    # Metric function from nlp/unicorn/rainbow/tasks/char_public_tasks.py
    metric_fns=[t5_metrics.accuracy],
)


# ============================== paws_wiki =====================================
@seqio.map_over_dataset
def _process_paws_wiki(example):
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 2)
  options = tf.constant(['no', 'yes'])
  return {
      'sentence1': example['sentence1'],
      'sentence2': example['sentence2'],
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['paws_wiki'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='paws_wiki:1.1.0', splits=SPLITS_DICT),
    preprocessors=[
        _process_paws_wiki,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    # Paper uses accuracy
    metric_fns=[t5_metrics.accuracy],
)


# ====================== definite_pronoun_resolution ===========================
@seqio.map_over_dataset
def _process_definite_pronoun_resolution(example):
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 2)
  options = example['candidates']
  return {
      'sentence': example['sentence'],
      'pronoun': example['pronoun'],
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['definite_pronoun_resolution'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='definite_pronoun_resolution:1.1.0',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test',
        }),
    preprocessors=[
        _process_definite_pronoun_resolution,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    # Metric function from t5/data/tasks.py
    metric_fns=[t5_metrics.accuracy],
)


# =============================== glue_mrpc ====================================
@seqio.map_over_dataset
def _process_glue_mrpc(example):
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 2)
  options = tf.constant(['no', 'yes'])
  return {
      'sentence1': example['sentence1'],
      'sentence2': example['sentence2'],
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['glue_mrpc'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/mrpc:2.0.0',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_glue_mrpc,
        preprocessors.format_options,
    ],
    postprocess_fn=functools.partial(
        t5_post.string_label_to_class_id, label_classes=['no', 'yes']),
    metric_fns=glue_utils.get_glue_metric('mrpc'),
)


# =============================== glue_qqp =====================================
@seqio.map_over_dataset
def _process_glue_qqp(example):
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 2)
  options = tf.constant(['no', 'yes'])
  return {
      'question1': tf.regex_replace(example['question1'], '""', '\''),
      'question2': tf.regex_replace(example['question2'], '""', '\''),
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['glue_qqp'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/qqp:2.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation'  # No test labels available for qqp.
        }),
    preprocessors=[
        _process_glue_qqp,
        preprocessors.format_options,
    ],
    postprocess_fn=functools.partial(
        t5_post.string_label_to_class_id, label_classes=['no', 'yes']),
    metric_fns=glue_utils.get_glue_metric('qqp'),
)


# =================================== copa =====================================
@seqio.map_over_dataset
def _process_copa(example):
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 2)
  options = tf.stack([example['choice1'], example['choice2']])
  connector = tf.strings.regex_replace(example['question'], 'cause', ' because')
  connector = tf.strings.regex_replace(connector, 'effect', ' so')
  glm_premise = tf.strings.regex_replace(example['premise'], r'.$', connector)
  return {
      'premise': example['premise'],
      'question': example['question'],
      'glm_premise': glm_premise,
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['copa'] = _TaskConfig(
    # Test set labels not available for copa.
    source=seqio.TfdsDataSource(
        tfds_name='super_glue/copa:1.0.2',
        splits={
            'train': 'train[:-50]',
            'validation': 'train[-50:]',
            'test': 'validation'
        }),
    preprocessors=[
        _process_copa,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_super_glue_metric('copa'),
)


# ============================== winogrande ====================================
@seqio.map_over_dataset
def _process_winogrande(example):
  """Process the winogrande dataset."""
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 2)
  context = tf.strings.split(
      example['sentence'], '_', result_type='RaggedTensor')[0][0]
  next_sentence = tf.strings.split(
      example['sentence'], '_', result_type='RaggedTensor')[0][1]
  options = tf.stack([
      tf.reduce_join([example['option1'], next_sentence]),
      tf.reduce_join([example['option2'], next_sentence])
  ])
  return {
      'context': context,
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['winogrande'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='winogrande:1.1.0',
        splits={
            'train': f'train_xl[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train_xl[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_winogrande,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)


# ========================== yelp_polarity_reviews =============================
@seqio.map_over_dataset
def _process_yelp_polarity_reviews(example):
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 2)
  options = tf.constant(['negative', 'positive'])
  text = example['text']
  text = tf.regex_replace(text, r'\\""', '"')
  text = tf.regex_replace(text, r'\\n', ' ')
  return {
      'text': text,
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


def _filter_yelp_polarity_reviews(dataset):

  def my_fn(example):
    return 'text' in example and 'label' in example

  return dataset.filter(my_fn)


TASK_CONFIGS['yelp_polarity_reviews'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='yelp_polarity_reviews:0.2.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test',
        }),
    preprocessors=[
        _filter_yelp_polarity_reviews,
        _process_yelp_polarity_reviews,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)


# ============================== cosmos_qa =====================================
@seqio.map_over_dataset
def _process_cosmos_qa(example):
  """Process cosmos_qa dataset example."""
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 4)
  options = tf.stack([
      example['answer0'],
      example['answer1'],
      example['answer2'],
      example['answer3'],
  ])
  return {
      'context': example['context'],
      'question': example['question'],
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


def _filter_cosmos_qa(dataset):

  def my_fn(example):
    return 'label' in example

  return dataset.filter(my_fn)


TASK_CONFIGS['cosmos_qa'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='cosmos_qa:1.0.0',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _filter_cosmos_qa,
        _process_cosmos_qa,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    # Metric function from nlp/unicorn/rainbow/ext5/qa_tasks.py
    metric_fns=[t5_metrics.accuracy],
)


# ============================== para_crawl enes ===============================
@seqio.map_over_dataset
def _process_para_crawl_enes(example):
  return {
      'lang1': 'English',
      'lang2': 'Spanish',
      'sent1': example['en'],
      'sent2': example['es'],
  }


TASK_CONFIGS['para_crawl_enes'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='para_crawl/enes:1.2.0', splits=PARACRAWL_SPLITS_DICT),
    preprocessors=[
        _process_para_crawl_enes,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.bleu],
)


# ============================== wmt14 enfr ====================================
@seqio.map_over_dataset
def _process_wmt14_translate_enfr(example):
  return {
      'lang1': 'English',
      'lang2': 'French',
      'sent1': example['en'],
      'sent2': example['fr'],
  }


TASK_CONFIGS['wmt14_enfr'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='wmt14_translate/fr-en:1.0.0', splits=WMT16_SPLITS_DICT),
    preprocessors=[
        _process_wmt14_translate_enfr,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.bleu],
)


# ============================== wmt16 =========================================
@seqio.map_over_dataset
def _process_wmt16_translate_deen(example):
  return {
      'lang1': 'English',
      'lang2': 'German',
      'sent1': example['en'],
      'sent2': example['de'],
  }


@seqio.map_over_dataset
def _process_wmt16_translate_tren(example):
  return {
      'lang1': 'English',
      'lang2': 'Turkish',
      'sent1': example['en'],
      'sent2': example['tr'],
  }


@seqio.map_over_dataset
def _process_wmt16_translate_csen(example):
  return {
      'lang1': 'English',
      'lang2': 'Czech',
      'sent1': example['en'],
      'sent2': example['cs'],
  }


@seqio.map_over_dataset
def _process_wmt16_translate_fien(example):
  return {
      'lang1': 'English',
      'lang2': 'Finnish',
      'sent1': example['en'],
      'sent2': example['fi'],
  }


@seqio.map_over_dataset
def _process_wmt16_translate_roen(example):
  return {
      'lang1': 'English',
      'lang2': 'Romanian',
      'sent1': example['en'],
      'sent2': example['ro'],
  }


@seqio.map_over_dataset
def _process_wmt16_translate_ruen(example):
  return {
      'lang1': 'English',
      'lang2': 'Russian',
      'sent1': example['en'],
      'sent2': example['ru'],
  }


wmt_language_to_process = {
    'de-en': _process_wmt16_translate_deen,
    'tr-en': _process_wmt16_translate_tren,
    'cs-en': _process_wmt16_translate_csen,
    'fi-en': _process_wmt16_translate_fien,
    'ro-en': _process_wmt16_translate_roen,
    'ru-en': _process_wmt16_translate_ruen,
}

for k in wmt_language_to_process:

  l2, l1 = k.split('-')[0], k.split('-')[1]

  TASK_CONFIGS[f'wmt16_translate_{l2}{l1}'] = _TaskConfig(
      source=seqio.TfdsDataSource(
          tfds_name=f'wmt16_translate/{l2}-{l1}:1.0.0',
          splits=WMT16_SPLITS_DICT),
      preprocessors=[
          wmt_language_to_process[f'{l2}-{l1}'],
      ],
      postprocess_fn=None,
      metric_fns=[t5_metrics.bleu],
  )


# ============================== common_gen ====================================
@seqio.map_over_dataset
def _process_common_gen(example):
  references = tf.concat(
      [example['references'],
       tf.expand_dims(example['target'], axis=0)],
      axis=0)
  return {
      'concepts':
          tf.strings.reduce_join(example['concepts'], separator=', '),
      'concepts_newline':
          tf.strings.reduce_join(example['concepts'], separator='\n'),
      'target':
          example['target'],
      'answers':
          references,
  }


TASK_CONFIGS['common_gen'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='gem/common_gen:1.1.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_common_gen,
    ],
    postprocess_fn=t5_post.qa,
    metric_fns=[gm_metrics.rouge_fn],
)


# ================================== dart ======================================
@seqio.map_over_dataset
def _process_dart(example):
  """Process dart dataset examples."""
  references = tf.concat(
      [example['references'],
       tf.expand_dims(example['target'], axis=0)],
      axis=0)
  tripleset = tf.strings.reduce_join(example['tripleset'], separator='; ')
  tripleset_newline = tf.strings.reduce_join(
      example['tripleset'], separator='\n')
  # Get rid of some undesirable cells like "[TABLECONTEXT]", "[TITLE]"
  tripleset = tf.regex_replace(tripleset, r'\[(.*?)\]', '')
  return {
      'tripleset': tripleset,
      'target': example['target'],
      'answers': references,
      'tripleset_newline': tripleset_newline
  }


TASK_CONFIGS['dart'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='gem/dart:1.1.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_dart,
    ],
    postprocess_fn=t5_post.qa,
    metric_fns=[gm_metrics.rouge_fn],
)


# ============================== e2e_nlg ====================================
@seqio.map_over_dataset
def _process_e2e_nlg(example):
  references = tf.concat(
      [example['references'],
       tf.expand_dims(example['target'], axis=0)],
      axis=0)
  meaning_representation = tf.regex_replace(example['meaning_representation'],
                                            r'\[', ' = ')
  meaning_representation = tf.regex_replace(meaning_representation, r'\]', '')
  return {
      'meaning_representation': meaning_representation,
      'target': example['target'],
      'answers': references,
  }


TASK_CONFIGS['e2e_nlg'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='gem/e2e_nlg:1.1.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_e2e_nlg,
    ],
    postprocess_fn=t5_post.qa,
    metric_fns=[gm_metrics.rouge_fn],
)


# ============================== web_nlg_en ====================================
@seqio.map_over_dataset
def _process_web_nlg_en(example):
  references = tf.concat(
      [example['references'],
       tf.expand_dims(example['target'], axis=0)],
      axis=0)
  input_string = tf.strings.reduce_join(example['input'], separator='; ')
  input_string = tf.regex_replace(input_string, '_', ' ')
  input_string = tf.regex_replace(input_string, r' \| ', ', ')
  return {
      'input_string': input_string,
      'target': example['target'],
      'answers': references,
  }


TASK_CONFIGS['web_nlg_en'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='gem/web_nlg_en:1.1.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_web_nlg_en,
    ],
    postprocess_fn=t5_post.qa,
    metric_fns=[gm_metrics.rouge_fn],
)


# ========================== wiki_lingua_english_en ============================
@seqio.map_over_dataset
def _process_wiki_lingua_english_en(example):
  return {
      'source': example['source'],
      'target': example['target'],
  }


TASK_CONFIGS['wiki_lingua_english_en'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='gem/wiki_lingua_english_en:1.1.0', splits=SPLITS_DICT),
    preprocessors=[
        _process_wiki_lingua_english_en,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.rouge],
)


# ================================= true_case ==================================
@seqio.map_over_dataset
def _process_true_case(example):
  text = example['en']
  return {
      'lower': tf.strings.lower(text),
      'answer': text,
  }


def _filter_true_case(dataset):

  def my_fn(example):
    return tf.logical_not(tf.math.equal(example['lower'], example['answer']))

  return dataset.filter(my_fn)


true_case_val_end = NUM_TRAIN_EXAMPLES + NUM_VAL_EXAMPLES
TASK_CONFIGS['true_case'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='para_crawl/enda:1.2.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[{NUM_TRAIN_EXAMPLES}:{true_case_val_end}]',
            'test': 'train[-3000:]',
        }),
    preprocessors=[
        _process_true_case,
        _filter_true_case,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.edit_distance],
)


# ================================= fix_punct ==================================
@seqio.map_over_dataset
def _process_fix_punct(example):
  text = example['en']
  return {
      'no_punct': tf.strings.regex_replace(text, r'[^\w\s]', ''),
      'answer': text,
  }


def _filter_fix_punct(dataset):

  def my_fn(example):
    return tf.logical_not(tf.math.equal(example['no_punct'], example['answer']))

  return dataset.filter(my_fn)


fix_punct_train_end = true_case_val_end + NUM_TRAIN_EXAMPLES
fix_punct_val_end = fix_punct_train_end + NUM_VAL_EXAMPLES
TASK_CONFIGS['fix_punct'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='para_crawl/enda:1.2.0',
        splits={
            'train': f'train[{true_case_val_end}:{fix_punct_train_end}]',
            'validation': f'train[{fix_punct_train_end}:{fix_punct_val_end}]',
            'test': 'train[-3000:]',
        }),
    preprocessors=[
        _process_fix_punct,
        _filter_fix_punct,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.edit_distance],
)


# ============================== word_segment ==================================
@seqio.map_over_dataset
def _process_word_segment(example):
  text = example['en']
  return {
      'no_space': tf.strings.regex_replace(text, ' ', ''),
      'answer': text,
  }


w_seg_train_end = fix_punct_val_end + NUM_TRAIN_EXAMPLES
w_seg_val_end = w_seg_train_end + NUM_VAL_EXAMPLES
TASK_CONFIGS['word_segment'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='para_crawl/enda:1.2.0',
        splits={
            'train': f'train[{fix_punct_val_end}:{w_seg_train_end}]',
            'validation': f'train[{w_seg_train_end}:{w_seg_val_end}]',
            'test': 'train[-3000:]',
        }),
    preprocessors=[
        _process_word_segment,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.edit_distance],
)


# =============================== CB ========================================
@seqio.map_over_dataset
def _process_cb(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 3)
  options = tf.constant(['Yes', 'No', 'It\'s impossible to say'])
  glm_options = tf.constant(['true', 'neither', 'false'])
  return {
      'premise': example['premise'],
      'hypothesis': example['hypothesis'],
      'options': options,
      'glm_options': glm_options,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


NUM_VAL_EXAMPLES_CB = 50
TASK_CONFIGS['cb'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='super_glue/cb:1.0.2',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES_CB}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES_CB}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_cb,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_super_glue_metric('cb'),
)


# ================================ CoLA ========================================
@seqio.map_over_dataset
def _process_cola(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['unacceptable', 'acceptable'])
  return {
      'sentence': example['sentence'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['cola'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/cola',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_cola,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],  # TODO(jasonwei) matthews_corrcoef
)


# ================================ SST2 ========================================
@seqio.map_over_dataset
def _process_sst2(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['negative', 'positive'])
  return {
      'sentence': example['sentence'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['sst2'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/sst2:2.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_sst2,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_glue_metric('sst2'),
)


# ================================ MNLI ========================================
@seqio.map_over_dataset
def _process_mnli(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 3)
  options = tf.constant(['yes', 'it is not possible to tell', 'no'])
  return {
      'premise': example['premise'],
      'hypothesis': example['hypothesis'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['mnli_matched'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/mnli:2.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation_matched',
        }),
    preprocessors=[
        _process_mnli,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_glue_metric('mnli'),
)

TASK_CONFIGS['mnli_mismatched'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/mnli:2.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation_mismatched',
        }),
    preprocessors=[
        _process_mnli,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_glue_metric('mnli'),
)


# ================================ QNLI ========================================
@seqio.map_over_dataset
def _process_qnli(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['yes', 'no'])
  return {
      'sentence': example['sentence'],
      'question': example['question'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['qnli'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/qnli:2.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_qnli,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_glue_metric('qnli'),
)


# ================================ WNLI ========================================
@seqio.map_over_dataset
def _process_wnli(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['no', 'yes'])
  return {
      'sentence1': example['sentence1'],
      'sentence2': example['sentence2'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['wnli'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/wnli:2.0.0',
        splits={
            'train': 'train[:-30]',
            'validation': 'train[-30:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_wnli,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_glue_metric('wnli'),
)


# ================================ SNLI ========================================
@seqio.map_over_dataset
def _process_snli(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 3)
  options = tf.constant(['yes', 'it is not possible to tell', 'no'])
  return {
      'premise': example['premise'],
      'hypothesis': example['hypothesis'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


def _filter_snli(dataset):

  def my_fn(example):
    return tf.math.reduce_any([
        tf.math.equal(example['label'], 0),
        tf.math.equal(example['label'], 1),
        tf.math.equal(example['label'], 2)
    ])

  return dataset.filter(my_fn)


TASK_CONFIGS['snli'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='snli:1.1.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'validation[:{NUM_VAL_EXAMPLES}]',
            'test': 'test',
        }),
    preprocessors=[
        _filter_snli,
        _process_snli,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)


# ================================ TREC ========================================
@seqio.map_over_dataset
def _process_trec(example):
  one_hot = tf.one_hot(tf.cast(example['label-coarse'], tf.int32), 6)
  options = tf.constant(
      ['description', 'entity', 'abbreviation', 'human', 'numeric', 'location'])
  return {
      'text': example['text'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['trec'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='trec:1.0.0',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test',
        }),
    preprocessors=[
        _process_trec,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)


# ================================ STSB ========================================
@seqio.map_over_dataset
def _process_stsb(example):
  options = tf.constant(['0', '1', '2', '3', '4', '5'])
  return {
      'sentence1': example['sentence1'],
      'sentence2': example['sentence2'],
      'label': example['label'],
      'options': options,
      'answer_str': tf.strings.as_string(example['label'], precision=0),
      'answer': example['label'],
  }


TASK_CONFIGS['stsb'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/stsb:2.0.0',
        splits={
            'train': 'train[:-100]',
            'validation': 'train[-100:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_stsb,
        preprocessors.format_options,
    ],
    postprocess_fn=t5_post.string_to_float,
    metric_fns=glue_utils.get_glue_metric('stsb'),
)


# ================================= PIQA =======================================
@seqio.map_over_dataset
def _process_piqa(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = [example['sol1'], example['sol2']]
  return {
      'goal': example['goal'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['piqa'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='piqa:1.0.0',
        splits={
            'train': 'train[:-100]',
            'validation': 'train[-100:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_piqa,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)


# ============================= OpenbookQA =====================================
@seqio.map_over_dataset
def _process_openbookqa(example):
  one_hot = tf.one_hot(tf.cast(example['answerKey'], tf.int32), 4)
  options = [
      example['question']['choice_A'], example['question']['choice_B'],
      example['question']['choice_C'], example['question']['choice_D']
  ]
  return {
      'question': example['question']['stem'],
      'fact': example['fact1'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['openbookqa'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='openbookqa:0.1.0',
        splits={
            'train': 'train',
            'validation': f'validation[:{NUM_VAL_EXAMPLES}]',
            'test': 'test',
        }),
    preprocessors=[
        _process_openbookqa,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)


# ================================ Hellaswag ===================================
@seqio.map_over_dataset
def _process_hellaswag(example):
  """Clean up hellaswag."""
  # Model will likely have a hard time producing a string with brackets.
  context = tf.strings.regex_replace(example['context'], r'\[header\]\s', '')
  context = tf.strings.regex_replace(context, r'\[.*?\]\s', '\n')

  options = tf.strings.regex_replace(example['endings'], r'\[.*?\]\s', '')
  # In case you want to shorten the endings, which are often multi-sentence.
  # endings = tf.strings.split(
  #     endings, sep='.', result_type='RaggedTensor')[:, :1]
  # endings = tf.strings.reduce_join(endings, axis=1)

  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 4)
  return {
      'context': context,
      'options': options,
      'activity_label': example['activity_label'],
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


def _filter_hellaswag(dataset):
  """Ensure that labels exist, and that there are three options."""

  def my_fn(example):
    return tf.math.reduce_any([
        tf.math.equal(example['label'], 0),
        tf.math.equal(example['label'], 1),
        tf.math.equal(example['label'], 2),
        tf.math.equal(example['label'], 3),
    ])

  return dataset.filter(my_fn)


TASK_CONFIGS['hellaswag'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='hellaswag:1.1.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _filter_hellaswag,
        _process_hellaswag,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)


# ========================== ADD TASKS TO REGISTRY. ============================
def register_few_shot_versions_of_task(zero_shot_name: str,
                                       prune_exemplars: bool = False,
                                       multishot_max_num_shots: int = 16,
                                       max_input_length: Optional[int] = None):
  """Registers one-shot and multi-shot versions of a zero-shot Task.

  Args:
    zero_shot_name: the base zero-shot task name.
    prune_exemplars: If True, prune excessive exemplars by input length.
    multishot_max_num_shots: Maximum number of exemplars for multi-shot tasks.
    max_input_length: Maximum length of all exemplars.
  """
  for shot_config in ShotConfig:
    if shot_config in [ShotConfig.ZERO, ShotConfig.MULTI]:
      continue
    # X-shot version of the task.
    few_shot.register_few_shot_version_of_task(
        base_task_name=zero_shot_name,
        new_task_name=zero_shot_name + shot_config.name_suffix,
        num_shots=shot_config.value,
        prune_exemplars=prune_exemplars,
        max_input_length=max_input_length)

  # Multi-shot version of the task.
  few_shot.register_few_shot_version_of_task(
      base_task_name=zero_shot_name,
      new_task_name=zero_shot_name + ShotConfig.MULTI.name_suffix,
      num_shots=multishot_max_num_shots,
      prune_exemplars=prune_exemplars,
      max_input_length=max_input_length)


def register_few_shot_versions_of_continuations_task(
    zero_shot_name: str) -> None:
  """Registers one, two, three, and five-shot versions of a zero-shot Task."""

  x_y_delimiter = ' '
  inputs_prefix = ''
  targets_prefix = ''
  example_separator = '\n\n'
  final_suffix = ''

  for shot_config in ShotConfig:
    if shot_config in [ShotConfig.ZERO, ShotConfig.MULTI]:
      continue
    few_shot.register_few_shot_version_of_task(
        base_task_name=zero_shot_name,
        new_task_name=zero_shot_name + shot_config.name_suffix,
        num_shots=shot_config.value,
        x_y_delimiter=x_y_delimiter,
        inputs_prefix=inputs_prefix,
        targets_prefix=targets_prefix,
        example_separator=example_separator,
        final_suffix=final_suffix,
    )


for t_name, config in TASK_CONFIGS.items():
  flan_pattern_name = utils.t_name_to_flan_pattern_name(t_name)
  for idx, patterns in enumerate(templates.PATTERNS[flan_pattern_name]):
    # Zero-shot task instantiated from template {idx}.
    # Note: Used primarily for zeroshot eval.
    inputs_pattern, targets_pattern = patterns

    # Task names:
    # Zero-shot version: f'{t_name}_type_{idx}'
    # One-shot version: f'{t_name}_type_{idx}_one_shot'
    # Multi-shot version: f'{t_name}_type_{idx}_multi_shot'

    # Zero-shot version of the task.
    zero_shot_task_name = utils.ZeroshotEvalTaskName.get(t_name, idx)
    seqio.TaskRegistry.add(
        zero_shot_task_name,
        source=config.source,
        preprocessors=config.preprocessors +
        # Format inputs and outputs according to the patterns. This should be
        # the same for all tasks.
        preprocessors.get_flan_formatter(inputs_pattern, targets_pattern) +
        # Tokenization for the prefix-LM. This should be the same for all tasks.
        preprocessors.FLAN_TOKENIZE,
        postprocess_fn=config.postprocess_fn,
        output_features=FLAN_OUTPUT_FEATURES,
        metric_fns=config.metric_fns)

    # Few-shot versions of the task.
    register_few_shot_versions_of_task(
        zero_shot_task_name,
        prune_exemplars=True,
        # The current input length is 1024. Save 64 for seperators.
        max_input_length=960,
        multishot_max_num_shots=16)

    if utils.is_classification(flan_pattern_name):
      zero_shot_task_name = utils.ZeroshotScoreEvalTaskName.get(
          t_name, idx)
      # Zeroshot rank-classifcation tasks from template {idx}.
      # Note: These are only used for scoring/rank-classification eval.
      # Task names:
      # Zero-shot version: f'{t_name}_type_{idx}_scoring_eval'
      # One-shot version: f'{t_name}_type_{idx}_scoring_eval_one_shot'
      # Multi-shot version: f'{t_name}_type_{idx}_scoring_eval_multi_shot'

      seqio.TaskRegistry.add(
          # Task name: f'{t_name}_type_{idx}_scoring_eval'
          zero_shot_task_name,
          source=config.source,
          preprocessors=config.preprocessors +
          # Format inputs and outputs according to the patterns. This should be
          # the same for all tasks.
          preprocessors.get_flan_formatter(inputs_pattern, targets_pattern) +
          [preprocessors.rank_classification_from_options] +
          # Tokenization for the prefix-LM. This should be the same for all
          # tasks.
          preprocessors.FLAN_TOKENIZE,
          postprocess_fn=t5_post.rank_classification,
          output_features=FLAN_OUTPUT_FEATURES,
          metric_fns=[t5_metrics.rank_classification])

      register_few_shot_versions_of_task(
          zero_shot_task_name,
          prune_exemplars=True,
          max_input_length=960,
          multishot_max_num_shots=16)

    # Zeroshot rank-classifcation tasks from template {idx}
    # WITHOUT OPTIONS OR FLAN.
    # These are only used for scoring eval for the baseline.
    if utils.is_classification(flan_pattern_name):
      inputs_pattern_no_options = utils.remove_input_patterns_options(
          inputs_pattern)
      seqio.TaskRegistry.add(
          # Task name: f'{t_name}_type_{idx}_score_eval_no_options'
          utils.ZeroshotScoreEvalNoOptionTaskName.get(t_name, idx),
          source=config.source,
          preprocessors=config.preprocessors +
          # Format inputs and outputs according to the patterns. This should be
          # the same for all tasks.
          preprocessors.get_flan_formatter(inputs_pattern_no_options,
                                            targets_pattern) +
          [preprocessors.rank_classification_from_options] +
          # Tokenization for the prefix-LM. This should be the same for all
          # tasks.
          preprocessors.FLAN_TOKENIZE,
          postprocess_fn=t5_post.rank_classification,
          output_features=FLAN_OUTPUT_FEATURES,
          metric_fns=[t5_metrics.rank_classification])

    # Zeroshot rank-classifcation tasks from template {idx}
    # Without options, but keep flan clean.
    # These are only used for scoring eval for flan.
    if utils.is_classification(flan_pattern_name):
      inputs_pattern_no_options = utils.remove_input_patterns_options(
          inputs_pattern)
      seqio.TaskRegistry.add(
          # Task name: f'{t_name}_type_{idx}_score_flan_no_options'
          utils.ZeroshotScoreFLANNoOptionTaskName.get(t_name, idx),
          source=config.source,
          preprocessors=config.preprocessors +
          # Format inputs and outputs according to the patterns. This should be
          # the same for all tasks.
          preprocessors.get_flan_formatter(inputs_pattern_no_options,
                                            targets_pattern) +
          [preprocessors.rank_classification_from_options] +
          # Tokenization for the prefix-LM. This should be the same for all
          # tasks.
          preprocessors.FLAN_TOKENIZE,
          postprocess_fn=t5_post.rank_classification,
          output_features=FLAN_OUTPUT_FEATURES,
          metric_fns=[t5_metrics.rank_classification])

  # Add a single task with all templates for that task.
  patterns_list = templates.PATTERNS[flan_pattern_name]
  for num_templates in NUM_TEMPLATES_LIST:
    selected_patterns = patterns_list[:num_templates]

    # Task names:
    # Zero-shot version: f'{t_name}_{num_templates}templates'
    # One-shot version: f'{t_name}_{num_templates}templates_one_shot'
    # Multi-shot version: f'{t_name}_{num_templates}templates_multi_shot'

    # Zero-shot version of the task.
    # Note: Used primarily for training.
    zero_shot_task_name = utils.ZeroshotTemplatedTaskName.get(
        t_name, num_templates)
    seqio.TaskRegistry.add(
        zero_shot_task_name,
        source=config.source,
        preprocessors=config.preprocessors +
        # This batch formatter applies many prompts to a single task.
        preprocessors.get_batch_flan_formatter(selected_patterns) +
        preprocessors.FLAN_TOKENIZE,
        output_features=FLAN_OUTPUT_FEATURES,
        metric_fns=config.metric_fns)

    # Few-shot versions of the task.
    register_few_shot_versions_of_task(
        zero_shot_task_name,
        prune_exemplars=True,
        max_input_length=960,
        multishot_max_num_shots=16)

  # For backwards compatibility.
  seqio.TaskRegistry.add(
      # Task name: f'{t_name}_all_prompts'.
      utils.AllPromptsTaskName.get(t_name),
      source=config.source,
      preprocessors=config.preprocessors +
      # This batch formatter applies many prompts to a single task.
      preprocessors.get_batch_flan_formatter(patterns_list) +
      preprocessors.FLAN_TOKENIZE,
      output_features=FLAN_OUTPUT_FEATURES,
      metric_fns=config.metric_fns)

  # Add task for non-flan baseline evaluation.
  if flan_pattern_name in baseline_templates.PATTERNS:
    continuation_patterns = baseline_templates.PATTERNS[flan_pattern_name]
    for idx, patterns in enumerate(continuation_patterns):
      inputs_pattern, targets_pattern = patterns
      name = f'continuations_{t_name}_type_{idx}'
      if utils.is_classification(flan_pattern_name):
        name += '_scoring_eval'
        seqio.TaskRegistry.add(
            name,
            source=config.source,
            preprocessors=config.preprocessors +
            preprocessors.get_glm_formatter(inputs_pattern, targets_pattern) +
            [preprocessors.GLM_RANK_CLASSIFICATION] +
            preprocessors.FLAN_TOKENIZE,
            postprocess_fn=t5_post.rank_classification,
            output_features=FLAN_OUTPUT_FEATURES,
            metric_fns=[t5_metrics.rank_classification])
      else:
        seqio.TaskRegistry.add(
            name,
            source=config.source,
            preprocessors=config.preprocessors +
            preprocessors.get_glm_formatter(inputs_pattern, targets_pattern) +
            preprocessors.FLAN_TOKENIZE,
            postprocess_fn=postprocessors.parse_glm_qa_answer,
            output_features=FLAN_OUTPUT_FEATURES,
            metric_fns=config.metric_fns)

      # Few-shot versions of the task.
      register_few_shot_versions_of_continuations_task(name)

