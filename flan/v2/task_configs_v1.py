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

"""Library of task configs for FLAN.

`TaskConfig` is a dataclass used to define how a dataset should be loaded and
preprocessed in FLAN Seqio tasks. Task configs in this module are used in
`tasks.py` to define various zeroshot and fewshot Seqio tasks.
"""
import dataclasses
import functools
from typing import Any, Callable, Dict, List, Mapping, Optional

from flan.v2 import preprocessors as prep
import seqio
from t5.data import glue_utils
from t5.data import postprocessors as t5_post
from t5.data import preprocessors as t5_prep
from t5.evaluation import metrics as t5_metrics
import tensorflow.compat.v1 as tf
import tqdm


@dataclasses.dataclass
class TaskConfig:
  """Dataclass for FLAN task config."""
  source: seqio.DataSource
  preprocessors: List[Callable[..., tf.data.Dataset]]
  postprocess_fn: Optional[Callable[..., Any]]
  metric_fns: List[seqio.MetricFnCallable]
  num_multi_shots: int = 1
  # TODO(kguu): we should manually decide `num_multi_shots` for every task.
  source_args: Optional[Mapping[str, Any]] = None


TASK_CONFIGS: Dict[str, TaskConfig] = {}

NUM_TRAIN_EXAMPLES = 30000
NUM_VAL_EXAMPLES = 200
SPLITS_DICT = {
    'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
    'validation': f'validation[:{NUM_VAL_EXAMPLES}]',
    'test': 'test',
}


# Multi-rouge/multi-bleu. When there are multiple references, we want to get the
# rouge score that is highest. According to the authors, this is how it was done
# in the GEM paper.
# Source:
#   https://github.com/google/BIG-bench/blob/main/bigbench/api/task_metrics.py
def rouge_fn(targets: List[List[str]],
             predictions: List[str]) -> Dict[str, float]:
  """Computes ROUGE by taking the max ROUGE-N per reference and N."""
  # Following strategy from https://www.aclweb.org/anthology/W04-1013/.
  # Identify best reference per response and ROUGE type.
  rouge_types = ['rouge1', 'rouge2', 'rougeLsum']
  max_references = {rouge_type: [] for rouge_type in rouge_types}
  for targ_for_resp, resp in tqdm.tqdm(
      zip(targets, predictions), total=len(targets)):
    # Compute individual scores per example/ref pair.
    resp_scores = [t5_metrics.rouge([t], [resp]) for t in targ_for_resp]
    # Find best scoring references for generated output and ROUGE type.
    for rouge_type in rouge_types:
      def _f(x, rs=resp_scores, rt=rouge_type):
        return rs[x][rt]
      best_score_index = max(
          range(len(resp_scores)), key=_f)
      best_ref = targ_for_resp[best_score_index]
      # Add the reference to the new reference list.
      max_references[rouge_type].append(best_ref)
  # Compute metric for each of the reference lists for a ref type.
  results = {}
  for rouge_type in rouge_types:
    results[rouge_type] = t5_metrics.rouge(max_references[rouge_type],
                                           predictions)[rouge_type]
  return results


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


bool_q_source_args = {
    'tfds_name': 'bool_q:1.0.0',
    'splits': {
        'train': f'train[:-{NUM_VAL_EXAMPLES}]',
        'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
        'test': 'validation'
    }
}
TASK_CONFIGS['bool_q'] = TaskConfig(
    source=seqio.TfdsDataSource(**bool_q_source_args),
    preprocessors=[
        _process_boolq,
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_super_glue_metric('boolq'),
    source_args=bool_q_source_args)


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


rte_source_args = {
    'tfds_name': 'super_glue/rte:1.0.2',
    'splits': {
        'train': f'train[:-{NUM_VAL_EXAMPLES}]',
        'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
        'test': 'validation',
    }
}
TASK_CONFIGS['rte'] = TaskConfig(
    source=seqio.TfdsDataSource(**rte_source_args),
    preprocessors=[
        _process_rte,
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_super_glue_metric('rte'),
    source_args=rte_source_args,
)

# =============================== Wsc ========================================
NUM_VAL_EXAMPLES_WSC = 50
WSC_SPLITS_DICT = {
    'train': f'train[:-{NUM_VAL_EXAMPLES_WSC}]',
    'validation': f'train[-{NUM_VAL_EXAMPLES_WSC}:]',
    'test': 'validation',
}


@seqio.map_over_dataset
def _process_wsc(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['no', 'yes'])
  return {
      'context':
          tf.strings.regex_replace(
              t5_prep._wsc_inputs(example),  # pylint: disable=protected-access
              r' X ',
              ' *' + example['span2_text'] + '* '),
      'text1':
          example['span1_text'],
      'text2':
          example['span2_text'],
      'options':
          options,
      'answer':
          tf.boolean_mask(options, one_hot)[0],
  }


wsc_source_args = {
    'tfds_name': 'super_glue/wsc.fixed:1.0.2',
    'splits': WSC_SPLITS_DICT
}
TASK_CONFIGS['wsc'] = TaskConfig(
    source=seqio.TfdsDataSource(**wsc_source_args),
    preprocessors=[
        _process_wsc,
        prep.format_options,
    ],
    postprocess_fn=None,
    # Metric function same as in t5/data/tasks.py
    metric_fns=[t5_metrics.accuracy],
    source_args=wsc_source_args,
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


TASK_CONFIGS['wsc273'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='wsc273:1.0.0',  # Only the test split is available.
    ),
    preprocessors=[
        _process_wsc273,
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)


# =============================== Wic ========================================
@seqio.map_over_dataset
def _process_wic(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['different meanings', 'the same meaning'])
  # options = tf.constant(['no', 'yes'])
  return {
      'sentence1': example['sentence1'],
      'sentence2': example['sentence2'],
      'word': example['word'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


wic_source_args = {
    'tfds_name': 'super_glue/wic:1.0.2',
    'splits': {
        'train': f'train[:-{NUM_VAL_EXAMPLES}]',
        'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
        'test': 'validation',
    }
}
TASK_CONFIGS['wic'] = TaskConfig(
    source=seqio.TfdsDataSource(**wic_source_args),
    preprocessors=[
        _process_wic,
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_super_glue_metric('wic'),
    source_args=wic_source_args,
)


# =============================== Natural Questions ============================
@seqio.map_over_dataset
def _process_natural_questions(example):
  return {
      'question': example['question'] + '?',
      'answer': example['answer'][0],
      'answers': example['answer'],
  }


TASK_CONFIGS['natural_questions'] = TaskConfig(
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
      'answer_full_str': answers[0],
      'passage': passage,
      'query': query_left,
      'answers': answers,
      'options_str': options_str,
      'options': options,
  }


record_source_args = {
    'tfds_name': 'super_glue/record:1.0.2',
    'splits': {
        'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
        'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
        'test': 'validation',
    }
}
TASK_CONFIGS['record'] = TaskConfig(
    source=seqio.TfdsDataSource(**record_source_args),
    preprocessors=[
        _process_record,
    ],
    postprocess_fn=t5_post.qa,
    metric_fns=glue_utils.get_super_glue_metric('record'),
    source_args=record_source_args,
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
TASK_CONFIGS['trivia_qa'] = TaskConfig(
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
  TASK_CONFIGS[f'arc_{config_name.lower()}'] = TaskConfig(
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
          prep.format_options,
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
TASK_CONFIGS['math_dataset'] = TaskConfig(
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
  """Remove empty email."""

  def my_fn(example):
    text = tf.reduce_join([example['email_body'], example['subject_line']])
    text = tf.strings.lower(tf.strings.regex_replace(text, r'\n', ' '))
    long_enough = tf.strings.length(text) > 0
    # If you want to filter out uses of "enron"
    # no_enron = tf.math.logical_not(
    # tf.strings.regex_full_match(text, r'.*enron.*'))
    return long_enough

  return dataset.filter(my_fn)


TASK_CONFIGS['aeslc'] = TaskConfig(
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
        article, sep=sep, result_type='RaggedTensor')[0][-1]
  highlights = example['highlights']
  highlights = tf.strings.regex_replace(highlights, r' \.', '.')
  highlights = tf.strings.regex_replace(highlights, '\n', ' ')
  return {
      'text': article,
      'highlights': highlights,
  }


TASK_CONFIGS['cnn_dailymail'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='cnn_dailymail:3.4.0', splits=SPLITS_DICT),
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


TASK_CONFIGS['gigaword'] = TaskConfig(
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


TASK_CONFIGS['multi_news'] = TaskConfig(
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


TASK_CONFIGS['newsroom'] = TaskConfig(
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
          prep.numbered_items_str(example['_critics']['value'][:10]),
  }


oart_prep_fn = functools.partial(prep.add_source_info,
    task_name="opinion_abstracts_rotten_tomatoes", task_source="Flan2021")
TASK_CONFIGS['opinion_abstracts_rotten_tomatoes'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='opinion_abstracts/rotten_tomatoes:1.0.0',
        splits={
            'train': 'train[:-600]',
            'validation': 'train[-600:-500]',
            'test': 'train[-500:]',
        }),
    preprocessors=[
        oart_prep_fn,
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
          prep.numbered_items_str(example['_argument_sentences']['value'][:10]),
  }

oaid_prep_fn = functools.partial(prep.add_source_info,
    task_name="opinion_abstracts_idebate", task_source="Flan2021")
TASK_CONFIGS['opinion_abstracts_idebate'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='opinion_abstracts/idebate:1.0.0',
        splits={
            'train': 'train[:-600]',
            'validation': 'train[-600:-500]',
            'test': 'train[-500:]',
        }),
    preprocessors=[
        oaid_prep_fn,
        _process_opinion_abstracts_idebate,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.rouge],
)


# ================================= CoQA =======================================
@seqio.map_over_dataset
def _process_coqa(example):
  return {
      'text':
          example['story'],
      'numbered_questions':
          prep.numbered_items_str(example['questions']),
      'numbered_answers':
          prep.numbered_items_str(example['answers']['input_text']),
  }


TASK_CONFIGS['coqa'] = TaskConfig(
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


TASK_CONFIGS['samsum'] = TaskConfig(
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


TASK_CONFIGS['xsum'] = TaskConfig(
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


TASK_CONFIGS['squad_v1'] = TaskConfig(
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


TASK_CONFIGS['squad_v2'] = TaskConfig(
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


TASK_CONFIGS['drop'] = TaskConfig(
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

  def _remove_cannotanswer(str_field):
    return tf.strings.strip(
        tf.strings.regex_replace(str_field, 'CANNOTANSWER', ''))

  return {
      'title': _remove_cannotanswer(example['title']),
      'background': _remove_cannotanswer(example['background']),
      'context': _remove_cannotanswer(example['context']),
      'question': _remove_cannotanswer(example['question']),
      'answer': _remove_cannotanswer(example['orig_answer']['text']),
  }


TASK_CONFIGS['quac'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='quac:1.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_quac,
        functools.partial(prep.filter_non_strings, field='answer'),
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.trivia_qa],
)


# ================================ multirc =====================================
@seqio.map_over_dataset
def _process_multirc(example):
  """Process multirc."""
  response = example['answer']
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 2)
  options = tf.constant(['no', 'yes'])
  glm_options_left = tf.constant(['[False]', '[True]'])
  glm_options_right = tf.fill(tf.shape(glm_options_left), value=response)
  glm_options = tf.strings.join([glm_options_left, glm_options_right],
                                separator=' ')
  return {
      'paragraph': example['paragraph'],
      'question': example['question'],
      'response': response,
      'options': options,
      'glm_options': glm_options,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
      'idx/paragraph': example['idx']['paragraph'],
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


multirc_source_args = {
    'tfds_name': 'super_glue/multirc:1.0.2',
    'splits': {
        'train': f'train[:-{NUM_VAL_EXAMPLES}]',
        'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
        'test': 'validation',
    }
}
TASK_CONFIGS['multirc'] = TaskConfig(
    source=seqio.TfdsDataSource(**multirc_source_args),
    preprocessors=[
        _process_multirc,
        prep.format_options,
    ],
    postprocess_fn=flan_post_multirc,
    metric_fns=glue_utils.get_super_glue_metric('multirc'),
    source_args=multirc_source_args,
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


TASK_CONFIGS['ag_news_subset'] = TaskConfig(
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
        prep.format_options,
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
  TASK_CONFIGS[f'anli_{config_name}'] = TaskConfig(
      source=seqio.TfdsDataSource(
          tfds_name=f'anli/{config_name}:0.1.0',
          splits={
              'train': t_set,
              'validation': f'validation[:{NUM_VAL_EXAMPLES}]',
              'test': 'test',
          }),
      preprocessors=[
          _process_anli,
          prep.format_options,
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
  glm_options = tf.constant(['negative tweet.', 'positive tweet.'])
  return {
      'text': example['text'],
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_options': glm_options,
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


TASK_CONFIGS['sentiment140'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='sentiment140:1.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test',
        }),
    preprocessors=[
        _process_sentiment140,
        prep.format_options,
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
TASK_CONFIGS['story_cloze'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='story_cloze/2016:1.0.0',
        splits={
            'train': f'validation[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'validation[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test',
        }),
    preprocessors=[
        _process_story_cloze,
        prep.format_options,
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
  glm_options = tf.constant(
      ['negative movie review.', 'positive movie review.'])
  return {
      'text': tf.regex_replace(example['text'], '<br />', '\n'),
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_options': glm_options,
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


# The other configs are `byte` and `subwords8k`/`subwords32k` (restricted
# vocabulary). We don't need them.
TASK_CONFIGS['imdb_reviews'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='imdb_reviews/plain_text:1.0.0',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test'
        }),
    preprocessors=[
        _process_imdb_reviews,
        prep.format_options,
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
  glm_options = tf.constant(['different meanings.', 'the same meaning.'])
  return {
      'sentence1': example['sentence1'],
      'sentence2': example['sentence2'],
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_options': glm_options,
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


TASK_CONFIGS['paws_wiki'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='paws_wiki:1.1.0', splits=SPLITS_DICT),
    preprocessors=[
        _process_paws_wiki,
        prep.format_options,
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


TASK_CONFIGS['definite_pronoun_resolution'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='definite_pronoun_resolution:1.1.0',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test',
        }),
    preprocessors=[
        _process_definite_pronoun_resolution,
        prep.format_options,
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
  glm_options = tf.constant(['different meanings.', 'the same meaning.'])
  return {
      'sentence1': example['sentence1'],
      'sentence2': example['sentence2'],
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_options': glm_options,
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


TASK_CONFIGS['glue_mrpc'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/mrpc:2.0.0',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_glue_mrpc,
        prep.format_options,
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
  glm_options = tf.constant(
      ['asking about different things.', 'asking about the same thing.'])
  return {
      'question1': tf.regex_replace(example['question1'], '""', '\''),
      'question2': tf.regex_replace(example['question2'], '""', '\''),
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_options': glm_options,
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


TASK_CONFIGS['glue_qqp'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/qqp:2.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation'  # No test labels available for qqp.
        }),
    preprocessors=[
        _process_glue_qqp,
        prep.format_options,
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


copa_source_args = {
    'tfds_name': 'super_glue/copa:1.0.2',
    'splits': {
        'train': 'train[:-50]',
        'validation': 'train[-50:]',
        'test': 'validation'
    }
}
TASK_CONFIGS['copa'] = TaskConfig(
    # Test set labels not available for copa.
    source=seqio.TfdsDataSource(**copa_source_args),
    preprocessors=[
        _process_copa,
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_super_glue_metric('copa'),
    source_args=copa_source_args,
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


TASK_CONFIGS['winogrande'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='winogrande:1.1.0',
        splits={
            'train': f'train_xl[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train_xl[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_winogrande,
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)


# ========================== yelp_polarity_reviews =============================
@seqio.map_over_dataset
def _process_yelp_polarity_reviews(example):
  """Process tfds for yelp polarity reviews."""
  label = tf.cast(example['label'], tf.int32)
  one_hot = tf.one_hot(label, 2)
  options = tf.constant(['negative', 'positive'])
  glm_options = tf.constant(['negative yelp review.', 'positive yelp review.'])
  text = example['text']
  text = tf.regex_replace(text, r'\\""', '"')
  text = tf.regex_replace(text, r'\\n', ' ')
  return {
      'text': text,
      'options': options,
      'label': label,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_options': glm_options,
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


def _filter_yelp_polarity_reviews(dataset):

  def my_fn(example):
    return 'text' in example and 'label' in example

  return dataset.filter(my_fn)


TASK_CONFIGS['yelp_polarity_reviews'] = TaskConfig(
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
        prep.format_options,
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


TASK_CONFIGS['cosmos_qa'] = TaskConfig(
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
        prep.format_options,
    ],
    postprocess_fn=None,
    # Metric function from nlp/unicorn/rainbow/ext5/qa_tasks.py
    metric_fns=[t5_metrics.accuracy],
)

# ============================== para_crawl enes ===============================
PARACRAWL_SPLITS_DICT = {
    'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
    'validation': f'train[-{1000+NUM_VAL_EXAMPLES}:-1000]',
    'test': 'train[-1000:]',
}


@seqio.map_over_dataset
def _process_para_crawl_enes(example):
  return {
      'lang1': 'English',
      'lang2': 'Spanish',
      'sent1': example['en'],
      'sent2': example['es'],
  }


pc_prep_fn = functools.partial(prep.add_source_info,
    task_name="para_crawl_enes", task_source="Flan2021")
TASK_CONFIGS['para_crawl_enes'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='para_crawl/enes:1.2.0', splits=PARACRAWL_SPLITS_DICT),
    preprocessors=[
        pc_prep_fn,
        _process_para_crawl_enes,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.bleu],
)

# ============================== wmt14 enfr ====================================
WMT16_SPLITS_DICT = {
    'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
    'validation': f'validation[:{NUM_VAL_EXAMPLES}]',
    'test': 'test',
}


@seqio.map_over_dataset
def _process_wmt14_translate_enfr(example):
  return {
      'lang1': 'English',
      'lang2': 'French',
      'sent1': example['en'],
      'sent2': example['fr'],
  }


TASK_CONFIGS['wmt14_enfr'] = TaskConfig(
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

  TASK_CONFIGS[f'wmt16_translate_{l2}{l1}'] = TaskConfig(
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


TASK_CONFIGS['common_gen'] = TaskConfig(
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
    metric_fns=[rouge_fn],
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


TASK_CONFIGS['dart'] = TaskConfig(
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
    metric_fns=[rouge_fn],
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


TASK_CONFIGS['e2e_nlg'] = TaskConfig(
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
    metric_fns=[rouge_fn],
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


TASK_CONFIGS['web_nlg_en'] = TaskConfig(
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
    metric_fns=[rouge_fn],
)


# ========================== wiki_lingua_english_en ============================
@seqio.map_over_dataset
def _process_wiki_lingua_english_en(example):
  return {
      'source': example['source'],
      'target': example['target'],
  }


TASK_CONFIGS['wiki_lingua_english_en'] = TaskConfig(
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

tc_prep_fn = functools.partial(prep.add_source_info,
    task_name="true_case", task_source="Flan2021")
true_case_val_end = NUM_TRAIN_EXAMPLES + NUM_VAL_EXAMPLES
TASK_CONFIGS['true_case'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='para_crawl/enda:1.2.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[{NUM_TRAIN_EXAMPLES}:{true_case_val_end}]',
            'test': 'train[-3000:]',
        }),
    preprocessors=[
        tc_prep_fn,
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

fp_prep_fn = functools.partial(prep.add_source_info,
    task_name="fix_punct", task_source="Flan2021")
fix_punct_train_end = true_case_val_end + NUM_TRAIN_EXAMPLES
fix_punct_val_end = fix_punct_train_end + NUM_VAL_EXAMPLES
TASK_CONFIGS['fix_punct'] = TaskConfig(
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

ws_prep_fn = functools.partial(prep.add_source_info,
    task_name="word_segment", task_source="Flan2021")
w_seg_train_end = fix_punct_val_end + NUM_TRAIN_EXAMPLES
w_seg_val_end = w_seg_train_end + NUM_VAL_EXAMPLES
TASK_CONFIGS['word_segment'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='para_crawl/enda:1.2.0',
        splits={
            'train': f'train[{fix_punct_val_end}:{w_seg_train_end}]',
            'validation': f'train[{w_seg_train_end}:{w_seg_val_end}]',
            'test': 'train[-3000:]',
        }),
    preprocessors=[
        ws_prep_fn,
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
cb_source_args = {
    'tfds_name': 'super_glue/cb:1.0.2',
    'splits': {
        'train': f'train[:-{NUM_VAL_EXAMPLES_CB}]',
        'validation': f'train[-{NUM_VAL_EXAMPLES_CB}:]',
        'test': 'validation',
    }
}
TASK_CONFIGS['cb'] = TaskConfig(
    source=seqio.TfdsDataSource(**cb_source_args),
    preprocessors=[
        _process_cb,
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_super_glue_metric('cb'),
    source_args=cb_source_args)


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


TASK_CONFIGS['cola'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/cola:2.0.0',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_cola,
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],  # TODO(jasonwei) matthews_corrcoef
)


# ================================ SST2 ========================================
@seqio.map_over_dataset
def _process_sst2(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['negative', 'positive'])
  glm_options = tf.constant(
      ['negative movie review.', 'positive movie review.'])
  return {
      'sentence': example['sentence'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_options': glm_options,
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


TASK_CONFIGS['sst2'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/sst2:2.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_sst2,
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_glue_metric('sst2'),
)


# ================================ MNLI ========================================
@seqio.map_over_dataset
def _process_mnli(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 3)
  options = tf.constant(['yes', 'it is not possible to tell', 'no'])
  glm_options = tf.constant(['true', 'not possible to tell', 'false'])
  return {
      'premise': example['premise'],
      'hypothesis': example['hypothesis'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_options': glm_options,
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


TASK_CONFIGS['mnli_matched'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/mnli:2.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation_matched',
        }),
    preprocessors=[
        _process_mnli,
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_glue_metric('mnli'),
)

TASK_CONFIGS['mnli_mismatched'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/mnli:2.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation_mismatched',
        }),
    preprocessors=[
        _process_mnli,
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_glue_metric('mnli'),
)


# ================================ QNLI ========================================
@seqio.map_over_dataset
def _process_qnli(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['yes', 'no'])
  glm_options_left = tf.constant(['[correct]', '[incorrect]'])
  glm_options_right = tf.fill(
      tf.shape(glm_options_left), value=example['sentence'])
  glm_options = tf.strings.join([glm_options_left, glm_options_right],
                                separator=' ')
  return {
      'sentence': example['sentence'],
      'question': example['question'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_options': glm_options,
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


TASK_CONFIGS['qnli'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/qnli:2.0.0',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_qnli,
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_glue_metric('qnli'),
)


# ================================ WNLI ========================================
@seqio.map_over_dataset
def _process_wnli(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['no', 'yes'])
  glm_options = tf.constant(['false', 'true'])
  return {
      'sentence1': example['sentence1'],
      'sentence2': example['sentence2'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_options': glm_options,
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


TASK_CONFIGS['wnli'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/wnli:2.0.0',
        splits={
            'train': 'train[:-30]',
            'validation': 'train[-30:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_wnli,
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_glue_metric('wnli'),
)


# ================================ SNLI ========================================
@seqio.map_over_dataset
def _process_snli(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 3)
  options = tf.constant(['yes', 'it is not possible to tell', 'no'])
  glm_options = tf.constant(['true', 'not possible to tell', 'false'])
  return {
      'premise': example['premise'],
      'hypothesis': example['hypothesis'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_options': glm_options,
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


def _filter_snli(dataset):

  def my_fn(example):
    return tf.math.reduce_any([
        tf.math.equal(example['label'], 0),
        tf.math.equal(example['label'], 1),
        tf.math.equal(example['label'], 2)
    ])

  return dataset.filter(my_fn)


TASK_CONFIGS['snli'] = TaskConfig(
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
        prep.format_options,
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


TASK_CONFIGS['trec'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='trec:1.0.0',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'test',
        }),
    preprocessors=[
        _process_trec,
        prep.format_options,
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
      'answer': tf.strings.as_string(example['label'], precision=0),
      'answer_int': example['label'],
  }


TASK_CONFIGS['stsb'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='glue/stsb:2.0.0',
        splits={
            'train': 'train[:-100]',
            'validation': 'train[-100:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_stsb,
        prep.format_options,
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


TASK_CONFIGS['piqa'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='piqa:1.0.0',
        splits={
            'train': 'train[:-100]',
            'validation': 'train[-100:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_piqa,
        prep.format_options,
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


TASK_CONFIGS['openbookqa'] = TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='openbookqa:0.1.0',
        splits={
            'train': 'train',
            'validation': f'validation[:{NUM_VAL_EXAMPLES}]',
            'test': 'test',
        }),
    preprocessors=[
        _process_openbookqa,
        prep.format_options,
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


TASK_CONFIGS['hellaswag'] = TaskConfig(
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
        prep.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)
