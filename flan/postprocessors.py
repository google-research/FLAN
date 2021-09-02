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

"""Seqio postprocessors for FLAN evaluations."""

import tensorflow.compat.v1 as tf


def remove_leading_quotes_and_spaces(s):
  if len(s) < 1:
    return s
  s_copy = str(s)
  while s_copy and not s_copy[0].isalpha():
    s_copy = s_copy[1:]
  return s_copy


def parse_glm_qa_answer(answer, example=None, is_target=False):
  """Returns answer, or a dict with answers and context if the example is provided."""

  if is_target:
    if 'answers' in example:
      return [tf.compat.as_text(a) for a in example['answers']]
    return answer

  # Take the answer up until the next question.
  # clean_answer = answer.split('\n')[0]  # Answer should be one line.
  clean_answer = answer.split('Q:')[0]  # trivia_qa, translation, etc
  clean_answer = clean_answer.split('Concepts:')[0]  # common_gen
  clean_answer = clean_answer.split('Data:')[0]  # dart, e2e_nlg, web_nlg_en

  return remove_leading_quotes_and_spaces(clean_answer)
