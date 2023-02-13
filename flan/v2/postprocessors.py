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

"""Seqio postprocessors."""
import re
from typing import Any, Optional, Mapping

import t5


def take_first_word(answer: str,
                    example: Optional[Mapping[str, Any]] = None,
                    is_target: bool = False,
                    lower_case: bool = False) -> str:
  """Take the first word in prediction."""
  del example
  if is_target:
    return answer
  answer = answer.strip()
  answer = answer.split("\n")[0].split(" ")[0]
  if answer and answer[-1] in [";", ".", ","]:
    answer = answer[:-1]
  if lower_case:
    answer = answer.lower()
  return answer


def take_first_line(answer: str,
                    example: Optional[Mapping[str, Any]] = None,
                    is_target: bool = False) -> str:
  """Take the first line in prediction."""
  del example
  if is_target:
    return answer
  answer = answer.strip().split("\n")[0]
  return answer


def take_first_paragraph(answer: str,
                         example=None,
                         is_target: bool = False,
                         remove: Optional[str] = None,
                         termination: str = "\n\n\n") -> str:
  """Take the first paragraph in prediction."""
  del example
  if is_target:
    return answer
  if remove is not None:
    answer = answer.replace(remove, "")
  answer = answer.lstrip().split(termination)[0]
  return answer


def take_last_delimited_number(answer: str,
                               example: Optional[Mapping[str, Any]] = None,
                               is_target: bool = False,
                               output_delim: str = ",",
                               text_before_output: str = "correct order "):
  """Take the first delimited number after the `text_before_output` string ."""
  del example
  if is_target:
    return answer

  if output_delim == ",":
    digit_regex = r"([0-9]+(,[0-9]+)+)"
  elif output_delim == " ":
    digit_regex = r"([0-9]+(\s[0-9]+)+)"
  elif not output_delim:
    digit_regex = r"([0-9]+)"
  else:
    raise ValueError(f"Unknown delimiter: {output_delim}")

  # Match the first delimited number that comes after `text_before_output`
  match_expression = text_before_output + digit_regex
  matches = re.search(match_expression, answer)

  if matches:
    answer = str(matches.groups()[0])
  else:
    answer = "null"

  if output_delim:
    # Delete delimiter if there is
    answer = answer.replace(output_delim, "").strip()

  # Delete leading 0 in the prediction
  answer = answer.lstrip("0")

  return answer


def trivia_qa(answer: str, **kwargs) -> str:
  """Returns answer, or all answers if the full example is provided."""
  if isinstance(answer, str):
    answer = answer.split("\n")[0]
  return t5.data.postprocessors.qa(answer, **kwargs)


def strip_after_separator(answer, separator=" ", **unused_kwargs):
  if isinstance(answer, str):
    answer = answer.split(separator)[0]
  return answer


def lambada_0shot(answer, **unused_kwargs):
  answer = strip_after_separator(answer, separator=" ")
  answer = strip_after_separator(answer, separator=".")
  answer = strip_after_separator(answer, separator=",")
  return strip_after_separator(answer, separator="?")


def take_cot_answer(answer: str,
                    example: Optional[Mapping[str, Any]] = None,
                    is_target: bool = False,
                    **unused_kwargs):
  """Postprocess target and answer containing china-of-thought process.

  Args:
    answer: text string from model output.
    example: input example.
    is_target: whether apply postprocess function on target.

  Returns:
    output: answer removing chain-of-thought texts.
  """
  del example
  answer = answer.strip()
  # corner case 1: unified_qa_science target always has a period at end.
  if answer[-1] in [".", ",", "?", " ", "\n"]:
    answer = answer[:-1].strip()

  if is_target:
    # corner case 2: target = (B), prediction = B.
    if answer[0] == "(" and answer[-1] == ")":
      answer = answer[1:-1].strip()
    return answer
  else:
    answer = answer.split("answer is")[-1].strip()
    answer = answer.split("final answer")[-1].strip()
    answer = answer.split("Final answer")[-1].strip()
    answer = answer.split("answer:")[-1].strip()
    answer = answer.split("Answer:")[-1].strip()
    if answer and answer[0] in [".", ",", "?", " ", "\n", ":"]:
      answer = answer[1:].strip()
    if answer and answer[-1] in [".", ",", "?", " ", "\n", ":"]:
      answer = answer[:-1].strip()
    # corner case 2: target = (B), prediction = B.
    if answer and answer[0] == "(" and answer[-1] == ")":
      answer = answer[1:-1].strip()
    # TODO(yunxuanli) corner case 3: target = (B), prediction = yes (option B)
    return answer


def take_bbsh_cot_answer(answer: str,
                         example: Optional[Mapping[str, Any]] = None,
                         is_target: bool = False,
                         **unused_kwargs):
  """Postprocess bbsh answer containing china-of-thought process.

  Args:
    answer: text string from model output.
    example: input example.
    is_target: whether apply postprocess function on target.

  Returns:
    output: answer removing chain-of-thought texts.
  """
  del example
  answer = answer.strip()
  # corner case 1: target has a period at end.
  if answer and answer[-1] in [".", ",", "?", " ", "\n"]:
    answer = answer[:-1].strip()

  if is_target:
    return answer
  else:
    answer = answer.split("answer is")[-1].strip()
    answer = answer.split("final answer")[-1].strip()
    answer = answer.split("Final answer")[-1].strip()
    answer = answer.split("answer:")[-1].strip()
    answer = answer.split("Answer:")[-1].strip()
    if answer and answer[0] in [".", ",", "?", " ", "\n", ":"]:
      answer = answer[1:].strip()
    if answer and answer[-1] in [".", ",", "?", " ", "\n", ":"]:
      answer = answer[:-1].strip()
    return answer
