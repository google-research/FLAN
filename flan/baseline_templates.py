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

"""Templates for baseline GLM prompts."""

# pylint: disable=line-too-long

PATTERNS = {
    "anli": [
        ("{context}\nquestion: {hypothesis} Is it true, false or neither?\nanswer:", "{glm_answer}"),
    ],
    "cb": [
        ("{premise}\nquestion: {hypothesis} Is it true, false or neither?\nanswer:", "{glm_answer}"),
    ],
    "rte": [
        ("{premise}\nquestion: {hypothesis} True or false?\nanswer:", "{glm_answer}"),
    ],
    "arc": [
        ("Question: {question}\nAnswer:", "{answer}"),
    ],
    "bool_q": [
        ("{text}\nquestion: {question}?\nanswer:", "{answer}"),
    ],
    "openbookqa": [
        ("{fact}\n{question}", "{answer}"),
    ],
    "copa": [
        ("{glm_premise}", "{answer}"),
    ],
    "hellaswag": [
        ("{context}", "{answer}"),
    ],
    "piqa": [
        ("{goal}", "{answer}"),
    ],
    "story_cloze": [
        ("{context}", "{answer}"),
    ],
    "record": [
        ("{passage} {query}", "{answer}"),
    ],
    "multirc": [
        ("READING COMPREHENSION ANSWER KEY\n{paragraph}\n\n{question}", "[{glm_answer}] {response}"),
    ],
    "winogrande": [
        ("{context}", "{answer}"),
    ],
    "common_gen": [
        ("Concepts: {concepts}\n\nSentence describing concepts:", "{target}"),
    ],
    "dart": [
        ("Data: ({tripleset})\n\nDescription of data:", "{target}"),
    ],
    "e2e_nlg": [
        ("Data: ({meaning_representation})\n\nDescription of data:", "{target}"),
    ],
    "web_nlg_en": [
        ("Data: ({input_string})\n\nDescription of data:", "{target}"),
    ],
    "para_crawl": [
        ("Q: What is the {lang2} translation of {sent1}?\nA:", "{sent2}"),
        ("Q: What is the {lang1} translation of {sent2}?\nA:", "{sent1}"),
    ],
    "wmt14_enfr": [
        ("Q: What is the {lang2} translation of {sent1}?\nA:", "{sent2}"),
        ("Q: What is the {lang1} translation of {sent2}?\nA:", "{sent1}"),
    ],
    "wmt16_translate": [
        ("Q: What is the {lang2} translation of {sent1}?\nA:", "{sent2}"),
        ("Q: What is the {lang1} translation of {sent2}?\nA:", "{sent1}"),
    ],
    "natural_questions": [
        ("Q: {question}\nA:", "{answer}"),
    ],
    "trivia_qa": [
        ("Q: {question}\nA:", "{answer}"),
    ],
    "wsc273": [
        ("{context}", "{answer}"),
    ],
    # These are not true few-shot in GPT-3 because they actually include
    # examples from the same context.
    # "drop": [
    #     ("{context}\n\nQ: {question}\n\nA:", "{answer}"),
    # ],
    # "squad_v2": [
    #     ("{context}\n\nQ: {question}?\n\nA:", "{answer}"),
    # ],
}
