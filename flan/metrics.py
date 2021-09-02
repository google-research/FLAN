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

"""Functions for computing metrics."""

from typing import Dict, List
from t5.evaluation import metrics
import tqdm


# Multi-rouge/multi-bleu. When there are multiple references, we want to get the
# rouge score that is highest. According to the authors, this is how it was done
# in the GEM paper.
# Source: https://github.com/google/BIG-bench/blob/main/bigbench/api/task_metrics.py
def rouge_fn(targets: List[List[str]],
             predictions: List[str]) -> Dict[str, float]:
  """Computes ROUGE by taking the max ROUGE-N per reference and N."""
  # Following strategy from https://www.aclweb.org/anthology/W04-1013/.
  # Identify best reference per response and ROUGE type.
  rouge_types = ["rouge1", "rouge2", "rougeLsum"]
  max_references = {rouge_type: [] for rouge_type in rouge_types}
  for targ_for_resp, resp in tqdm.tqdm(
      zip(targets, predictions), total=len(targets)):
    # Compute individual scores per example/ref pair.
    resp_scores = [metrics.rouge([t], [resp]) for t in targ_for_resp]
    # Find best scoring references for generated output and ROUGE type.
    for rouge_type in rouge_types:
      best_score_index = max(
          range(len(resp_scores)), key=lambda x: resp_scores[x][rouge_type])
      best_ref = targ_for_resp[best_score_index]
      # Add the reference to the new reference list.
      max_references[rouge_type].append(best_ref)
  # Compute metric for each of the reference lists for a ref type.
  results = {}
  for rouge_type in rouge_types:
    results[rouge_type] = metrics.rouge(max_references[rouge_type],
                                        predictions)[rouge_type]
  return results
