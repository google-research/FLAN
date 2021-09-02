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

"""Tests for utils."""

from absl.testing import absltest
from absl.testing import parameterized

from flan import utils


class FLANUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('ZeroshotEvalTaskName', utils.ZeroshotEvalTaskName,
       ('foo', 1), 'foo_type_1', ('foo', 1)),
      ('ZeroshotScoreEvalTaskName', utils.ZeroshotScoreEvalTaskName,
       ('foo', 1), 'foo_type_1_scoring_eval', ('foo', 1)),
      ('AllPromptsTaskName', utils.AllPromptsTaskName,
       ('foo',), 'foo_all_prompts', 'foo'),
      ('ZeroshotTemplatedTaskName', utils.ZeroshotTemplatedTaskName,
       ('foo', 8), 'foo_8templates', ('foo', 8)),
      ('OneshotTemplatedTaskName', utils.XshotTemplatedTaskName,
       ('foo', 5, 'one'), 'foo_5templates_one_shot', ('foo', 5, 'one')),
      ('MultishotTemplatedTaskName', utils.XshotTemplatedTaskName,
       ('foo', 5, 'multi'), 'foo_5templates_multi_shot', ('foo', 5, 'multi')),
      )
  def test_seqio_task_name_class(self, task_name_cls, args, task_name, parse):
    """Tests seqio task name classes."""
    result_task_name = task_name_cls.get(*args)
    self.assertEqual(result_task_name, task_name)
    self.assertTrue(task_name_cls.match(task_name))
    result_parse = task_name_cls.parse(task_name)
    self.assertSequenceEqual(result_parse, parse)


if __name__ == '__main__':
  absltest.main()
