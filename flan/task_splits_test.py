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

"""Tests for task_splits."""

import collections

from absl.testing import absltest

from flan import task_splits


class TaskSplitsTest(absltest.TestCase):

  def create_task_clusters(self):
    return collections.OrderedDict([
        ('a', ['a0', 'a1', 'a2']),
        ('b', ['b0', 'b1', 'b2', 'b3', 'b4']),
        ('c', ['c0']),
    ])

  def test_intra_cluster_splits(self):
    task_clusters = self.create_task_clusters()
    splits = task_splits.generate_intra_cluster_splits(10, task_clusters)

    def cluster_counts(tasks):
      counts = collections.Counter()
      for task in tasks:
        cluster = task[0]
        counts[cluster] += 1
      return counts

    for split in splits:
      # Verify that every cluster holds out exactly one test task, unless the
      # cluster size is <= 1.
      train_cluster_counts = cluster_counts(split.train_tasks)
      test_cluster_counts = cluster_counts(split.test_tasks)

      expected_train_cluster_counts = collections.Counter({
          'a': 2, 'b': 4, 'c': 1})
      expected_test_cluster_counts = collections.Counter({
          'a': 1, 'b': 1})

      self.assertEqual(train_cluster_counts, expected_train_cluster_counts)
      self.assertEqual(test_cluster_counts, expected_test_cluster_counts)

  def test_intra_cluster_splits_task_collapse(self):
    task_clusters = self.create_task_clusters()
    task_collapse_map = {
        'b2': 'collapsed',
        'b3': 'collapsed',
        'b4': 'collapsed',
    }
    splits = task_splits.generate_intra_cluster_splits(
        task_clusters=task_clusters, task_collapse_map=task_collapse_map)

    def num_collapsed_tasks(tasks):
      return sum(1 if task in task_collapse_map else 0 for task in tasks)

    for split in splits:
      # Verify that every split keeps the clustered tasks on the same side.
      train_collapsed = num_collapsed_tasks(split.train_tasks)
      test_collapsed = num_collapsed_tasks(split.test_tasks)
      all_test = (train_collapsed == 0 and test_collapsed == 3)
      all_train = (train_collapsed == 3 and test_collapsed == 0)
      self.assertTrue(all_test or all_train)

  def test_intra_cluster_splits_holdout_coverage(self):
    task_clusters = self.create_task_clusters()
    splits = task_splits.generate_intra_cluster_splits(10, task_clusters)

    # Just look at first three splits.
    splits = splits[:3]

    # A map from a task name to a cluster name.
    task_to_cluster = {}
    for key, values in task_clusters.items():
      for val in values:
        task_to_cluster[val] = key

    # For each cluster, see which tasks have been heldout as test.
    heldout = collections.defaultdict(set)
    for split in splits:
      for task in split.test_tasks:
        heldout[task_to_cluster[task]].add(task)

    # Since cluster 'a' has 3 tasks, all 3 should be heldout.
    self.assertEqual(set(task_clusters['a']), heldout['a'])

    # cluster 'b' has 5 tasks -- we should be missing only 2 after 3 splits.
    self.assertLen(set(task_clusters['b']) - heldout['b'], 2)

    # cluster 'c' has just 1 task, so it should never be held out.
    self.assertEmpty(heldout['c'])

  def test_inter_cluster_splits(self):
    task_clusters = self.create_task_clusters()
    splits = task_splits.generate_inter_cluster_splits(10, task_clusters)

    # There should be 3 splits, one for each cluster.
    self.assertLen(splits, 3)

    for split in splits:
      test_clusters = set([task[0] for task in split.test_tasks])
      train_clusters = set([task[0] for task in split.train_tasks])
      # All test tasks should belong to the same cluster.
      self.assertLen(test_clusters, 1)
      # No overlap in clusters between train and test.
      self.assertEmpty(test_clusters & train_clusters)

  def test_deterministic(self):
    task_clusters = self.create_task_clusters()
    splits1 = task_splits.generate_inter_cluster_splits(10, task_clusters)
    splits2 = task_splits.generate_inter_cluster_splits(10, task_clusters)
    self.assertEqual(splits1, splits2)

    splits3 = task_splits.generate_intra_cluster_splits(10, task_clusters)
    splits4 = task_splits.generate_intra_cluster_splits(10, task_clusters)
    self.assertEqual(splits3, splits4)


if __name__ == '__main__':
  absltest.main()
