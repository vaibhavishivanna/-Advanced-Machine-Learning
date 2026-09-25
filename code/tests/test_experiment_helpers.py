import unittest

import numpy as np

from run_experiments import build_conditions, derived_seed, sample_column_indices


class TestExperimentHelpers(unittest.TestCase):
    def test_conditions_are_unique_and_share_intersection(self):
        conditions = build_conditions([5, 10, 15], 1, [1, 2, 3], 10)
        pairs = [(condition.block_size, condition.num_blocks) for condition in conditions]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertIn((0, 0), pairs)
        intersection = next(
            condition for condition in conditions
            if (condition.block_size, condition.num_blocks) == (10, 1)
        )
        self.assertTrue(intersection.include_size_sweep)
        self.assertTrue(intersection.include_count_sweep)

    def test_stratified_sampling_is_reproducible(self):
        labels = np.repeat(np.arange(4), 10)
        first = sample_column_indices(labels, 0.9, random_state=7, stratified=True)
        second = sample_column_indices(labels, 0.9, random_state=7, stratified=True)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 36)
        counts = np.bincount(labels[first])
        np.testing.assert_array_equal(counts, np.repeat(9, 4))

    def test_stratified_sampling_uses_exact_requested_total(self):
        labels = np.repeat(np.arange(38), [64] * 37 + [46])
        indices = sample_column_indices(labels, 0.9, random_state=11, stratified=True)
        self.assertEqual(len(labels), 2414)
        self.assertEqual(len(indices), round(0.9 * len(labels)))
        self.assertEqual(len(np.unique(indices)), len(indices))
        self.assertEqual(len(np.unique(labels[indices])), 38)

    def test_sampling_validation(self):
        labels = np.arange(5)
        for fraction in (0, -0.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                sample_column_indices(labels, fraction, random_state=0)

    def test_derived_seeds_are_stable_and_distinct(self):
        self.assertEqual(
            derived_seed(10, "ORL", "noise", 1, 10, 3),
            derived_seed(10, "ORL", "noise", 1, 10, 3),
        )
        self.assertNotEqual(
            derived_seed(10, "ORL", "noise", 1, 10, 3),
            derived_seed(10, "YaleB", "noise", 1, 10, 3),
        )


if __name__ == "__main__":
    unittest.main()
