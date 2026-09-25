import importlib.util
import unittest

import numpy as np

from evaluation import clustering_metrics, relative_reconstruction_error


class TestRelativeReconstructionError(unittest.TestCase):
    def test_exact_reconstruction_is_zero(self):
        W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        H = np.array([[1.0, 2.0], [3.0, 4.0]])
        clean = W @ H
        self.assertEqual(relative_reconstruction_error(clean, W, H), 0.0)

    def test_known_value(self):
        clean = np.array([[3.0, 4.0]])
        W = np.array([[1.0]])
        H = np.array([[0.0, 0.0]])
        self.assertAlmostEqual(relative_reconstruction_error(clean, W, H), 1.0)

    def test_invalid_shapes_and_values(self):
        clean = np.ones((3, 4))
        with self.assertRaises(ValueError):
            relative_reconstruction_error(clean, np.ones((2, 1)), np.ones((1, 4)))
        with self.assertRaises(ValueError):
            relative_reconstruction_error(clean, np.ones((3, 2)), np.ones((1, 4)))
        with self.assertRaises(ValueError):
            relative_reconstruction_error(clean, np.ones((3, 1)), np.ones((1, 5)))
        with self.assertRaises(ValueError):
            relative_reconstruction_error(np.zeros((2, 2)), np.ones((2, 1)), np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            relative_reconstruction_error([[np.nan]], [[1]], [[1]])


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn not installed")
class TestClusteringMetrics(unittest.TestCase):
    def test_accuracy_and_nmi_use_majority_mapped_labels(self):
        from sklearn.metrics import normalized_mutual_info_score

        H = np.array([[0.0, 0.0, 0.0, 20.0, 10.0, 20.0]])
        labels = np.array([2, 1, 1, 0, 0, 0])
        metrics = clustering_metrics(H, labels, n_clusters=3, random_state=0)
        expected_nmi = normalized_mutual_info_score(
            labels,
            metrics["predicted_labels"],
        )
        raw_cluster_nmi = normalized_mutual_info_score(
            labels,
            metrics["cluster_labels"],
        )
        self.assertAlmostEqual(metrics["accuracy"], 5 / 6)
        self.assertAlmostEqual(metrics["nmi"], expected_nmi)
        self.assertNotAlmostEqual(metrics["nmi"], raw_cluster_nmi)


if __name__ == "__main__":
    unittest.main()
