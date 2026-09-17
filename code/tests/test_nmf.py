import unittest
import numpy as np
from algorithm import standard_nmf, robust_l1_nmf


class TestNMF(unittest.TestCase):
    def test_factors_and_training_losses(self):
        rng = np.random.default_rng(42)
        V = rng.random((18, 3)) @ rng.random((3, 14))
        original = V.copy()
        for algorithm in (standard_nmf, robust_l1_nmf):
            for seed in (0, 1, 2):
                W, H, losses = algorithm(V, 3, max_iter=100, random_state=seed)
                self.assertEqual(W.shape, (18, 3))
                self.assertEqual(H.shape, (3, 14))
                self.assertTrue((W >= 0).all() and (H >= 0).all())
                self.assertTrue(np.isfinite(W @ H).all())
                self.assertEqual(len(losses), 101)
                self.assertLess(losses[-1], losses[0])
                self.assertTrue((np.diff(losses) <= 1e-10).all())
                error = V - W @ H
                expected = .5*np.sum(error**2) if algorithm == standard_nmf else np.hypot(error, 1e-3).sum()
                self.assertAlmostEqual(losses[-1], expected)
        np.testing.assert_array_equal(V, original)

    def test_simple_reconstruction(self):
        V = np.arange(1, 9)[:, None] @ np.arange(1, 7)[None, :] / 48
        for algorithm in (standard_nmf, robust_l1_nmf):
            W, H, _ = algorithm(V, 1, max_iter=500)
            np.testing.assert_allclose(W @ H, V, atol=1e-7)
            W, H, _ = algorithm(np.zeros((4, 3)), 2, max_iter=5)
            np.testing.assert_array_equal(W @ H, np.zeros((4, 3)))

    def test_reproducible_seed(self):
        V = np.arange(1, 21).reshape(5, 4) / 20
        for algorithm in (standard_nmf, robust_l1_nmf):
            first = algorithm(V, 2, max_iter=10, random_state=4)
            second = algorithm(V, 2, max_iter=10, random_state=4)
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a, b)

    def test_invalid_inputs(self):
        for algorithm in (standard_nmf, robust_l1_nmf):
            for V in ([[1, -1]], [[np.nan]], [[1j]], [], [1, 2]):
                with self.assertRaises(ValueError):
                    algorithm(V, 1)
            with self.assertRaises(ValueError):
                algorithm(np.ones((3, 4)), 5)
            with self.assertRaises(ValueError):
                algorithm(np.ones((3, 4)), 2, max_iter=0)
        with self.assertRaises(ValueError):
            robust_l1_nmf(np.ones((3, 4)), 2, smoothing=0)


if __name__ == '__main__':
    unittest.main() 