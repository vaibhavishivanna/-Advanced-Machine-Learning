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
                tolerance = 1e-10 * np.maximum(1.0, np.abs(losses[:-1]))
                self.assertTrue((np.diff(losses) <= tolerance).all())
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

    def test_tiny_positive_scale_is_preserved(self):
        V = np.full((4, 3), 1e-20)
        for algorithm in (standard_nmf, robust_l1_nmf):
            W, H, _ = algorithm(V, rank=1, max_iter=10)
            relative_error = np.linalg.norm(V - W @ H) / np.linalg.norm(V)
            self.assertLess(relative_error, 1e-12)

    def test_reproducible_seed(self):
        V = np.arange(1, 21).reshape(5, 4) / 20
        for algorithm in (standard_nmf, robust_l1_nmf):
            first = algorithm(V, 2, max_iter=10, random_state=4)
            second = algorithm(V, 2, max_iter=10, random_state=4)
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a, b)

    def test_corrupted_input_remains_stable(self):
        rng = np.random.default_rng(12)
        clean = rng.random((30, 4)) @ rng.random((4, 24))
        clean /= clean.max()
        corrupted = clean.copy()
        corrupted[5:15, ::3] = 1.0
        original = corrupted.copy()

        for algorithm in (standard_nmf, robust_l1_nmf):
            W, H, losses = algorithm(
                corrupted,
                rank=4,
                max_iter=80,
                random_state=3,
            )
            self.assertTrue(np.isfinite(W).all() and np.isfinite(H).all())
            self.assertTrue((W >= 0).all() and (H >= 0).all())
            self.assertTrue(np.isfinite(losses).all())
            self.assertLess(losses[-1], losses[0])
            np.testing.assert_array_equal(corrupted, original)

    def test_optional_early_stopping(self):
        V = np.arange(1, 9)[:, None] @ np.arange(1, 7)[None, :] / 48
        for algorithm in (standard_nmf, robust_l1_nmf):
            W, H, losses = algorithm(
                V,
                rank=1,
                max_iter=500,
                tol=1e-5,
                min_iter=5,
                patience=3,
            )
            self.assertGreaterEqual(len(losses) - 1, 5)
            self.assertLess(len(losses) - 1, 500)
            error = V - W @ H
            expected = (
                0.5 * np.sum(error ** 2)
                if algorithm == standard_nmf
                else np.hypot(error, 1e-3).sum()
            )
            self.assertAlmostEqual(losses[-1], expected)

    def test_robust_method_reduces_sparse_outlier_effect(self):
        rng = np.random.default_rng(123)
        clean = rng.random((30, 3)) @ rng.random((3, 24))
        clean *= 0.25 / clean.max()
        corrupted = clean.copy()
        corrupted[[2, 9, 17, 25], [1, 6, 12, 20]] = 1.0

        standard_W, standard_H, _ = standard_nmf(
            corrupted,
            rank=3,
            max_iter=50,
            random_state=4,
        )
        robust_W, robust_H, _ = robust_l1_nmf(
            corrupted,
            rank=3,
            max_iter=50,
            random_state=4,
        )
        standard_error = np.linalg.norm(clean - standard_W @ standard_H)
        robust_error = np.linalg.norm(clean - robust_W @ robust_H)
        self.assertLess(robust_error, 0.5 * standard_error)

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
                algorithm(np.ones((3, 4)), 2, tol=0)
            with self.assertRaises(ValueError):
                algorithm(np.ones((3, 4)), 2, min_iter=-1)
            with self.assertRaises(ValueError):
                algorithm(np.ones((3, 4)), 2, patience=0)
        with self.assertRaises(ValueError):
            robust_l1_nmf(np.ones((3, 4)), 2, smoothing=0)


if __name__ == '__main__':
    unittest.main()
