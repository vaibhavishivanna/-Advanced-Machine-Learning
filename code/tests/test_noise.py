import unittest

import numpy as np

from experiments.noise import add_occlusion_noise


class TestOcclusionNoise(unittest.TestCase):
    def test_strict_layout_is_non_overlapping_and_reproducible(self):
        clean = np.zeros((7 * 9, 3), dtype=np.uint8)

        first, first_mask = add_occlusion_noise(
            clean,
            (7, 9),
            block_size=2,
            num_blocks=7,
            fill_value=255,
            allow_overlap=False,
            random_state=42,
            return_mask=True,
        )
        second, second_mask = add_occlusion_noise(
            clean,
            (7, 9),
            block_size=2,
            num_blocks=7,
            fill_value=255,
            allow_overlap=False,
            random_state=42,
            return_mask=True,
        )

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_mask, second_mask)
        np.testing.assert_array_equal(first_mask.sum(axis=0), [28, 28, 28])
        np.testing.assert_array_equal(first == 255, first_mask)
        np.testing.assert_array_equal(clean, np.zeros_like(clean))

    def test_impossible_strict_layout_raises_clear_error(self):
        clean = np.zeros((5 * 5, 1), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "Cannot place 2 non-overlapping"):
            add_occlusion_noise(
                clean,
                (5, 5),
                block_size=3,
                num_blocks=2,
                allow_overlap=False,
                random_state=0,
            )

    def test_zero_blocks_returns_an_unchanged_copy_and_empty_mask(self):
        clean = np.arange(24, dtype=np.float64).reshape(12, 2)
        noisy, mask = add_occlusion_noise(
            clean,
            (3, 4),
            block_size=2,
            num_blocks=0,
            random_state=0,
            return_mask=True,
        )

        np.testing.assert_array_equal(noisy, clean)
        self.assertIsNot(noisy, clean)
        self.assertFalse(mask.any())

    def test_generator_random_state_is_supported(self):
        clean = np.zeros((16, 1), dtype=np.float64)
        noisy = add_occlusion_noise(
            clean,
            (4, 4),
            block_size=1,
            num_blocks=2,
            random_state=np.random.default_rng(9),
        )
        self.assertEqual(np.count_nonzero(noisy), 2)

    def test_invalid_inputs_are_rejected(self):
        clean = np.zeros((16, 2), dtype=np.float64)
        cases = (
            ({"V": np.zeros(16)}, ValueError),
            ({"V": np.full((16, 1), np.nan)}, ValueError),
            ({"img_shape": (4,)}, ValueError),
            ({"img_shape": (4, 5)}, ValueError),
            ({"block_size": 0}, ValueError),
            ({"block_size": 5}, ValueError),
            ({"block_size": 1.5}, TypeError),
            ({"num_blocks": -1}, ValueError),
            ({"num_blocks": 1.5}, TypeError),
            ({"allow_overlap": "no"}, TypeError),
            ({"return_mask": 1}, TypeError),
            ({"fill_value": np.inf}, ValueError),
        )
        defaults = {
            "V": clean,
            "img_shape": (4, 4),
            "block_size": 2,
            "num_blocks": 1,
            "random_state": 0,
        }
        for changes, error in cases:
            arguments = {**defaults, **changes}
            with self.subTest(changes=changes):
                with self.assertRaises(error):
                    add_occlusion_noise(**arguments)

        with self.assertRaisesRegex(ValueError, "cannot be represented"):
            add_occlusion_noise(
                clean.astype(np.float16),
                (4, 4),
                block_size=2,
                fill_value=1e100,
            )


if __name__ == "__main__":
    unittest.main()
