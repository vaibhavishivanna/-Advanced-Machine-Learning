import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments.data_utils import (
    load_faces,
    load_yaleb,
    normalize_unit_interval,
    random_subsample,
)


def _save_grayscale(path, shape, value):
    pixels = np.full(shape, value, dtype=np.uint8)
    Image.fromarray(pixels).save(path)


class TestDataLoading(unittest.TestCase):
    def test_load_faces_resizes_skips_ambient_and_labels_consecutively(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "a_empty"))
            first_subject = os.path.join(root, "b_subject")
            second_subject = os.path.join(root, "c_subject")
            os.makedirs(first_subject)
            os.makedirs(second_subject)

            _save_grayscale(os.path.join(first_subject, "1.pgm"), (6, 4), 10)
            _save_grayscale(os.path.join(first_subject, "2.PGM"), (6, 4), 20)
            _save_grayscale(os.path.join(second_subject, "1.pgm"), (6, 4), 30)
            _save_grayscale(
                os.path.join(second_subject, "subject_Ambient.pgm"),
                (6, 4),
                250,
            )
            notes_path = os.path.join(second_subject, "notes.txt")
            with open(notes_path, "w", encoding="utf-8") as handle:
                handle.write("ignored")

            matrix, labels, image_shape = load_faces(root, reduce=2)

        self.assertEqual(matrix.shape, (6, 3))
        self.assertEqual(image_shape, (3, 2))
        np.testing.assert_array_equal(labels, [0, 0, 1])
        np.testing.assert_array_equal(matrix[0], [10, 20, 30])

    def test_yaleb_loader_excludes_ambient_image(self):
        with tempfile.TemporaryDirectory() as root:
            subject = os.path.join(root, "yaleB01")
            os.makedirs(subject)
            _save_grayscale(os.path.join(subject, "normal.pgm"), (8, 8), 40)
            _save_grayscale(
                os.path.join(subject, "yaleB01_Ambient.pgm"), (8, 8), 200
            )
            matrix, labels, image_shape = load_yaleb(root, reduce=4)

        self.assertEqual(matrix.shape, (4, 1))
        self.assertEqual(image_shape, (2, 2))
        np.testing.assert_array_equal(labels, [0])

    def test_inconsistent_image_sizes_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            subject = os.path.join(root, "subject")
            os.makedirs(subject)
            _save_grayscale(os.path.join(subject, "1.pgm"), (6, 4), 10)
            _save_grayscale(os.path.join(subject, "2.pgm"), (7, 4), 20)
            with self.assertRaisesRegex(
                ValueError, "All images must have the same size"
            ):
                load_faces(root)

    def test_missing_or_empty_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            missing = os.path.join(root, "missing")
            with self.assertRaises(FileNotFoundError):
                load_faces(missing)

            empty_subject = os.path.join(root, "empty_subject")
            os.makedirs(empty_subject)
            with self.assertRaisesRegex(FileNotFoundError, "No matching image"):
                load_faces(root)

    def test_invalid_reduction_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "subject"))
            invalid_reductions = (
                (0, ValueError),
                (-1, ValueError),
                (1.5, TypeError),
            )
            for reduction, error in invalid_reductions:
                with self.subTest(reduction=reduction):
                    with self.assertRaises(error):
                        load_faces(root, reduce=reduction)

    def test_empty_skip_suffixes_are_supported(self):
        with tempfile.TemporaryDirectory() as root:
            subject = os.path.join(root, "subject")
            os.makedirs(subject)
            _save_grayscale(os.path.join(subject, "image.pgm"), (3, 3), 10)
            matrix, labels, image_shape = load_faces(root, skip_suffixes=())

        self.assertEqual(matrix.shape, (9, 1))
        self.assertEqual(image_shape, (3, 3))
        np.testing.assert_array_equal(labels, [0])


class TestDataPreprocessing(unittest.TestCase):
    def test_normalization(self):
        values = np.array([[0, 255], [64, 128]], dtype=np.uint8)
        normalized = normalize_unit_interval(values)
        np.testing.assert_allclose(normalized, values.astype(float) / 255.0)
        self.assertEqual(normalized.dtype, np.float64)

    def test_normalization_validation(self):
        invalid = (
            np.array([0, 1]),
            np.array([[0, 256]]),
            np.array([[-1, 0]]),
            np.array([[np.nan]]),
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    normalize_unit_interval(values)

    def test_subsampling_is_reproducible_and_aligned(self):
        values = np.arange(30).reshape(3, 10)
        labels = np.arange(10)
        first_values, first_labels = random_subsample(
            values, labels, frac=0.6, random_state=12
        )
        second_values, second_labels = random_subsample(
            values, labels, frac=0.6, random_state=12
        )

        np.testing.assert_array_equal(first_values, second_values)
        np.testing.assert_array_equal(first_labels, second_labels)
        self.assertEqual(first_values.shape, (3, 6))
        np.testing.assert_array_equal(first_values[0], first_labels)

    def test_subsampling_validation(self):
        values = np.zeros((3, 4))
        labels = np.arange(4)
        cases = (
            ((values[:, :3], labels), ValueError),
            ((values, labels.reshape(2, 2)), ValueError),
            ((values, labels, 0), ValueError),
            ((values, labels, 1.1), ValueError),
            ((values, labels, np.nan), ValueError),
            ((values, labels, "0.5"), TypeError),
        )
        for arguments, error in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(error):
                    random_subsample(*arguments)


if __name__ == "__main__":
    unittest.main()
