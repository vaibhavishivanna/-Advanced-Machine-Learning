"""Dataset loading and preprocessing utilities."""

import os

import numpy as np
from PIL import Image


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _suffix_tuple(value, name, allow_empty=False):
    if value is None:
        if allow_empty:
            return ()
        raise ValueError(f"{name} must contain at least one suffix")
    if isinstance(value, str):
        suffixes = (value,)
    else:
        try:
            suffixes = tuple(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a string or sequence of strings") from exc
    if not suffixes:
        if allow_empty:
            return ()
        raise ValueError(f"{name} must contain at least one suffix")
    if any(not isinstance(suffix, str) or not suffix for suffix in suffixes):
        raise ValueError(f"{name} must contain non-empty strings")
    return tuple(suffix.lower() for suffix in suffixes)


def load_faces(root, reduce=1, skip_suffixes=("Ambient.pgm",), valid_ext=".pgm"):
    """Load subject-folder face images into a pixels-by-images matrix.

    ``root`` must contain one directory per subject. Images are converted to
    grayscale and optionally downsampled by the integer factor ``reduce``.

    Returns:
        ``(V, Y, img_shape)`` where ``V`` contains values in ``[0, 255]``,
        ``Y`` contains consecutive subject labels, and ``img_shape`` is
        ``(height, width)`` after resizing.
    """
    factor = _positive_integer(reduce, "reduce")
    valid_suffixes = _suffix_tuple(valid_ext, "valid_ext")
    ignored_suffixes = _suffix_tuple(
        skip_suffixes, "skip_suffixes", allow_empty=True
    )

    try:
        root_path = os.fspath(root)
    except TypeError as exc:
        raise TypeError("root must be a filesystem path") from exc
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"Dataset directory does not exist: '{root_path}'")

    subjects = sorted(
        entry
        for entry in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, entry))
    )
    if not subjects:
        raise FileNotFoundError(
            f"No subject subdirectories found under '{root_path}'"
        )

    images = []
    labels = []
    image_shape = None
    next_label = 0

    for subject in subjects:
        subject_dir = os.path.join(root_path, subject)
        subject_images = []
        for filename in sorted(os.listdir(subject_dir)):
            lowercase_name = filename.lower()
            if not lowercase_name.endswith(valid_suffixes):
                continue
            if ignored_suffixes and lowercase_name.endswith(ignored_suffixes):
                continue

            image_path = os.path.join(subject_dir, filename)
            if not os.path.isfile(image_path):
                continue
            try:
                with Image.open(image_path) as source:
                    image = source.convert("L")
                    if factor > 1:
                        resized_width = image.width // factor
                        resized_height = image.height // factor
                        if resized_width < 1 or resized_height < 1:
                            raise ValueError(
                                f"reduce={factor} makes image '{image_path}' empty"
                            )
                        image = image.resize((resized_width, resized_height))
                    pixels = np.asarray(image, dtype=np.float64)
            except OSError as exc:
                raise ValueError(f"Could not read image '{image_path}': {exc}") from exc

            current_shape = tuple(pixels.shape)
            if image_shape is None:
                image_shape = current_shape
            elif current_shape != image_shape:
                raise ValueError(
                    f"Image '{image_path}' has shape {current_shape}; expected "
                    f"{image_shape}. All images must have the same size."
                )
            subject_images.append(pixels.reshape(-1))

        if subject_images:
            images.extend(subject_images)
            labels.extend([next_label] * len(subject_images))
            next_label += 1

    if not images:
        raise FileNotFoundError(
            f"No matching image files found under '{root_path}'"
        )

    return (
        np.column_stack(images),
        np.asarray(labels, dtype=int),
        image_shape,
    )


def load_orl(root="data/ORL", reduce=3):
    """Load ORL images, using a default downsampling factor of three."""
    return load_faces(root, reduce=reduce)


def load_yaleb(root="data/CroppedYaleB", reduce=4):
    """Load Extended YaleB images while excluding ambient calibration files."""
    return load_faces(root, reduce=reduce, skip_suffixes=("Ambient.pgm",))


def normalize_unit_interval(V):
    """Rescale a finite image matrix from ``[0, 255]`` to ``[0, 1]``."""
    values = np.asarray(V)
    if values.ndim != 2:
        raise ValueError("V must be a two-dimensional (pixels, images) matrix")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError("V must contain real numeric values")
    if not np.all(np.isfinite(values)):
        raise ValueError("V must contain only finite values")
    if values.size and (np.min(values) < 0 or np.max(values) > 255):
        raise ValueError("V values must lie in the interval [0, 255]")
    return values.astype(np.float64, copy=False) / 255.0


def random_subsample(V, Y, frac=0.9, random_state=None):
    """Randomly sample a fraction of matching image columns and labels."""
    values = np.asarray(V)
    labels = np.asarray(Y)
    if values.ndim != 2:
        raise ValueError("V must be a two-dimensional (pixels, images) matrix")
    if labels.ndim != 1:
        raise ValueError("Y must be a one-dimensional label array")
    if values.shape[1] != labels.size:
        raise ValueError(
            f"V contains {values.shape[1]} images but Y contains {labels.size} labels"
        )
    if values.shape[1] == 0:
        raise ValueError("V and Y must contain at least one sample")
    if isinstance(frac, (bool, np.bool_)) or not isinstance(
        frac, (int, float, np.integer, np.floating)
    ):
        raise TypeError("frac must be a real number")
    fraction = float(frac)
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("frac must be finite and in the interval (0, 1]")

    sample_count = int(round(fraction * values.shape[1]))
    if sample_count == 0:
        raise ValueError("frac is too small to select any samples")
    try:
        rng = np.random.default_rng(random_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("random_state is not a valid NumPy random seed") from exc
    indices = rng.choice(values.shape[1], size=sample_count, replace=False)
    return values[:, indices], labels[indices]


if __name__ == "__main__":
    import sys

    dataset_root = sys.argv[1] if len(sys.argv) > 1 else "data/ORL"
    matrix, target, shape = load_faces(dataset_root, reduce=3)
    print(
        f"Loaded {dataset_root}: V.shape={matrix.shape}, Y.shape={target.shape}, "
        f"img_shape={shape}, subjects={len(set(target))}, "
        f"value range=[{matrix.min()}, {matrix.max()}]"
    )
