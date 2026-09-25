"""Occlusion-noise generation utilities."""

import numpy as np


def _positive_integer(value, name):
    """Return ``value`` as an int, rejecting booleans and invalid integers."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _nonnegative_integer(value, name):
    """Return ``value`` as an int, rejecting booleans and negative values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_image_shape(img_shape):
    try:
        dimensions = tuple(img_shape)
    except TypeError as exc:
        raise TypeError("img_shape must contain height and width") from exc
    if len(dimensions) != 2:
        raise ValueError("img_shape must contain exactly (height, width)")
    return (
        _positive_integer(dimensions[0], "image height"),
        _positive_integer(dimensions[1], "image width"),
    )


def _validate_fill_value(fill_value, dtype):
    value = np.asarray(fill_value)
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.number):
        raise TypeError("fill_value must be a numeric scalar")
    if np.iscomplexobj(value) or not np.isfinite(value):
        raise ValueError("fill_value must be a finite real number")

    scalar = value.item()
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        if scalar != int(scalar) or not limits.min <= scalar <= limits.max:
            raise ValueError(f"fill_value cannot be represented by {dtype}")
    elif np.issubdtype(dtype, np.floating):
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.asarray(scalar, dtype=dtype)
        if not np.isfinite(converted):
            raise ValueError(f"fill_value cannot be represented by {dtype}")
    return scalar


def _distributed_starts(length, size, count, rng):
    """Create ``count`` non-overlapping interval starts with random gaps."""
    spare = length - count * size
    if spare == 0:
        gaps = np.zeros(count + 1, dtype=int)
    else:
        gaps = rng.multinomial(spare, np.full(count + 1, 1.0 / (count + 1)))

    starts = []
    position = int(gaps[0])
    for index in range(count):
        starts.append(position)
        position += size + int(gaps[index + 1])
    return starts


def _non_overlapping_positions(height, width, size, count, rng):
    """Generate a guaranteed non-overlapping random block layout."""
    row_count = height // size
    column_count = width // size
    capacity = row_count * column_count
    if count > capacity:
        raise ValueError(
            f"Cannot place {count} non-overlapping {size}x{size} blocks in "
            f"an image of shape ({height}, {width}); maximum supported is "
            f"{capacity}. Reduce num_blocks or set allow_overlap=True."
        )

    row_starts = _distributed_starts(height, size, row_count, rng)
    column_starts = _distributed_starts(width, size, column_count, rng)
    positions = np.asarray(
        [(top, left) for top in row_starts for left in column_starts],
        dtype=int,
    )
    rng.shuffle(positions)
    return positions[:count]


def add_occlusion_noise(
    V,
    img_shape,
    block_size,
    num_blocks=1,
    fill_value=255.0,
    allow_overlap=True,
    random_state=None,
    return_mask=False,
):
    """Add square occlusion blocks independently to every image in ``V``.

    Args:
        V: Two-dimensional data matrix with one flattened image per column.
        img_shape: Image dimensions as ``(height, width)``.
        block_size: Edge length of each square block.
        num_blocks: Number of blocks placed on each image.
        fill_value: Pixel value written inside each block.
        allow_overlap: Whether blocks on the same image may overlap.
        random_state: Seed or NumPy random generator.
        return_mask: If true, also return a Boolean matrix marking every
            pixel covered by an occlusion block.

    Returns:
        A corrupted copy of ``V``. If ``return_mask`` is true, returns
        ``(V_noisy, mask)`` instead. The mask has the same shape as ``V``.
    """
    values = np.asarray(V)
    if values.ndim != 2:
        raise ValueError("V must be a two-dimensional (pixels, images) matrix")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError("V must contain real numeric values")
    if not np.all(np.isfinite(values)):
        raise ValueError("V must contain only finite values")

    height, width = _validate_image_shape(img_shape)
    size = _positive_integer(block_size, "block_size")
    count = _nonnegative_integer(num_blocks, "num_blocks")
    if size > height or size > width:
        raise ValueError(
            f"block_size={size} does not fit image ({height}, {width})"
        )
    if not isinstance(allow_overlap, (bool, np.bool_)):
        raise TypeError("allow_overlap must be a Boolean")
    if not isinstance(return_mask, (bool, np.bool_)):
        raise TypeError("return_mask must be a Boolean")
    if values.shape[0] != height * width:
        raise ValueError(
            f"V has {values.shape[0]} rows but img_shape ({height}, {width}) "
            f"implies {height * width}"
        )

    fill = _validate_fill_value(fill_value, values.dtype)
    try:
        rng = np.random.default_rng(random_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("random_state is not a valid NumPy random seed") from exc

    noisy = values.copy()
    mask = np.zeros(values.shape, dtype=bool) if return_mask else None
    for image_index in range(values.shape[1]):
        if allow_overlap:
            tops = rng.integers(0, height - size + 1, size=count)
            lefts = rng.integers(0, width - size + 1, size=count)
            positions = zip(tops, lefts)
        else:
            positions = _non_overlapping_positions(
                height, width, size, count, rng
            )

        image = noisy[:, image_index].reshape(height, width)
        image_mask = (
            mask[:, image_index].reshape(height, width)
            if mask is not None
            else None
        )
        for top, left in positions:
            top = int(top)
            left = int(left)
            image[top:top + size, left:left + size] = fill
            if image_mask is not None:
                image_mask[top:top + size, left:left + size] = True

    if return_mask:
        return noisy, mask
    return noisy


if __name__ == "__main__":
    image_height, image_width = 20, 20
    clean = np.zeros((image_height * image_width, 3))
    corrupted = add_occlusion_noise(
        clean,
        (image_height, image_width),
        block_size=5,
        num_blocks=2,
        random_state=0,
    )
    print("corrupted pixels per image:", (corrupted == 255.0).sum(axis=0))
