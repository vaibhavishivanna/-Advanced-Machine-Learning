"""
COMP4328/5328/8328 - Advanced Machine Learning - Assignment 1
Part: Data + Experiments (Person 2)
Module: data_utils.py - dataset loading and preprocessing
"""
import os
import numpy as np
from PIL import Image


def load_faces(root, reduce=1, skip_suffixes=('Ambient.pgm',), valid_ext='.pgm'):
    """
    Load a face dataset stored as <root>/<subject_folder>/<image>.pgm
    into a (pixels x images) matrix. Works for both ORL and Extended YaleB.

    Args:
        root: dataset folder, e.g. 'data/ORL' or 'data/CroppedYaleB'.
        reduce: down-sampling factor applied to width and height.
        skip_suffixes: filenames ending in any of these are skipped
            (drops YaleB's Ambient calibration frames).
        valid_ext: only files with this extension are read as images.

    Returns:
        V: ndarray (D, N) - D pixels per image, N images, values in [0, 255].
        Y: ndarray (N,) - integer subject label per image.
        img_shape: (height, width) of the resized images.
    """
    images, labels = [], []
    img_shape = None

    subjects = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    if not subjects:
        raise FileNotFoundError(f"No subject sub-folders found under '{root}'")

    for label, subject in enumerate(subjects):
        subject_dir = os.path.join(root, subject)
        for fname in sorted(os.listdir(subject_dir)):
            if not fname.endswith(valid_ext):
                continue
            if any(fname.endswith(suf) for suf in skip_suffixes):
                continue

            img = Image.open(os.path.join(subject_dir, fname)).convert('L')
            if reduce > 1:
                img = img.resize([s // reduce for s in img.size])

            if img_shape is None:
                img_shape = img.size[::-1]

            images.append(np.asarray(img, dtype=np.float64).reshape(-1, 1))
            labels.append(label)

    V = np.concatenate(images, axis=1)
    Y = np.array(labels)
    return V, Y, img_shape


def load_orl(root='data/ORL', reduce=3):
    """Load ORL (400 images, 40 subjects, native 92x112 -> 30x37)."""
    return load_faces(root, reduce=reduce)


def load_yaleb(root='data/CroppedYaleB', reduce=4):
    """Load Extended YaleB (2414 images, 38 subjects, native 168x192 -> 42x48)."""
    return load_faces(root, reduce=reduce, skip_suffixes=('Ambient.pgm',))


def normalize_unit_interval(V):
    """Rescale pixel intensities from [0, 255] to [0, 1]."""
    return V / 255.0


def random_subsample(V, Y, frac=0.9, random_state=None):
    """Randomly sample a `frac` fraction of columns (images) and labels."""
    rng = np.random.default_rng(random_state)
    n = V.shape[1]
    n_sub = int(round(frac * n))
    idx = rng.choice(n, size=n_sub, replace=False)
    return V[:, idx], Y[idx]


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'data/ORL'
    V, Y, shape = load_faces(root, reduce=3)
    print(f'Loaded {root}: V.shape={V.shape}, Y.shape={Y.shape}, '
          f'img_shape={shape}, subjects={len(set(Y))}, '
          f'value range=[{V.min()}, {V.max()}]')
