"""
COMP4328/5328/8328 - Advanced Machine Learning - Assignment 1
Part: Data + Experiments (Person 2)
Module: noise.py - occlusion noise generation
"""
import numpy as np


def add_occlusion_noise(V, img_shape, block_size, num_blocks=1,
                         fill_value=255.0, allow_overlap=True,
                         random_state=None):
    """
    Corrupt each image (column) in V with `num_blocks` random
    block_size x block_size squares filled with `fill_value`.

    Args:
        V: (D, N) clean data matrix, D must equal img_shape[0]*img_shape[1].
        img_shape: (H, W).
        block_size: edge length b of each square occlusion block.
        num_blocks: number of blocks stamped onto each image.
        fill_value: pixel value written inside each block (255 = white).
        allow_overlap: if False, avoids overlapping blocks on the same
            image on a best-effort basis.
        random_state: int seed or numpy Generator, for reproducibility.

    Returns:
        V_noisy: (D, N) ndarray, a corrupted copy of V.
    """
    rng = np.random.default_rng(random_state)
    H, W = img_shape
    if V.shape[0] != H * W:
        raise ValueError(f"V has {V.shape[0]} rows but img_shape {img_shape} "
                          f"implies {H * W}")
    if block_size > H or block_size > W:
        raise ValueError(f"block_size={block_size} does not fit image {img_shape}")

    D, N = V.shape
    V_noisy = V.copy()

    for n in range(N):
        img = V_noisy[:, n].reshape(H, W)
        placed = []
        for _ in range(num_blocks):
            top, left = None, None
            for _attempt in range(20):
                cand_top = rng.integers(0, H - block_size + 1)
                cand_left = rng.integers(0, W - block_size + 1)
                if allow_overlap or not _overlaps(placed, cand_top, cand_left, block_size):
                    top, left = cand_top, cand_left
                    break
            if top is None:
                top = rng.integers(0, H - block_size + 1)
                left = rng.integers(0, W - block_size + 1)
            placed.append((top, left))
            img[top:top + block_size, left:left + block_size] = fill_value
        V_noisy[:, n] = img.reshape(-1)

    return V_noisy


def _overlaps(placed, top, left, size):
    for (t, l) in placed:
        if not (left + size <= l or l + size <= left or
                top + size <= t or t + size <= top):
            return True
    return False


if __name__ == '__main__':
    H, W = 20, 20
    V = np.zeros((H * W, 3))
    V_noisy = add_occlusion_noise(V, (H, W), block_size=5, num_blocks=2, random_state=0)
    print('corrupted pixels per image:', (V_noisy == 255.0).sum(axis=0))
