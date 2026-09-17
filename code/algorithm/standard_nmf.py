"""Standard NMF with squared reconstruction error."""
import numpy as np


def standard_nmf(V, rank, max_iter=500, random_state=0):
    """Return W, H and loss history for a nonnegative matrix V.

    V has shape (pixels, images). W has shape (pixels, rank), and
    H has shape (rank, images). Use consistently scaled input, e.g. [0, 1].
    Runs max_iter updates; this is an iteration budget, not a convergence test.
    """

    if np.iscomplexobj(V):
        raise ValueError('V must be real')
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.size == 0 or not np.isfinite(V).all() or (V < 0).any():
        raise ValueError('V must be a finite, nonnegative 2D matrix')
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= min(V.shape):
        raise ValueError('rank must be an integer between 1 and min(V.shape)')
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError('max_iter must be a positive integer')

    pixels, images = V.shape
    rng = np.random.default_rng(random_state)
    scale = np.sqrt(max(V.mean(), 1e-12) / rank)
    W = rng.uniform(0.5, 1.5, (pixels, rank)) * scale
    H = rng.uniform(0.5, 1.5, (rank, images)) * scale

    loss_history = [0.5 * np.sum((V - W @ H) ** 2)]
    eps = np.finfo(float).tiny  

    for _ in range(max_iter):
        numerator = W.T @ V
        denominator = (W.T @ W) @ H
        H *= numerator / np.maximum(denominator, eps)

        numerator = V @ H.T
        denominator = W @ (H @ H.T)
        W *= numerator / np.maximum(denominator, eps)

        loss = 0.5 * np.sum((V - W @ H) ** 2)
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite loss: check the input scale')
        loss_history.append(loss)

    return W, H, np.array(loss_history)
