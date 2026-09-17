"""Robust NMF using a smooth approximation to absolute reconstruction error."""
import numpy as np


def robust_l1_nmf(V, rank, max_iter=500, random_state=0, smoothing=1e-3):
    """Return W, H and loss history, with the same shapes as standard_nmf.

    Minimize sum(sqrt((V-WH)**2 + smoothing**2)). This approximates L1
    reconstruction loss; it is not L1-regularized NMF with a sparse error term.
    The default smoothing assumes pixel intensities in [0, 1].
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
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError('smoothing must be finite and positive')

    pixels, images = V.shape
    rng = np.random.default_rng(random_state)
    scale = np.sqrt(max(V.mean(), 1e-12) / rank)
    W = rng.uniform(0.5, 1.5, (pixels, rank)) * scale
    H = rng.uniform(0.5, 1.5, (rank, images)) * scale

    loss_history = [np.sum(np.hypot(V - W @ H, smoothing))]
    eps = np.finfo(float).tiny

    for _ in range(max_iter):
        residual = V - W @ H
        weights = 1.0 / np.hypot(residual, smoothing)
        weighted_V = weights * V

        numerator = W.T @ weighted_V
        denominator = W.T @ (weights * (W @ H))
        H *= numerator / np.maximum(denominator, eps)

        numerator = weighted_V @ H.T
        denominator = (weights * (W @ H)) @ H.T
        W *= numerator / np.maximum(denominator, eps)

        loss = np.sum(np.hypot(V - W @ H, smoothing))
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite loss: check the input scale')
        loss_history.append(loss)

    return W, H, np.array(loss_history)
