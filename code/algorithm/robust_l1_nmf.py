"""Robust NMF using a smooth approximation to absolute reconstruction error."""
import numpy as np


def robust_l1_nmf(
    V,
    rank,
    max_iter=500,
    random_state=0,
    smoothing=1e-3,
    *,
    tol=None,
    min_iter=20,
    patience=5,
):
    """Return W, H and loss history, with the same shapes as standard_nmf.

    Minimize sum(sqrt((V-WH)**2 + smoothing**2)). This approximates L1
    reconstruction loss; it is not L1-regularized NMF with a sparse error term.
    The default smoothing assumes pixel intensities in [0, 1].
    When ``tol`` is provided, the same relative-loss early-stopping rule as
    ``standard_nmf`` is used.
    """

    if np.iscomplexobj(V):
        raise ValueError('V must be real')
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.size == 0 or not np.isfinite(V).all() or (V < 0).any():
        raise ValueError('V must be a finite, nonnegative 2D matrix')
    if (
        isinstance(rank, bool)
        or not isinstance(rank, (int, np.integer))
        or not 1 <= rank <= min(V.shape)
    ):
        raise ValueError('rank must be an integer between 1 and min(V.shape)')
    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError('max_iter must be a positive integer')
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError('smoothing must be finite and positive')
    if tol is not None and (not np.isfinite(tol) or tol <= 0):
        raise ValueError('tol must be None or a finite positive number')
    if (
        isinstance(min_iter, bool)
        or not isinstance(min_iter, (int, np.integer))
        or min_iter < 0
    ):
        raise ValueError('min_iter must be a nonnegative integer')
    if (
        isinstance(patience, bool)
        or not isinstance(patience, (int, np.integer))
        or patience < 1
    ):
        raise ValueError('patience must be a positive integer')

    pixels, images = V.shape
    rng = np.random.default_rng(random_state)
    scale = np.sqrt(max(V.mean(), 1e-12) / rank)
    W = rng.uniform(0.5, 1.5, (pixels, rank)) * scale
    H = rng.uniform(0.5, 1.5, (rank, images)) * scale

    reconstruction = W @ H
    loss_history = [np.sum(np.hypot(V - reconstruction, smoothing))]
    update_floor = np.finfo(float).tiny
    relative_floor = np.finfo(float).eps
    stalled_updates = 0

    for iteration in range(1, max_iter + 1):
        residual = V - reconstruction
        weights = 1.0 / np.hypot(residual, smoothing)
        weighted_V = weights * V
        weighted_reconstruction = weights * reconstruction

        numerator = W.T @ weighted_V
        denominator = W.T @ weighted_reconstruction
        H *= numerator / np.maximum(denominator, update_floor)

        reconstruction = W @ H
        weighted_reconstruction = weights * reconstruction
        numerator = weighted_V @ H.T
        denominator = weighted_reconstruction @ H.T
        W *= numerator / np.maximum(denominator, update_floor)

        reconstruction = W @ H
        loss = np.sum(np.hypot(V - reconstruction, smoothing))
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite loss: check the input scale')
        loss_history.append(loss)

        if tol is not None and iteration >= min_iter:
            previous_loss = loss_history[-2]
            relative_improvement = (
                (previous_loss - loss) / max(abs(previous_loss), relative_floor)
            )
            stalled_updates = (
                stalled_updates + 1
                if 0 <= relative_improvement < tol
                else 0
            )
            if stalled_updates >= patience:
                break

    return W, H, np.array(loss_history)
