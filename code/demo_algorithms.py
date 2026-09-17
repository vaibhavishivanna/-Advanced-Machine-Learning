"""Run both algorithms on a small synthetic matrix; no dataset needed."""
import numpy as np
from algorithm import standard_nmf, robust_l1_nmf


def main():
    rng = np.random.default_rng(7)
    V = rng.uniform(.1, .8, (40, 4)) @ rng.uniform(.1, .8, (4, 25))
    V /= V.max()

    for algorithm in (standard_nmf, robust_l1_nmf):
        W, H, losses = algorithm(V, rank=4, max_iter=500, random_state=0)
        print(algorithm.__name__)
        print('W shape:', W.shape, 'H shape:', H.shape)
        print('Updates:', len(losses) - 1)
        print(f'Training loss: {losses[0]:.6f} -> {losses[-1]:.6f}')
        # Reconstructed images would be W @ H.


if __name__ == '__main__':
    main()