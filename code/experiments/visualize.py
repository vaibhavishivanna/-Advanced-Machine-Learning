"""
COMP4328/5328/8328 - Advanced Machine Learning - Assignment 1
Part: Data + Experiments (Person 2)
Module: visualize.py - original vs. corrupted comparison figures
"""
import os
import matplotlib.pyplot as plt

from noise import add_occlusion_noise


def show_original_vs_corrupted(V_clean, img_shape, block_sizes, num_blocks_list,
                                 sample_index=0, random_state=0, save_path=None,
                                 dataset_name=''):
    """
    Plot one row: the original image, followed by that same image
    corrupted under every (block_size, num_blocks) setting requested.

    Args:
        V_clean: (D, N) clean data matrix.
        img_shape: (H, W).
        block_sizes: list of block edge lengths b to demonstrate.
        num_blocks_list: list of block counts k to demonstrate.
        sample_index: which column of V_clean to use as the demo image.
        random_state: seed, for reproducible block placement.
        save_path: if given, saves the figure to this path (PNG).
        dataset_name: used in the figure title.

    Returns:
        the matplotlib Figure.
    """
    H, W = img_shape
    settings = [(b, k) for b in block_sizes for k in num_blocks_list]
    n_cols = 1 + len(settings)

    fig, axes = plt.subplots(1, n_cols, figsize=(2.1 * n_cols, 2.6))

    original = V_clean[:, sample_index].reshape(H, W)
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original')
    axes[0].axis('off')

    for ax, (b, k) in zip(axes[1:], settings):
        V_noisy = add_occlusion_noise(
            V_clean[:, sample_index:sample_index + 1], img_shape,
            block_size=b, num_blocks=k, random_state=random_state)
        corrupted = V_noisy[:, 0].reshape(H, W)
        ax.imshow(corrupted, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'b={b}, k={k}')
        ax.axis('off')

    if dataset_name:
        fig.suptitle(f'{dataset_name}: occlusion noise, block size (b) and block count (k)', y=1.05)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
