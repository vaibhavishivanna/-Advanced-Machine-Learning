"""Load both datasets and generate example occlusion outputs.

Run from the code/ folder, with ORL and CroppedYaleB placed under
code/data/ (per the assignment's submission instructions):

    python -m experiments.run_demo \
        --orl_root data/ORL \
        --yaleb_root data/CroppedYaleB \
        --out_dir outputs  \
        --block_sizes 10 15 20 \
        --num_blocks 1

Run with --help for all options.
"""
import argparse
import os
import numpy as np

if __package__:
    from .data_utils import load_orl, load_yaleb, normalize_unit_interval
    from .noise import add_occlusion_noise
    from .visualize import show_original_vs_corrupted
else:
    from data_utils import load_orl, load_yaleb, normalize_unit_interval
    from noise import add_occlusion_noise
    from visualize import show_original_vs_corrupted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orl_root', default='data/ORL')
    parser.add_argument('--yaleb_root', default='data/CroppedYaleB')
    parser.add_argument('--out_dir', default='outputs')
    parser.add_argument('--block_sizes', type=int, nargs='+', default=[10, 15, 20])
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[1])
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--allow_overlap',
        action='store_true',
        help='allow blocks to overlap instead of enforcing separate regions',
    )
    args = parser.parse_args()

    fig_dir = os.path.join(args.out_dir, 'figures')
    arr_dir = os.path.join(args.out_dir, 'arrays')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(arr_dir, exist_ok=True)

    datasets = [
        ('ORL', load_orl, args.orl_root),
        ('YaleB', load_yaleb, args.yaleb_root),
    ]

    for name, loader, root in datasets:
        print(f'==> Loading {name} dataset from "{root}" ...')
        V, Y, img_shape = loader(root)
        print(f'    V.shape={V.shape}  Y.shape={Y.shape}  img_shape={img_shape}  '
              f'num_subjects={len(set(Y))}  value_range=[{V.min():.0f}, {V.max():.0f}]')

        fig_path = os.path.join(fig_dir, f'{name}_occlusion_comparison.png')
        show_original_vs_corrupted(
            V, img_shape, args.block_sizes, args.num_blocks,
            sample_index=args.sample_index, random_state=args.seed,
            save_path=fig_path, dataset_name=name,
            allow_overlap=args.allow_overlap)
        print(f'    saved comparison figure -> {fig_path}')

        V_unit = normalize_unit_interval(V)
        np.save(os.path.join(arr_dir, f'{name}_clean.npy'), V_unit)
        np.save(os.path.join(arr_dir, f'{name}_labels.npy'), Y)

        for b in args.block_sizes:
            for k in args.num_blocks:
                V_noisy = add_occlusion_noise(V, img_shape, block_size=b,
                                               num_blocks=k,
                                               allow_overlap=args.allow_overlap,
                                               random_state=args.seed)
                V_noisy_unit = normalize_unit_interval(V_noisy)
                np.save(os.path.join(arr_dir, f'{name}_noisy_b{b}_k{k}.npy'), V_noisy_unit)

        print(f'    saved clean + {len(args.block_sizes) * len(args.num_blocks)} '
              f'noisy arrays -> {arr_dir}')

    print('\nDone.')


if __name__ == '__main__':
    main()
