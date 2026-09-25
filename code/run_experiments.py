"""Run reproducible NMF robustness experiments.

Example, from the ``code`` directory::

    python run_experiments.py \
        --orl-root data/ORL \
        --yaleb-root data/CroppedYaleB \
        --output-dir outputs/evaluation \
        --runs 5 --sample-frac 0.9 --max-iter 200

The runner performs a clean baseline, a block-size sweep with a fixed block
count, and a block-count sweep with a fixed block size.  Both algorithms use
the same sampled columns, corrupted matrices, and initialisation seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from algorithm import robust_l1_nmf, standard_nmf
from evaluation import clustering_metrics, relative_reconstruction_error
from experiments.data_utils import load_orl, load_yaleb, normalize_unit_interval
from experiments.noise import add_occlusion_noise


@dataclass(frozen=True)
class NoiseCondition:
    """One unique corruption setting and the plots in which it belongs."""

    condition_id: str
    block_size: int
    num_blocks: int
    include_size_sweep: bool
    include_count_sweep: bool


RESULT_FIELDS = [
    "dataset",
    "run",
    "condition_id",
    "include_size_sweep",
    "include_count_sweep",
    "block_size",
    "num_blocks",
    "nominal_occluded_fraction",
    "actual_occluded_fraction",
    "observed_changed_fraction",
    "algorithm",
    "rank",
    "max_iter",
    "tol",
    "min_iter",
    "patience",
    "iterations",
    "smoothing",
    "sample_frac",
    "actual_sample_frac",
    "sample_count",
    "sample_seed",
    "noise_seed",
    "initialisation_seed",
    "rre",
    "accuracy",
    "nmi",
    "runtime_seconds",
    "initial_training_loss",
    "final_training_loss",
]


def build_conditions(block_sizes, size_sweep_count, block_counts, count_sweep_size):
    """Build unique baseline, size-sweep, and count-sweep conditions."""

    block_sizes = _positive_integer_list("block_sizes", block_sizes)
    block_counts = _positive_integer_list("block_counts", block_counts)
    size_sweep_count = _positive_integer("size_sweep_count", size_sweep_count)
    count_sweep_size = _positive_integer("count_sweep_size", count_sweep_size)

    conditions = {
        (0, 0): {
            "condition_id": "baseline",
            "block_size": 0,
            "num_blocks": 0,
            "include_size_sweep": True,
            "include_count_sweep": True,
        }
    }

    for block_size in block_sizes:
        key = (block_size, size_sweep_count)
        conditions[key] = {
            "condition_id": f"b{block_size}_k{size_sweep_count}",
            "block_size": block_size,
            "num_blocks": size_sweep_count,
            "include_size_sweep": True,
            "include_count_sweep": False,
        }

    for num_blocks in block_counts:
        key = (count_sweep_size, num_blocks)
        if key in conditions:
            conditions[key]["include_count_sweep"] = True
        else:
            conditions[key] = {
                "condition_id": f"b{count_sweep_size}_k{num_blocks}",
                "block_size": count_sweep_size,
                "num_blocks": num_blocks,
                "include_size_sweep": False,
                "include_count_sweep": True,
            }

    return [NoiseCondition(**condition) for condition in conditions.values()]


def sample_column_indices(labels, fraction, random_state, stratified=True):
    """Return an exact-size, reproducible sample of column indices."""

    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")

    rng = np.random.default_rng(random_state)
    target_count = max(1, int(round(fraction * labels.size)))
    if not stratified:
        return np.sort(rng.choice(labels.size, size=target_count, replace=False))

    unique_labels, class_counts = np.unique(labels, return_counts=True)
    ideal_counts = class_counts * (target_count / labels.size)
    selected_counts = np.floor(ideal_counts).astype(int)

    # Retain every class whenever the requested sample is large enough.
    minimum_counts = np.ones_like(selected_counts) if target_count >= len(unique_labels) else 0
    selected_counts = np.maximum(selected_counts, minimum_counts)
    selected_counts = np.minimum(selected_counts, class_counts)

    while selected_counts.sum() < target_count:
        available = selected_counts < class_counts
        priority = np.where(available, ideal_counts - selected_counts, -np.inf)
        selected_counts[int(np.argmax(priority))] += 1
    while selected_counts.sum() > target_count:
        removable = selected_counts > minimum_counts
        priority = np.where(removable, selected_counts - ideal_counts, -np.inf)
        selected_counts[int(np.argmax(priority))] -= 1

    selected = []
    for label, count in zip(unique_labels, selected_counts):
        candidates = np.flatnonzero(labels == label)
        selected.extend(rng.choice(candidates, size=int(count), replace=False).tolist())
    return np.array(sorted(selected), dtype=int)


def derived_seed(base_seed, *parts):
    """Derive a stable uint32 seed from semantic experiment identifiers."""

    digest = hashlib.sha256()
    for part in (base_seed, *parts):
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:4], byteorder="little")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("ORL", "YaleB"),
                        default=["ORL", "YaleB"])
    parser.add_argument("--orl-root", default="data/ORL")
    parser.add_argument("--yaleb-root", default="data/CroppedYaleB")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[10, 15, 20])
    parser.add_argument("--size-sweep-count", type=int, default=1)
    parser.add_argument("--block-counts", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--count-sweep-size", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--sample-frac", type=float, default=0.9)
    parser.add_argument("--sampling", choices=("stratified", "random"),
                        default="stratified")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="relative-loss tolerance used for early stopping")
    parser.add_argument("--min-iter", type=int, default=20,
                        help="minimum updates before checking convergence")
    parser.add_argument("--patience", type=int, default=5,
                        help="consecutive small improvements required to stop")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="always use the full max-iter budget")
    parser.add_argument("--orl-rank", type=int, default=0,
                        help="0 uses the number of ORL classes")
    parser.add_argument("--yaleb-rank", type=int, default=0,
                        help="0 uses the number of YaleB classes")
    parser.add_argument("--smoothing", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=4328)
    parser.add_argument("--allow-overlap", action="store_true",
                        help="allow blocks to overlap; disabled by default")
    parser.add_argument("--clustering-metrics", action="store_true",
                        help="also compute optional Accuracy and NMI")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true",
                             help="continue a matching interrupted experiment")
    output_mode.add_argument("--overwrite", action="store_true",
                             help="replace an existing results checkpoint")
    return parser.parse_args(argv)


def validate_args(args, conditions):
    if args.runs < 1:
        raise ValueError("runs must be positive")
    if args.max_iter < 1:
        raise ValueError("max_iter must be positive")
    if not np.isfinite(args.tol) or args.tol <= 0:
        raise ValueError("tol must be finite and positive")
    if args.min_iter < 0:
        raise ValueError("min_iter must be nonnegative")
    if args.patience < 1:
        raise ValueError("patience must be positive")
    if not 0 < args.sample_frac <= 1:
        raise ValueError("sample_frac must be in the interval (0, 1]")
    if not np.isfinite(args.smoothing) or args.smoothing <= 0:
        raise ValueError("smoothing must be finite and positive")
    if args.orl_rank < 0 or args.yaleb_rank < 0:
        raise ValueError("rank overrides must be zero or positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if not conditions:
        raise ValueError("at least one experiment condition is required")


def main(argv=None):
    args = parse_args(argv)
    conditions = build_conditions(
        args.block_sizes,
        args.size_sweep_count,
        args.block_counts,
        args.count_sweep_size,
    )
    validate_args(args, conditions)

    os.makedirs(args.output_dir, exist_ok=True)
    reconstruction_dir = os.path.join(args.output_dir, "reconstructions")
    os.makedirs(reconstruction_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.csv")
    indices_path = os.path.join(args.output_dir, "sample_indices.json")
    metadata_path = os.path.join(args.output_dir, "metadata.json")

    dataset_specs = {
        "ORL": (load_orl, args.orl_root, args.orl_rank),
        "YaleB": (load_yaleb, args.yaleb_root, args.yaleb_rank),
    }

    configuration = {
        key: value
        for key, value in vars(args).items()
        if key not in {"resume", "overwrite"}
    }
    metadata = {
        "arguments": configuration,
        "conditions": [asdict(condition) for condition in conditions],
        "datasets": {},
    }
    results = []
    sampled_indices = {}
    if os.path.exists(results_path):
        if args.resume:
            results, sampled_indices, saved_metadata = _load_checkpoint(
                results_path,
                indices_path,
                metadata_path,
            )
            if saved_metadata.get("arguments") != configuration:
                raise ValueError("saved arguments do not match this resumed experiment")
            if saved_metadata.get("conditions") != metadata["conditions"]:
                raise ValueError("saved noise conditions do not match this resumed experiment")
            metadata = saved_metadata
        elif not args.overwrite:
            raise FileExistsError(
                f"{results_path} already exists; use --resume or --overwrite"
            )
    elif args.resume:
        print("No checkpoint found; starting a new experiment.")

    completed = {
        (row["dataset"], int(row["run"]), row["condition_id"], row["algorithm"])
        for row in results
    }

    for dataset_name in args.datasets:
        loader, root, rank_override = dataset_specs[dataset_name]
        print(f"==> Loading {dataset_name} from {root}")
        V_raw, labels, img_shape = loader(root)
        rank = rank_override or len(np.unique(labels))
        if rank > min(V_raw.shape):
            raise ValueError(f"rank={rank} is invalid for {dataset_name} shape {V_raw.shape}")

        dataset_metadata = {
            "root": root,
            "matrix_shape": list(V_raw.shape),
            "image_shape": list(img_shape),
            "class_count": int(len(np.unique(labels))),
            "rank": int(rank),
        }
        saved_dataset_metadata = metadata["datasets"].get(dataset_name)
        if (
            saved_dataset_metadata is not None
            and saved_dataset_metadata != dataset_metadata
        ):
            raise ValueError(
                f"saved dataset metadata does not match the current {dataset_name} data"
            )
        metadata["datasets"][dataset_name] = dataset_metadata

        for run_index in range(args.runs):
            sample_seed = derived_seed(args.seed, dataset_name, "sample", run_index)
            indices = sample_column_indices(
                labels,
                args.sample_frac,
                sample_seed,
                stratified=args.sampling == "stratified",
            )
            sample_key = f"{dataset_name}_run{run_index}"
            stored_indices = sampled_indices.get(sample_key)
            if stored_indices is not None and stored_indices != indices.tolist():
                raise ValueError(f"saved sample indices do not match {sample_key}")
            sampled_indices[sample_key] = indices.tolist()
            clean_raw = V_raw[:, indices]
            clean = normalize_unit_interval(clean_raw)
            selected_labels = labels[indices]
            if rank > min(clean.shape):
                raise ValueError(
                    f"rank={rank} is invalid after sampling {dataset_name} "
                    f"to shape {clean.shape}"
                )

            for condition in conditions:
                expected_keys = {
                    (dataset_name, run_index, condition.condition_id, "standard_nmf"),
                    (dataset_name, run_index, condition.condition_id, "robust_l1_nmf"),
                }
                existing_keys = expected_keys & completed
                if existing_keys:
                    if existing_keys != expected_keys:
                        raise ValueError(
                            "checkpoint contains only one algorithm for "
                            f"{dataset_name} run {run_index} {condition.condition_id}"
                        )
                    print(
                        f"    skipping completed {dataset_name} run={run_index + 1}/"
                        f"{args.runs} condition={condition.condition_id}"
                    )
                    continue

                noise_seed = derived_seed(
                    args.seed,
                    dataset_name,
                    "noise",
                    run_index,
                    condition.block_size,
                    condition.num_blocks,
                )
                initialisation_seed = derived_seed(
                    args.seed,
                    dataset_name,
                    "initialisation",
                    run_index,
                    condition.block_size,
                    condition.num_blocks,
                )

                if condition.num_blocks == 0:
                    noisy_raw = clean_raw.copy()
                    occlusion_mask = np.zeros(clean_raw.shape, dtype=bool)
                else:
                    noisy_raw, occlusion_mask = add_occlusion_noise(
                        clean_raw,
                        img_shape,
                        block_size=condition.block_size,
                        num_blocks=condition.num_blocks,
                        fill_value=255.0,
                        allow_overlap=args.allow_overlap,
                        random_state=noise_seed,
                        return_mask=True,
                    )
                noisy = normalize_unit_interval(noisy_raw)

                nominal_fraction = (
                    condition.block_size ** 2 * condition.num_blocks
                    / float(img_shape[0] * img_shape[1])
                )
                actual_occluded_fraction = float(np.mean(occlusion_mask))
                changed_fraction = float(np.mean(noisy_raw != clean_raw))
                snapshot = {
                    "clean": clean[:, 0],
                    "noisy": noisy[:, 0],
                }

                algorithms = (
                    ("standard_nmf", standard_nmf),
                    ("robust_l1_nmf", robust_l1_nmf),
                )
                convergence_tolerance = None if args.no_early_stopping else args.tol
                for algorithm_name, algorithm in algorithms:
                    print(
                        f"    {dataset_name} run={run_index + 1}/{args.runs} "
                        f"condition={condition.condition_id} algorithm={algorithm_name}"
                    )
                    start = time.perf_counter()
                    if algorithm_name == "robust_l1_nmf":
                        W, H, losses = algorithm(
                            noisy,
                            rank,
                            max_iter=args.max_iter,
                            random_state=initialisation_seed,
                            smoothing=args.smoothing,
                            tol=convergence_tolerance,
                            min_iter=args.min_iter,
                            patience=args.patience,
                        )
                    else:
                        W, H, losses = algorithm(
                            noisy,
                            rank,
                            max_iter=args.max_iter,
                            random_state=initialisation_seed,
                            tol=convergence_tolerance,
                            min_iter=args.min_iter,
                            patience=args.patience,
                        )
                    runtime = time.perf_counter() - start

                    rre = relative_reconstruction_error(clean, W, H)
                    optional = {"accuracy": "", "nmi": ""}
                    if args.clustering_metrics:
                        metrics = clustering_metrics(
                            H,
                            selected_labels,
                            n_clusters=len(np.unique(selected_labels)),
                            random_state=initialisation_seed,
                        )
                        optional = {
                            "accuracy": metrics["accuracy"],
                            "nmi": metrics["nmi"],
                        }

                    results.append({
                        "dataset": dataset_name,
                        "run": run_index,
                        "condition_id": condition.condition_id,
                        "include_size_sweep": condition.include_size_sweep,
                        "include_count_sweep": condition.include_count_sweep,
                        "block_size": condition.block_size,
                        "num_blocks": condition.num_blocks,
                        "nominal_occluded_fraction": nominal_fraction,
                        "actual_occluded_fraction": actual_occluded_fraction,
                        "observed_changed_fraction": changed_fraction,
                        "algorithm": algorithm_name,
                        "rank": rank,
                        "max_iter": args.max_iter,
                        "tol": convergence_tolerance if convergence_tolerance is not None else "",
                        "min_iter": args.min_iter,
                        "patience": args.patience,
                        "iterations": len(losses) - 1,
                        "smoothing": args.smoothing if algorithm_name == "robust_l1_nmf" else "",
                        "sample_frac": args.sample_frac,
                        "actual_sample_frac": len(indices) / len(labels),
                        "sample_count": len(indices),
                        "sample_seed": sample_seed,
                        "noise_seed": noise_seed,
                        "initialisation_seed": initialisation_seed,
                        "rre": rre,
                        "accuracy": optional["accuracy"],
                        "nmi": optional["nmi"],
                        "runtime_seconds": runtime,
                        "initial_training_loss": float(losses[0]),
                        "final_training_loss": float(losses[-1]),
                    })
                    completed.add((dataset_name, run_index, condition.condition_id, algorithm_name))
                    snapshot[algorithm_name] = W @ H[:, 0]

                if run_index == 0:
                    snapshot_path = os.path.join(
                        reconstruction_dir,
                        f"{dataset_name}_{condition.condition_id}.npz",
                    )
                    temporary_snapshot = snapshot_path + ".tmp.npz"
                    np.savez_compressed(
                        temporary_snapshot,
                        **snapshot,
                        image_shape=np.asarray(img_shape),
                        source_column=np.asarray(indices[0]),
                        block_size=np.asarray(condition.block_size),
                        num_blocks=np.asarray(condition.num_blocks),
                    )
                    os.replace(temporary_snapshot, snapshot_path)

                # Checkpoint after each condition because full YaleB runs can be
                # long even when convergence checks are enabled.
                _write_csv(results_path, results)
                _write_json(indices_path, sampled_indices)
                _write_json(metadata_path, metadata)

    print(f"\nSaved {len(results)} result rows to {args.output_dir}")
    print("Next: python plot_results.py --results "
          f"{os.path.join(args.output_dir, 'results.csv')} --output-dir {args.output_dir}")


def _write_csv(path, rows):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def _write_json(path, value):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def _load_checkpoint(results_path, indices_path, metadata_path):
    required_paths = (indices_path, metadata_path)
    missing = [path for path in required_paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "checkpoint is incomplete; missing " + ", ".join(missing)
        )

    with open(results_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RESULT_FIELDS:
            raise ValueError("checkpoint columns do not match the current result schema")
        results = list(reader)
    with open(indices_path, encoding="utf-8") as handle:
        sampled_indices = json.load(handle)
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    return results, sampled_indices, metadata


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_integer_list(name, values):
    if not values:
        raise ValueError(f"{name} must not be empty")
    return [_positive_integer(name, value) for value in values]


if __name__ == "__main__":
    main()
