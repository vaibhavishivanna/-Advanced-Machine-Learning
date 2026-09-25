import csv
import os
import tempfile
import unittest

import numpy as np

from plot_results import aggregate_results, read_results


class TestResultAggregation(unittest.TestCase):
    def test_reader_accepts_results_without_convergence_columns(self):
        row = {
            "dataset": "ORL",
            "run": 0,
            "condition_id": "baseline",
            "include_size_sweep": True,
            "include_count_sweep": True,
            "block_size": 0,
            "num_blocks": 0,
            "rank": 2,
            "max_iter": 10,
            "iterations": 10,
            "sample_count": 4,
            "nominal_occluded_fraction": 0.0,
            "actual_occluded_fraction": 0.0,
            "observed_changed_fraction": 0.0,
            "sample_frac": 1.0,
            "actual_sample_frac": 1.0,
            "rre": 0.1,
            "runtime_seconds": 0.2,
            "initial_training_loss": 2.0,
            "final_training_loss": 1.0,
            "accuracy": "",
            "nmi": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "results.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            parsed = read_results(path)[0]

        self.assertIsNone(parsed["tol"])
        self.assertIsNone(parsed["min_iter"])
        self.assertIsNone(parsed["patience"])

    def test_mean_sample_standard_deviation_and_grouping(self):
        rows = []
        for run, rre, runtime in ((0, 0.1, 1.0), (1, 0.2, 2.0), (2, 0.3, 3.0)):
            rows.append({
                "dataset": "ORL",
                "condition_id": "b5_k1",
                "block_size": 5,
                "num_blocks": 1,
                "algorithm": "standard_nmf",
                "run": run,
                "rre": rre,
                "runtime_seconds": runtime,
                "accuracy": None,
                "nmi": None,
            })
        rows.append({
            "dataset": "ORL",
            "condition_id": "b5_k1",
            "block_size": 5,
            "num_blocks": 1,
            "algorithm": "robust_l1_nmf",
            "run": 0,
            "rre": 0.4,
            "runtime_seconds": 4.0,
            "accuracy": None,
            "nmi": None,
        })

        summary = aggregate_results(rows)
        standard = next(row for row in summary if row["algorithm"] == "standard_nmf")
        robust = next(row for row in summary if row["algorithm"] == "robust_l1_nmf")

        self.assertAlmostEqual(standard["mean_rre"], 0.2)
        self.assertAlmostEqual(standard["std_rre"], 0.1)
        self.assertAlmostEqual(standard["mean_runtime_seconds"], 2.0)
        self.assertAlmostEqual(standard["std_runtime_seconds"], 1.0)
        self.assertEqual(standard["runs"], 3)
        self.assertEqual(robust["std_rre"], 0.0)
        self.assertEqual(robust["runs"], 1)

    def test_optional_metric_standard_deviations(self):
        rows = []
        for run, accuracy, nmi in ((0, 0.6, 0.4), (1, 0.8, 0.6)):
            rows.append({
                "dataset": "ORL",
                "condition_id": "baseline",
                "block_size": 0,
                "num_blocks": 0,
                "algorithm": "standard_nmf",
                "run": run,
                "rre": 0.2,
                "runtime_seconds": 1.0,
                "accuracy": accuracy,
                "nmi": nmi,
            })
        summary = aggregate_results(rows)[0]
        self.assertAlmostEqual(summary["mean_accuracy"], 0.7)
        self.assertAlmostEqual(summary["std_accuracy"], np.sqrt(0.02))
        self.assertAlmostEqual(summary["mean_nmi"], 0.5)
        self.assertAlmostEqual(summary["std_nmi"], np.sqrt(0.02))


if __name__ == "__main__":
    unittest.main()
