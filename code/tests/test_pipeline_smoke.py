import csv
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from plot_results import main as plot_main
from run_experiments import main as experiment_main


class TestPipelineSmoke(unittest.TestCase):
    def test_tiny_end_to_end_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = os.path.join(temporary_directory, "ORL")
            for subject in ("s1", "s2"):
                subject_dir = os.path.join(data_root, subject)
                os.makedirs(subject_dir)
                for image_index in range(2):
                    value = 40 + 70 * (subject == "s2") + image_index * 10
                    pixels = np.full((6, 6), value, dtype=np.uint8)
                    Image.fromarray(pixels, mode="L").save(
                        os.path.join(subject_dir, f"{image_index + 1}.pgm")
                    )

            output_dir = os.path.join(temporary_directory, "outputs")
            experiment_main([
                "--datasets", "ORL",
                "--orl-root", data_root,
                "--output-dir", output_dir,
                "--block-sizes", "1",
                "--size-sweep-count", "1",
                "--block-counts", "1",
                "--count-sweep-size", "1",
                "--runs", "1",
                "--sample-frac", "1",
                "--max-iter", "2",
                "--orl-rank", "2",
                "--seed", "9",
            ])

            results_path = os.path.join(output_dir, "results.csv")
            with open(results_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)  # 2 conditions x 2 algorithms
            self.assertTrue(all(np.isfinite(float(row["rre"])) for row in rows))
            self.assertTrue(all(int(row["iterations"]) == 2 for row in rows))
            self.assertTrue(all(float(row["actual_sample_frac"]) == 1.0 for row in rows))

            with self.assertRaises(FileExistsError):
                experiment_main([
                    "--datasets", "ORL",
                    "--orl-root", data_root,
                    "--output-dir", output_dir,
                    "--block-sizes", "1",
                    "--block-counts", "1",
                    "--runs", "1",
                    "--sample-frac", "1",
                    "--max-iter", "2",
                    "--orl-rank", "2",
                    "--seed", "9",
                ])

            experiment_main([
                "--datasets", "ORL",
                "--orl-root", data_root,
                "--output-dir", output_dir,
                "--block-sizes", "1",
                "--size-sweep-count", "1",
                "--block-counts", "1",
                "--count-sweep-size", "1",
                "--runs", "1",
                "--sample-frac", "1",
                "--max-iter", "2",
                "--orl-rank", "2",
                "--seed", "9",
                "--resume",
            ])
            with open(results_path, newline="", encoding="utf-8") as handle:
                resumed_rows = list(csv.DictReader(handle))
            self.assertEqual(resumed_rows, rows)

            stale_figure_dir = os.path.join(output_dir, "figures", "reconstructions")
            os.makedirs(stale_figure_dir, exist_ok=True)
            stale_figure = os.path.join(stale_figure_dir, "old_condition.png")
            with open(stale_figure, "wb") as handle:
                handle.write(b"stale")

            plot_main(["--results", results_path, "--output-dir", output_dir])
            self.assertFalse(os.path.exists(stale_figure))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "summary.csv")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "rre_summary_table.tex")))
            self.assertTrue(os.path.isfile(
                os.path.join(output_dir, "figures", "ORL_rre_vs_block_size.png")
            ))


if __name__ == "__main__":
    unittest.main()
