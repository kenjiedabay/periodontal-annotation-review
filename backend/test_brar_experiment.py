import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from app.brar_experiment.prepare import SAFE_METADATA, prepare
from app.brar_experiment.validate import stratified_folds


class BrarPreparationTest(unittest.TestCase):
    def test_stratified_folds_are_complete_and_balanced(self):
        import numpy as np
        labels = np.asarray([0] * 10 + [1] * 15 + [2] * 20)
        folds = stratified_folds(labels, 5, 42)
        self.assertEqual(sorted(index for fold in folds for index in fold), list(range(45)))
        for fold in folds:
            self.assertEqual([sum(labels[index] == label for index in fold) for label in range(3)], [2, 3, 4])

    def test_prepare_creates_stratified_leakage_safe_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            root, output = tmp_path / "brar", tmp_path / "output"
            root.mkdir()
            fields = ["File name", "Age", "Gender", "Bone resorption", "Bone resorption Age", "Level", "Number of missing teeth", "Implant", "Residual root", "Functional tooth logarithm"]
            with (root / "meta_data.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                for level in (1, 2, 3):
                    folder = root / f"level_{level}"; folder.mkdir()
                    for index in range(10):
                        name = f"p_{level}_{index}.jpg"; Image.new("L", (20 + index, 10), level * 40 + index).save(folder / name)
                        writer.writerow({"File name": name, "Age": 30 + index, "Gender": index % 2, "Bone resorption": .2, "Bone resorption Age": .5, "Level": level, "Number of missing teeth": 0, "Implant": 0, "Residual root": 0, "Functional tooth logarithm": 14})
            report = prepare(root, output, 7)
            with (output / "brar_split_manifest.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(report["records"], 30)
            self.assertEqual(set(rows[0]), {"file_name", "image_path", "split", "level", *SAFE_METADATA})
            self.assertNotIn("Bone resorption", rows[0]); self.assertNotIn("Bone resorption Age", rows[0])
            self.assertEqual({row["split"] for row in rows}, {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
