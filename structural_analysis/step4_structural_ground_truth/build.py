"""Canonical output location; reuse the audited confirmed-only Step 4 builder."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('confirmed_structural_builder', HERE.parent / 'step4_structural_dataset' / 'build.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if __name__ == '__main__':
    module.main(HERE)
