"""Run the full preprocessing pipeline (steps 1-4) in order."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "stock_prices.csv"
STEPS = [
    "src/01_handle_missing_data.py",
    "src/02_encode_categorical.py",
    "src/03_scale_features.py",
    "src/04_train_test_split.py",
]


def main():
    if not RAW_FILE.exists():
        sys.exit(f"Dataset not found: {RAW_FILE}\nPlace stock_prices.csv there first.")

    total_start = time.time()
    for number, step in enumerate(STEPS, start=1):
        print(f"\n{'=' * 60}\nSTEP {number}/{len(STEPS)}: {step}\n{'=' * 60}")
        start = time.time()
        result = subprocess.run([sys.executable, step], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"\nStep failed: {step}")
        print(f"Finished in {time.time() - start:.1f}s")

    print(f"\nPipeline complete in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
