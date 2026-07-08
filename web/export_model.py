"""Export a saved DigitRecognizerModel .npz file for browser inference."""

import argparse
import json
from pathlib import Path

import numpy as np


ARRAY_KEYS = (
    "W1", "b1", "gamma1", "beta1", "running_mean1", "running_var1",
    "W2", "b2", "gamma2", "beta2", "running_mean2", "running_var2",
    "W3", "b3",
)
DIMENSION_KEYS = ("input_dim", "hidden_dim1", "hidden_dim2", "output_dim")


def export_model(source, destination):
    """Convert required inference tensors from NPZ to compact JSON."""
    source = Path(source)
    destination = Path(destination)
    with np.load(source, allow_pickle=False) as saved:
        required = set(ARRAY_KEYS + DIMENSION_KEYS)
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"Model is missing keys: {sorted(missing)}")

        payload = {
            key: np.asarray(saved[key], dtype=np.float32).reshape(-1).tolist()
            for key in ARRAY_KEYS
        }
        payload.update({key: int(saved[key]) for key in DIMENSION_KEYS})

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        json.dump(payload, output, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(
        description="Export NumPy digit-recognizer weights for the static website.")
    parser.add_argument("source", help="Saved .npz model")
    parser.add_argument(
        "destination", nargs="?", default="web/models/digit-model.json",
        help="Output JSON path (default: web/models/digit-model.json)")
    args = parser.parse_args()
    export_model(args.source, args.destination)
    print(f"Exported browser model to {args.destination}")


if __name__ == "__main__":
    main()
