from __future__ import annotations

import argparse
import json
from pathlib import Path

from photomatagent.scientific.capabilities.generation.inverse_retrieval import (
    train_inverse_index,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_DATA = REPOSITORY_ROOT / "data" / "photoelectric_vae" / "training"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "src"
    / "photomatagent"
    / "scientific"
    / "capabilities"
    / "generation"
    / "assets"
    / "photoelectric_vae"
    / "jarvis_inverse_v1"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the JARVIS inverse-design baseline")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_TRAINING_DATA,
        help="Directory containing the committed JARVIS CSV and raw archives",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Model artifact directory",
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train_inverse_index(
        args.dataset_dir / "jarvis_all_ir_candidates.csv",
        [
            args.dataset_dir / "raw/jarvis_dft3d_2025.zip",
            args.dataset_dir / "raw/jarvis_dft2d_2022.zip",
        ],
        args.output_dir,
        ridge_alpha=args.ridge_alpha,
        random_seed=args.seed,
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
