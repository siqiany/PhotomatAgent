from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from photomatagent.scientific.capabilities.generation.conditional_vae import (
    ConditionalVAE,
    VAEConfig,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_ROOT = (
    REPOSITORY_ROOT
    / "src"
    / "photomatagent"
    / "scientific"
    / "capabilities"
    / "generation"
    / "assets"
    / "photoelectric_vae"
)


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the property-conditioned composition VAE")
    parser.add_argument(
        "--inverse-model-dir",
        type=Path,
        default=DEFAULT_ASSET_ROOT / "jarvis_inverse_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ASSET_ROOT / "jarvis_cvae_v1" / "checkpoint.pt",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--condition-dropout", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    arrays = np.load(args.inverse_model_dir / "inverse_index.npz")
    properties = arrays["properties"].astype(np.float32)
    present = arrays["present"]
    center = arrays["center"].astype(np.float32)
    scale = arrays["scale"].astype(np.float32)
    compositions = arrays["compositions"].astype(np.float32)
    conditions = np.where(present, (properties - center) / scale, 0.0).astype(np.float32)
    conditions = np.clip(conditions, -8.0, 8.0)

    indices = np.random.permutation(len(conditions))
    split = int(0.8 * len(indices))
    train_indices, validation_indices = indices[:split], indices[split:]
    dataset = TensorDataset(
        torch.from_numpy(compositions[train_indices]), torch.from_numpy(conditions[train_indices])
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    config = VAEConfig(
        composition_dim=compositions.shape[1], condition_dim=conditions.shape[1]
    )
    model = ConditionalVAE(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for composition, condition in loader:
            mask = torch.rand_like(condition).ge(args.condition_dropout)
            sparse_condition = condition * mask
            reconstruction, mu, logvar = model(composition, sparse_condition)
            loss = model.loss(reconstruction, composition, mu, logvar, beta=args.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(composition)
            seen += len(composition)
        model.eval()
        with torch.inference_mode():
            validation_composition = torch.from_numpy(compositions[validation_indices])
            validation_condition = torch.from_numpy(conditions[validation_indices])
            reconstruction, mu, logvar = model(validation_composition, validation_condition)
            validation_loss = float(
                model.loss(
                    reconstruction, validation_composition, mu, logvar, beta=args.beta
                )
            )
            validation_mae = float(
                torch.mean(torch.abs(reconstruction - validation_composition))
            )
        record = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "validation_loss": validation_loss,
            "validation_composition_mae": validation_mae,
        }
        history.append(record)
        print(json.dumps(record))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_type": "conditional_composition_vae",
        "config": {
            "composition_dim": config.composition_dim,
            "condition_dim": config.condition_dim,
            "hidden_dim": config.hidden_dim,
            "latent_dim": config.latent_dim,
        },
        "model_state_dict": model.state_dict(),
        "condition_center": center.tolist(),
        "condition_scale": scale.tolist(),
        "property_fields": arrays["property_fields"].tolist(),
        "vocabulary": arrays["vocabulary"].tolist(),
        "training": {
            "record_count": len(conditions),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "condition_dropout": args.condition_dropout,
            "seed": args.seed,
            "final_metrics": history[-1],
            "data_source": portable_path(args.inverse_model_dir),
        },
    }
    torch.save(checkpoint, args.output)
    report_path = args.output.with_name("training_report.json")
    report_path.write_text(
        json.dumps(
            {"checkpoint": portable_path(args.output), **checkpoint["training"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
