"""Build a group-meeting data/profile slide from packaged VAE assets."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
ASSET_ROOT = (
    REPOSITORY_ROOT
    / "src"
    / "photomatagent"
    / "scientific"
    / "capabilities"
    / "generation"
    / "assets"
    / "photoelectric_vae"
)
INDEX_PATH = ASSET_ROOT / "jarvis_inverse_v1" / "inverse_index.npz"
CHECKPOINT_PATH = ASSET_ROOT / "jarvis_cvae_v1" / "checkpoint.pt"
SOURCE_CSV = HERE.parent / "source_data" / "property_coverage.csv"
OUTPUT_STEM = HERE / "vae_training_data_profile"

PROPERTY_LABELS = {
    "gap_selected_eV": "Band gap",
    "cutoff_wavelength_um_from_gap": "Cutoff wavelength",
    "formation_energy_eV_per_atom": "Formation energy",
    "energy_above_hull_eV_per_atom": "Energy above hull",
    "density_g_cm3": "Density",
    "dielectric_mean": "Mean dielectric",
    "avg_electron_mass_m0": "Electron mass",
    "avg_hole_mass_m0": "Hole mass",
    "bulk_modulus_GPa": "Bulk modulus",
    "shear_modulus_GPa": "Shear modulus",
    "exfoliation_energy_meV_per_atom": "Exfoliation energy",
    "max_IR_mode_cm-1": "Max IR mode",
    "min_IR_mode_cm-1": "Min IR mode",
    "spillage": "Spillage",
}

TARGETS = {
    "gap_selected_eV": 0.35,
    "formation_energy_eV_per_atom": -1.20,
    "energy_above_hull_eV_per_atom": 0.05,
    "density_g_cm3": 6.0,
    "dielectric_mean": 15.0,
    "avg_electron_mass_m0": 0.20,
}


def configure_matplotlib() -> None:
    cjk_family = "DejaVu Sans"
    font_candidates = [
        Path.home() / ".local" / "share" / "fonts" / "windows-cjk" / "msyh.ttc",
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.is_file():
            font_manager.fontManager.addfont(font_path)
            cjk_family = font_manager.FontProperties(fname=font_path).get_name()
            break
    bold_font_candidates = [
        Path.home() / ".local" / "share" / "fonts" / "windows-cjk" / "msyhbd.ttc",
        Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),
    ]
    for font_path in bold_font_candidates:
        if font_path.is_file():
            font_manager.fontManager.addfont(font_path)
            break
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                cjk_family,
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def load_profile() -> tuple[list[dict[str, float | int | str]], dict]:
    arrays = np.load(INDEX_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    properties = arrays["properties"]
    present = arrays["present"]
    fields = arrays["property_fields"].tolist()
    center = arrays["center"]
    scale = arrays["scale"]
    rows: list[dict[str, float | int | str]] = []
    for index, field in enumerate(fields):
        values = properties[present[:, index], index]
        target = TARGETS.get(field)
        target_z = (
            (target - float(center[index])) / float(scale[index])
            if target is not None
            else np.nan
        )
        rows.append(
            {
                "property": field,
                "label": PROPERTY_LABELS[field],
                "available_n": len(values),
                "total_n": len(properties),
                "coverage_percent": len(values) / len(properties) * 100,
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "target": target if target is not None else np.nan,
                "target_robust_z": target_z,
            }
        )
    return rows, checkpoint


def write_source_data(rows: list[dict[str, float | int | str]]) -> None:
    SOURCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_metric_card(ax, x, y, width, height, value, label, color="#6750A4"):
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor="#F6F1FB",
        edgecolor=color,
        linewidth=1.2,
    )
    ax.add_patch(card)
    ax.text(
        x + width / 2,
        y + height * 0.62,
        value,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        color=color,
    )
    ax.text(
        x + width / 2,
        y + height * 0.23,
        label,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
        color="#344054",
    )


def build_figure(rows: list[dict[str, float | int | str]], checkpoint: dict):
    configure_matplotlib()
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.18, 1.0],
        height_ratios=[1.0, 0.62],
        left=0.07,
        right=0.97,
        top=0.84,
        bottom=0.09,
        wspace=0.28,
        hspace=0.38,
    )
    ax_coverage = fig.add_subplot(grid[:, 0])
    ax_targets = fig.add_subplot(grid[0, 1])
    ax_facts = fig.add_subplot(grid[1, 1])

    fig.suptitle(
        "训练数据决定了多性质条件 VAE 的能力边界",
        x=0.07,
        y=0.975,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color="#172033",
    )
    fig.text(
        0.07,
        0.910,
        "11,240 条 JARVIS 组成–性质配对；不同性质的标签覆盖率差异显著",
        ha="left",
        fontsize=12,
        color="#5B6574",
    )

    labels = [str(row["label"]) for row in rows][::-1]
    coverage = np.asarray([float(row["coverage_percent"]) for row in rows][::-1])
    counts = np.asarray([int(row["available_n"]) for row in rows][::-1])
    colors = [
        "#5A7D9A" if value >= 60 else "#8F77B5" if value >= 20 else "#D09A52"
        for value in coverage
    ]
    bars = ax_coverage.barh(labels, coverage, color=colors, height=0.68)
    ax_coverage.axvline(20, color="#D8DDE5", linewidth=0.8, linestyle="--")
    ax_coverage.axvline(60, color="#D8DDE5", linewidth=0.8, linestyle="--")
    ax_coverage.set_xlim(0, 126)
    ax_coverage.set_xlabel("Label coverage in training data (%)")
    ax_coverage.set_title("a   14 个条件性质的有效标签覆盖率", loc="left", fontweight="bold")
    ax_coverage.grid(axis="x", color="#E8EBEF", linewidth=0.7)
    ax_coverage.set_axisbelow(True)
    for bar, value, count in zip(bars, coverage, counts, strict=True):
        ax_coverage.text(
            value + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%  (n={count:,})",
            va="center",
            fontsize=8.5,
            color="#344054",
        )
    ax_coverage.text(
        0.0,
        -0.12,
        "蓝：≥60%   紫：20–60%   橙：<20%。低覆盖性质可以输入，但条件学习证据更弱。",
        transform=ax_coverage.transAxes,
        fontsize=9,
        color="#5B6574",
    )

    selected = [row for row in rows if row["property"] in TARGETS]
    target_labels = [str(row["label"]) for row in selected][::-1]
    target_z = np.asarray([float(row["target_robust_z"]) for row in selected][::-1])
    ypos = np.arange(len(selected))
    ax_targets.axvspan(-1, 1, color="#E7F2ED", alpha=0.9, label="training IQR scale")
    ax_targets.axvline(0, color="#344054", linewidth=1.0)
    ax_targets.hlines(ypos, 0, target_z, color="#9AA4B2", linewidth=2)
    ax_targets.scatter(
        target_z,
        ypos,
        s=75,
        color="#6750A4",
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    ax_targets.set_yticks(ypos, target_labels)
    ax_targets.set_xlim(-2.2, 2.2)
    ax_targets.set_xlabel("Robust standardized target  (target − median) / IQR")
    ax_targets.set_title("b   演示请求位于训练分布的什么位置", loc="left", fontweight="bold")
    ax_targets.grid(axis="x", color="#E8EBEF", linewidth=0.7)
    ax_targets.set_axisbelow(True)
    ax_targets.text(
        0.99,
        0.02,
        "全部 |z| < 1，未触发 ±8 裁剪",
        transform=ax_targets.transAxes,
        ha="right",
        fontsize=9,
        color="#2F7664",
        fontweight="bold",
    )

    ax_facts.axis("off")
    ax_facts.set_title("c   模型与训练事实", loc="left", fontweight="bold", pad=4)
    config = checkpoint["config"]
    state = checkpoint["model_state_dict"]
    parameter_count = sum(value.numel() for value in state.values())
    metrics = checkpoint["training"]["final_metrics"]
    cards = [
        ("11,240", "training records"),
        (str(config["composition_dim"]), "elements"),
        (str(config["condition_dim"]), "conditions"),
        (str(config["latent_dim"]), "latent dims"),
        (f"{parameter_count:,}", "parameters"),
        (f"{metrics['validation_composition_mae']:.5f}", "validation MAE"),
    ]
    for index, (value, label) in enumerate(cards):
        column = index % 3
        row = index // 3
        add_metric_card(
            ax_facts,
            0.01 + column * 0.33,
            0.53 - row * 0.50,
            0.29,
            0.38,
            value,
            label,
        )
    ax_facts.text(
        0.01,
        -0.10,
        "10 epochs · batch 256 · β=0.02 · condition dropout=45% · single seed (42)",
        transform=ax_facts.transAxes,
        fontsize=9,
        color="#5B6574",
    )

    return fig


def main() -> None:
    rows, checkpoint = load_profile()
    write_source_data(rows)
    fig = build_figure(rows, checkpoint)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
