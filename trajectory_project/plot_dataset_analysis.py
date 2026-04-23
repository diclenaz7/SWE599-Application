"""Generate dataset summary figures from synthetic_trajectories.npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def discrete_curvature(xy: np.ndarray) -> np.ndarray:
    """Signed-magnitude curvature κ along a 2D polyline (uniform parameter)."""
    x = xy[:, 0]
    y = xy[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    cross = dx * ddy - dy * ddx
    speed_sq = dx * dx + dy * dy
    den = np.power(speed_sq, 1.5) + 1e-12
    return np.abs(cross) / den


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def plot_dataset_distribution(dataset: dict[str, np.ndarray], out_path: Path) -> None:
    trajectories = dataset["trajectories"]
    starts = dataset["starts"]
    goals = dataset["goals"]
    strengths = dataset["curve_strengths"]

    seg = np.linalg.norm(np.diff(trajectories, axis=1), axis=2)
    path_lengths = seg.sum(axis=1)
    chord_lengths = np.linalg.norm(goals - starts, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(strengths, bins=30, color="steelblue", edgecolor="white", alpha=0.9)
    axes[0].set_title("Curve strength (generator param)")
    axes[0].set_xlabel("strength")
    axes[0].set_ylabel("count")

    axes[1].hist(chord_lengths, bins=30, color="seagreen", edgecolor="white", alpha=0.9)
    axes[1].set_title("Start–goal chord length")
    axes[1].set_xlabel("distance")
    axes[1].set_ylabel("count")

    axes[2].hist(path_lengths, bins=30, color="coral", edgecolor="white", alpha=0.9)
    axes[2].set_title("Polyline path length")
    axes[2].set_xlabel("length")
    axes[2].set_ylabel("count")

    fig.suptitle("Dataset distribution", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_trajectories(dataset: dict[str, np.ndarray], out_path: Path, n: int = 24) -> None:
    trajectories = dataset["trajectories"]
    starts = dataset["starts"]
    goals = dataset["goals"]
    n = min(n, len(trajectories))
    indices = np.linspace(0, len(trajectories) - 1, n, dtype=int)

    fig, ax = plt.subplots(figsize=(7, 6))
    for idx in indices:
        traj = trajectories[idx]
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.75, linewidth=1.2)

    ax.scatter(starts[indices, 0], starts[indices, 1], marker="o", s=36, label="starts", zorder=5)
    ax.scatter(goals[indices, 0], goals[indices, 1], marker="x", s=48, label="goals", zorder=5)
    ax.set_title("Sample synthetic trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curvature_distribution(dataset: dict[str, np.ndarray], out_path: Path) -> None:
    trajectories = dataset["trajectories"]
    mean_kappa = []
    max_kappa = []
    for i in range(len(trajectories)):
        k = discrete_curvature(trajectories[i])
        mean_kappa.append(np.nanmean(k))
        max_kappa.append(np.nanmax(k))
    mean_kappa = np.array(mean_kappa)
    max_kappa = np.array(max_kappa)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(mean_kappa, bins=40, color="slateblue", edgecolor="white", alpha=0.9)
    axes[0].set_title("Mean |κ| per trajectory")
    axes[0].set_xlabel("mean curvature")
    axes[0].set_ylabel("count")

    axes[1].hist(max_kappa, bins=40, color="darkorange", edgecolor="white", alpha=0.9)
    axes[1].set_title("Max |κ| per trajectory")
    axes[1].set_xlabel("max curvature")
    axes[1].set_ylabel("count")

    fig.suptitle("Curvature distribution (discrete κ along polyline)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_start_goal_scatter(dataset: dict[str, np.ndarray], out_path: Path) -> None:
    starts = dataset["starts"]
    goals = dataset["goals"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(starts[:, 0], starts[:, 1], s=14, alpha=0.55, label="starts [0,2]²", c="tab:blue")
    ax.scatter(goals[:, 0], goals[:, 1], s=14, alpha=0.55, label="goals [8,10]²", c="tab:red", marker="s")
    ax.set_title("Start and goal sampling regions")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save dataset analysis figures.")
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("synthetic_trajectories.npz"),
        help="Path to dataset .npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for PNG outputs",
    )
    args = parser.parse_args()

    if not args.npz.is_file():
        raise SystemExit(f"Dataset not found: {args.npz}. Run generate_dataset.py first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.npz)

    plot_dataset_distribution(dataset, args.out_dir / "dataset_distribution.png")
    plot_sample_trajectories(dataset, args.out_dir / "sample_synthetic_trajectories.png")
    plot_curvature_distribution(dataset, args.out_dir / "curvature_distribution.png")
    plot_start_goal_scatter(dataset, args.out_dir / "start_goal_regions.png")

    print(f"Wrote figures to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
