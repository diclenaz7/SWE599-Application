import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion import Diffusion
from model import TrajectoryDenoiser


def parse_args():
    parser = argparse.ArgumentParser(description="Sample trajectories from a trained LASA diffusion model.")
    parser.add_argument("--checkpoint", default="outputs/lasa_diffusion.pt", help="Path to checkpoint.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of trajectories to generate.")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override diffusion timesteps.")
    parser.add_argument("--hidden", type=int, default=None, help="Override denoising MLP hidden width.")
    parser.add_argument("--output", default="outputs/generated_trajectories.png", help="Output figure path.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--show", action="store_true", help="Show the plot window after saving.")
    parser.add_argument("--start", nargs=2, type=float, default=None, metavar=("X", "Y"), help="Optional start point.")
    parser.add_argument("--goal", nargs=2, type=float, default=None, metavar=("X", "Y"), help="Optional goal point.")
    return parser.parse_args()


def load_model(checkpoint_path, device, seq_len=None, hidden=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    seq_len = seq_len or checkpoint.get("seq_len", 256)
    hidden = hidden or checkpoint.get("hidden", 256)
    cond_dim = checkpoint.get("cond_dim", 0)
    architecture = checkpoint.get("architecture", "mlp")

    model = TrajectoryDenoiser(
        seq_len=seq_len,
        hidden=hidden,
        cond_dim=cond_dim,
        architecture=architecture,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, checkpoint, seq_len


def normalize_condition(condition, checkpoint, num_samples, device):
    if condition is None or checkpoint.get("cond_dim", 0) == 0:
        return None

    condition = np.asarray(condition, dtype=np.float32).reshape(1, 4)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32).reshape(2)
    std = np.asarray(checkpoint["std"], dtype=np.float32).reshape(2)

    start = (condition[:, 0:2] - mean) / std
    goal = (condition[:, 2:4] - mean) / std
    normalized = np.concatenate([start, goal], axis=1)
    normalized = np.repeat(normalized, num_samples, axis=0)
    return torch.tensor(normalized, dtype=torch.float32, device=device)


def generate_samples(checkpoint_path, num_samples=10, seq_len=None, timesteps=None, hidden=None, seed=7, condition=None):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, checkpoint, seq_len = load_model(checkpoint_path, device, seq_len=seq_len, hidden=hidden)
    timesteps = timesteps or checkpoint.get("timesteps", 1000)
    diffusion = Diffusion(timesteps=timesteps, device=device)
    normalized_condition = normalize_condition(condition, checkpoint, num_samples, device)

    samples = diffusion.sample(model, shape=(num_samples, seq_len, 2), condition=normalized_condition)
    samples = samples.cpu().numpy()
    std = np.asarray(checkpoint["std"])
    mean = np.asarray(checkpoint["mean"])
    return samples * std + mean, checkpoint


def plot_samples(samples, output_path, title="Generated LASA-like Trajectories", show=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    for traj in samples:
        plt.plot(traj[:, 0], traj[:, 1], linewidth=2)

    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()


def main():
    args = parse_args()
    condition = None
    if args.start is not None and args.goal is not None:
        condition = [args.start[0], args.start[1], args.goal[0], args.goal[1]]

    samples, checkpoint = generate_samples(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        timesteps=args.timesteps,
        hidden=args.hidden,
        seed=args.seed,
        condition=condition,
    )
    shape_name = checkpoint.get("shape_name", "LASA")
    plot_samples(
        samples,
        args.output,
        title=f"Generated {shape_name} Trajectories",
        show=args.show,
    )
    print(f"Saved generated trajectories to {args.output}")


if __name__ == "__main__":
    main()
