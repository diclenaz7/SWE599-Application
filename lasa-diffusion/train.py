import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import DEFAULT_TRAINING_SHAPES, LASATrajectoryDataset
from diffusion import Diffusion
from model import TrajectoryDenoiser


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM-style LASA trajectory model.")
    parser.add_argument("--shape-name", default="Angle", help="LASA shape name, e.g. Angle, Sine, Spoon.")
    parser.add_argument(
        "--shape-names",
        nargs="+",
        default=None,
        help="Train on multiple LASA shapes, e.g. --shape-names Angle CShape Sine Spoon.",
    )
    parser.add_argument(
        "--curated-shapes",
        action="store_true",
        help=f"Train on the curated multi-shape subset: {', '.join(DEFAULT_TRAINING_SHAPES)}.",
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Fixed trajectory length.")
    parser.add_argument("--batch-size", type=int, default=7, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs.")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps.")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden width of the denoising network.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for checkpoint and plots.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of demonstrations reserved for validation.")
    parser.add_argument(
        "--architecture",
        choices=["mlp", "temporal-conv"],
        default="temporal-conv",
        help="Denoising network architecture.",
    )
    parser.add_argument(
        "--conditioning",
        choices=["none", "start-goal"],
        default="none",
        help="Optional conditioning signal for training.",
    )
    return parser.parse_args()


def selected_shape_names(args):
    if args.curated_shapes:
        return DEFAULT_TRAINING_SHAPES
    if args.shape_names:
        return args.shape_names
    return [args.shape_name]


def split_dataset(dataset, val_split, seed):
    if val_split <= 0 or len(dataset) < 2:
        return dataset, None

    val_size = max(1, int(round(len(dataset) * val_split)))
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def unpack_batch(batch, conditioning, device):
    if conditioning == "start-goal":
        x0, condition = batch
        return x0.to(device), condition.to(device)
    return batch.to(device), None


def noise_direction_metrics(predicted_noise, true_noise):
    predicted_positive = predicted_noise >= 0
    true_positive = true_noise >= 0

    correct = predicted_positive == true_positive
    accuracy = correct.float().mean().item()

    true_positives = (predicted_positive & true_positive).sum().item()
    false_positives = (predicted_positive & ~true_positive).sum().item()
    false_negatives = (~predicted_positive & true_positive).sum().item()

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return accuracy, f1


def run_epoch(model, loader, diffusion, loss_fn, optimizer, conditioning, device):
    is_training = optimizer is not None
    model.train(is_training)

    losses = []
    accuracies = []
    f1_scores = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in loader:
            x0, condition = unpack_batch(batch, conditioning, device)
            t = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=device)

            noisy_x, noise = diffusion.add_noise(x0, t)
            predicted_noise = model(noisy_x, t, condition=condition)
            loss = loss_fn(predicted_noise, noise)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy, f1 = noise_direction_metrics(predicted_noise.detach(), noise)
            losses.append(loss.item())
            accuracies.append(accuracy)
            f1_scores.append(f1)

    return {
        "loss": sum(losses) / len(losses),
        "accuracy": sum(accuracies) / len(accuracies),
        "f1": sum(f1_scores) / len(f1_scores),
    }


def plot_history(history, output_dir):
    metric_specs = [
        ("loss", "MSE Loss", "loss.png"),
        ("accuracy", "Noise Direction Accuracy", "accuracy.png"),
        ("f1", "Noise Direction F1 Score", "f1.png"),
    ]

    for metric, title, filename in metric_specs:
        plt.figure(figsize=(7, 4))
        plt.plot(history[f"train_{metric}"], label="training")
        if history[f"val_{metric}"]:
            plt.plot(history[f"val_{metric}"], label="validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    shape_names = selected_shape_names(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    if args.output_dir == "outputs" and len(shape_names) > 1:
        output_dir = Path("outputs/multi_shape_temporal_conv")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LASATrajectoryDataset(shape_names=shape_names, seq_len=args.seq_len, conditioning=args.conditioning)
    train_dataset, val_dataset = split_dataset(dataset, args.val_split, args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset is not None else None

    print(f"Training shapes: {', '.join(shape_names)}")
    print(f"Demonstrations per shape: {dataset.shape_counts}")
    print(f"Total demonstrations: {len(dataset)}")

    cond_dim = 4 if args.conditioning == "start-goal" else 0
    model = TrajectoryDenoiser(
        seq_len=args.seq_len,
        hidden=args.hidden,
        cond_dim=cond_dim,
        architecture=args.architecture,
    ).to(device)
    diffusion = Diffusion(timesteps=args.timesteps, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": [],
    }

    for epoch in tqdm(range(args.epochs)):
        train_metrics = run_epoch(model, train_loader, diffusion, loss_fn, optimizer, args.conditioning, device)
        val_metrics = (
            run_epoch(model, val_loader, diffusion, loss_fn, None, args.conditioning, device)
            if val_loader is not None
            else None
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])

        if val_metrics is not None:
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1"])

        if epoch % 200 == 0 or epoch == args.epochs - 1:
            message = (
                f"Epoch {epoch}, "
                f"Train loss: {train_metrics['loss']:.4f}, "
                f"Train acc: {train_metrics['accuracy']:.4f}, "
                f"Train F1: {train_metrics['f1']:.4f}"
            )
            if val_metrics is not None:
                message += (
                    f", Val loss: {val_metrics['loss']:.4f}, "
                    f"Val acc: {val_metrics['accuracy']:.4f}, "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )
            print(message)

    checkpoint_path = output_dir / "lasa_diffusion.pt"
    shape_name = shape_names[0] if len(shape_names) == 1 else "multi_shape"
    torch.save(
        {
            "model": model.state_dict(),
            "mean": dataset.mean,
            "std": dataset.std,
            "shape_name": shape_name,
            "shape_names": shape_names,
            "shape_counts": dataset.shape_counts,
            "seq_len": args.seq_len,
            "timesteps": args.timesteps,
            "hidden": args.hidden,
            "architecture": args.architecture,
            "conditioning": args.conditioning,
            "cond_dim": cond_dim,
            "losses": history["train_loss"],
            "train_losses": history["train_loss"],
            "val_losses": history["val_loss"],
            "train_accuracy": history["train_accuracy"],
            "val_accuracy": history["val_accuracy"],
            "train_f1": history["train_f1"],
            "val_f1": history["val_f1"],
            "metric_definitions": {
                "loss": "Mean squared error between predicted and true diffusion noise.",
                "accuracy": "Proxy metric: fraction of noise coordinates with the correct predicted sign.",
                "f1": "Proxy metric: F1 score after binarizing each noise coordinate by sign.",
            },
            "val_split": args.val_split,
        },
        checkpoint_path,
    )

    plot_history(history, output_dir)

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved loss plot to {output_dir / 'loss.png'}")
    print(f"Saved accuracy plot to {output_dir / 'accuracy.png'}")
    print(f"Saved F1 plot to {output_dir / 'f1.png'}")


if __name__ == "__main__":
    main()
