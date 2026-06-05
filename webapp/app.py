from pathlib import Path
import importlib.util
import io
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "lasa-diffusion" / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

DIFFUSION_DIR = ROOT / "lasa-diffusion"

DEFAULT_TRAINING_SHAPES = [
    "Angle",
    "CShape",
    "GShape",
    "JShape",
    "LShape",
    "Sine",
    "Spoon",
    "WShape",
]


def load_module_from_path(runtime_name, module_path):
    module_path = Path(module_path).resolve()
    spec = importlib.util.spec_from_file_location(runtime_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {runtime_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset_class():
    module = load_module_from_path("lasa_diffusion_dataset_runtime", DIFFUSION_DIR / "dataset.py")
    return module.LASATrajectoryDataset


LASATrajectoryDataset = load_dataset_class()


def load_generate_samples():
    module = load_module_from_path("lasa_diffusion_sample_runtime", DIFFUSION_DIR / "sample.py")
    return module.generate_samples


generate_samples = load_generate_samples()


DEFAULT_CHECKPOINT = DIFFUSION_DIR / "outputs" / "multi_shape_temporal_conv" / "lasa_diffusion.pt"
CANVAS_SIZE = 420
DEFAULT_NUM_SAMPLES = 6
DEFAULT_TIMESTEPS = 1000
DEFAULT_DRAWING_STRENGTH = 0.35
DEFAULT_START_GOAL_STRENGTH = 0.65
DEFAULT_REFERENCE_COUNT = 12

st.set_page_config(page_title="SWE 599 Trajectory Diffusion Demo", layout="wide")


def list_checkpoints():
    checkpoints = sorted(DIFFUSION_DIR.glob("outputs/**/*.pt"))

    def checkpoint_priority(path):
        if "multi_shape_temporal_conv" in path.parts:
            return (0, str(path))
        if "temporal_conv_metrics" in path.parts:
            return (1, str(path))
        if "temporal_conv" in path.parts:
            return (2, str(path))
        if "start_goal" in path.parts:
            return (3, str(path))
        return (4, str(path))

    checkpoints = sorted(checkpoints, key=checkpoint_priority)
    if DEFAULT_CHECKPOINT.exists() and DEFAULT_CHECKPOINT not in checkpoints:
        checkpoints.insert(0, DEFAULT_CHECKPOINT)
    return checkpoints


@st.cache_data(show_spinner=False)
def load_checkpoint_metadata(checkpoint_path, modified_time):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {
        "shape_name": checkpoint.get("shape_name"),
        "shape_names": checkpoint.get("shape_names"),
        "shape_counts": checkpoint.get("shape_counts"),
        "seq_len": checkpoint.get("seq_len"),
        "timesteps": checkpoint.get("timesteps"),
        "hidden": checkpoint.get("hidden"),
        "architecture": checkpoint.get("architecture"),
        "conditioning": checkpoint.get("conditioning"),
        "cond_dim": checkpoint.get("cond_dim", 0),
    }


def checkpoint_shape_names(metadata):
    shape_names = metadata.get("shape_names")
    if shape_names:
        return list(shape_names)

    shape_name = metadata.get("shape_name")
    if shape_name and shape_name != "multi_shape":
        return [shape_name]

    return list(DEFAULT_TRAINING_SHAPES)


@st.cache_data(show_spinner=False)
def load_reference_trajectories(shape_names, seq_len):
    dataset = LASATrajectoryDataset(shape_names=list(shape_names), seq_len=seq_len)
    return dataset.raw_data.astype(np.float32), dataset.shape_labels.tolist(), dataset.shape_counts


def select_reference_preview(references, reference_labels, selected_shape_names, max_count):
    if len(references) <= max_count:
        return references, list(reference_labels)

    labels = np.asarray(reference_labels)
    grouped_indices = [np.flatnonzero(labels == shape).tolist() for shape in selected_shape_names]
    grouped_indices = [indices for indices in grouped_indices if indices]

    selected_indices = []
    offset = 0
    while len(selected_indices) < max_count:
        added_any = False
        for indices in grouped_indices:
            if offset < len(indices):
                selected_indices.append(indices[offset])
                added_any = True
                if len(selected_indices) == max_count:
                    break
        if not added_any:
            break
        offset += 1

    return references[selected_indices], [reference_labels[index] for index in selected_indices]


def extract_drawn_points(canvas_json):
    if not canvas_json:
        return None

    points = []
    for obj in canvas_json.get("objects", []):
        if obj.get("type") == "path":
            for command in obj.get("path", []):
                numeric_values = [value for value in command[1:] if isinstance(value, (int, float))]
                if len(numeric_values) >= 2:
                    points.append(numeric_values[-2:])
        elif obj.get("type") == "line":
            points.extend([[obj.get("x1", 0), obj.get("y1", 0)], [obj.get("x2", 0), obj.get("y2", 0)]])
        elif obj.get("type") in {"polyline", "polygon"}:
            for point in obj.get("points", []):
                points.append([point.get("x", 0) + obj.get("left", 0), point.get("y", 0) + obj.get("top", 0)])

    if len(points) < 2:
        return None

    points = np.asarray(points, dtype=np.float32)
    points[:, 1] = CANVAS_SIZE - points[:, 1]
    return points


def resample_points(points, seq_len):
    if points is None or len(points) < 2:
        return None

    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    distances = np.concatenate([[0.0], np.cumsum(segment_lengths)])

    if distances[-1] == 0:
        return np.repeat(points[:1], seq_len, axis=0)

    target = np.linspace(0.0, distances[-1], seq_len)
    x = np.interp(target, distances, points[:, 0])
    y = np.interp(target, distances, points[:, 1])
    return np.stack([x, y], axis=-1)


def map_drawing_to_sample_space(drawing, samples):
    drawing_min = drawing.min(axis=0)
    drawing_span = np.maximum(drawing.max(axis=0) - drawing_min, 1e-6)
    drawing_unit = (drawing - drawing_min) / drawing_span

    sample_min = samples.min(axis=(0, 1))
    sample_span = np.maximum(samples.max(axis=(0, 1)) - sample_min, 1e-6)
    return drawing_unit * sample_span + sample_min


def apply_drawing_guidance(samples, drawing, strength):
    if drawing is None or strength <= 0:
        return samples, None

    guide = map_drawing_to_sample_space(drawing, samples)
    blend_guide = smooth_trajectories(guide[None], sigma=2)[0]
    guided = ((1.0 - strength) * samples) + (strength * blend_guide[None, :, :])
    return guided, guide


def apply_start_goal_guidance(samples, start_goal, strength):
    if start_goal is None or strength <= 0:
        return samples

    start = np.asarray(start_goal[:2], dtype=np.float32)
    goal = np.asarray(start_goal[2:], dtype=np.float32)
    weights = np.linspace(0.0, 1.0, samples.shape[1], dtype=np.float32).reshape(1, -1, 1)
    target_line = ((1.0 - weights) * start.reshape(1, 1, 2)) + (weights * goal.reshape(1, 1, 2))

    correction_start = start.reshape(1, 1, 2) - samples[:, :1, :]
    correction_goal = goal.reshape(1, 1, 2) - samples[:, -1:, :]
    correction = ((1.0 - weights) * correction_start) + (weights * correction_goal)
    endpoint_guided = samples + correction

    return ((1.0 - strength) * samples) + (strength * (0.85 * endpoint_guided + 0.15 * target_line))


def smooth_trajectories(samples, sigma=4):
    """Apply Gaussian smoothing along the time axis of every trajectory.

    Uses edge padding so the endpoints are preserved instead of being pulled
    toward the origin by zero-padded convolution.
    """
    radius = int(3 * sigma) + 1
    kernel_size = 2 * radius + 1
    x = np.arange(kernel_size) - radius
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    smoothed = np.empty_like(samples)
    for i in range(samples.shape[0]):
        for d in range(samples.shape[2]):
            padded = np.pad(samples[i, :, d], radius, mode="edge")
            smoothed[i, :, d] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def trajectory_smoothness(trajectories):
    second_diff = np.diff(trajectories, n=2, axis=1)
    return float(np.mean(np.linalg.norm(second_diff, axis=2)))


def mean_curvature(trajectories):
    velocity = np.diff(trajectories, axis=1)
    acceleration = np.diff(velocity, axis=1)
    speed = np.linalg.norm(velocity[:, 1:, :], axis=2)
    cross = np.abs(
        velocity[:, 1:, 0] * acceleration[:, :, 1]
        - velocity[:, 1:, 1] * acceleration[:, :, 0]
    )
    curvature = cross / np.maximum(speed**3, 1e-6)
    return float(np.mean(curvature))


def nearest_reference_distance(samples, references):
    if references is None or len(references) == 0:
        return None

    if references.shape[1] != samples.shape[1]:
        resampled_references = [resample_points(trajectory, samples.shape[1]) for trajectory in references]
        references = np.stack([trajectory for trajectory in resampled_references if trajectory is not None])
        if len(references) == 0:
            return None

    sample_flat = samples.reshape(samples.shape[0], -1)
    ref_flat = references.reshape(references.shape[0], -1)
    distances = np.linalg.norm(sample_flat[:, None, :] - ref_flat[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def endpoint_error(samples, start_goal):
    if start_goal is None:
        return None

    start = np.asarray(start_goal[:2], dtype=np.float32)
    goal = np.asarray(start_goal[2:], dtype=np.float32)
    start_error = np.linalg.norm(samples[:, 0, :] - start.reshape(1, 2), axis=1)
    goal_error = np.linalg.norm(samples[:, -1, :] - goal.reshape(1, 2), axis=1)
    return float(np.mean((start_error + goal_error) / 2.0))


def trajectories_to_dataframe(trajectories, source, shape_labels=None):
    rows = []
    for sample_idx, trajectory in enumerate(trajectories):
        for t, point in enumerate(trajectory):
            row = {
                "source": source,
                "sample": sample_idx,
                "t": t,
                "x": float(point[0]),
                "y": float(point[1]),
            }
            if shape_labels is not None:
                row["shape"] = shape_labels[sample_idx]
            rows.append(row)
    return pd.DataFrame(rows)


def dataframe_info(dataframe):
    memory_mb = dataframe.memory_usage(deep=True).sum() / (1024 * 1024)
    return pd.DataFrame(
        [
            {"item": "rows", "value": len(dataframe)},
            {"item": "columns", "value": len(dataframe.columns)},
            {"item": "samples", "value": dataframe["sample"].nunique()},
            {"item": "time steps per sample", "value": dataframe["t"].nunique()},
            {"item": "x min", "value": dataframe["x"].min()},
            {"item": "x max", "value": dataframe["x"].max()},
            {"item": "y min", "value": dataframe["y"].min()},
            {"item": "y max", "value": dataframe["y"].max()},
            {"item": "memory MB", "value": round(memory_mb, 4)},
        ]
    )


def trajectory_summary_dataframe(trajectories, source, shape_labels=None):
    rows = []
    for sample_idx, trajectory in enumerate(trajectories):
        segment_lengths = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
        row = {
            "source": source,
            "sample": sample_idx,
            "points": len(trajectory),
            "start_x": float(trajectory[0, 0]),
            "start_y": float(trajectory[0, 1]),
            "goal_x": float(trajectory[-1, 0]),
            "goal_y": float(trajectory[-1, 1]),
            "path_length": float(segment_lengths.sum()),
            "displacement": float(displacement),
            "x_profile": trajectory[:, 0].round(3).tolist(),
            "y_profile": trajectory[:, 1].round(3).tolist(),
        }
        if shape_labels is not None:
            row["shape"] = shape_labels[sample_idx]
        rows.append(row)
    return pd.DataFrame(rows)


def visual_table_config():
    return {
        "x_profile": st.column_config.LineChartColumn(
            "x over time",
            help="Mini chart of the x-coordinate across trajectory timesteps.",
            width="medium",
        ),
        "y_profile": st.column_config.LineChartColumn(
            "y over time",
            help="Mini chart of the y-coordinate across trajectory timesteps.",
            width="medium",
        ),
        "path_length": st.column_config.NumberColumn("path length", format="%.2f"),
        "displacement": st.column_config.NumberColumn("displacement", format="%.2f"),
        "start_x": st.column_config.NumberColumn("start x", format="%.2f"),
        "start_y": st.column_config.NumberColumn("start y", format="%.2f"),
        "goal_x": st.column_config.NumberColumn("goal x", format="%.2f"),
        "goal_y": st.column_config.NumberColumn("goal y", format="%.2f"),
    }


def plot_dataframe_preview(trajectories, title, color):
    fig, ax = plt.subplots(figsize=(5, 4))
    for trajectory in trajectories:
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.75, linewidth=1.8)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    return fig


def checkpoint_history(metadata):
    train_loss = metadata.get("train_losses") or metadata.get("losses") or []
    return {
        "loss": {
            "train": train_loss,
            "validation": metadata.get("val_losses") or [],
            "title": "Noise Prediction Loss",
            "ylabel": "MSE",
        },
        "accuracy": {
            "train": metadata.get("train_accuracy") or [],
            "validation": metadata.get("val_accuracy") or [],
            "title": "Noise Direction Accuracy",
            "ylabel": "accuracy",
        },
        "f1": {
            "train": metadata.get("train_f1") or [],
            "validation": metadata.get("val_f1") or [],
            "title": "Noise Direction F1 Score",
            "ylabel": "F1",
        },
    }


def metric_history_dataframe(history):
    rows = []
    for metric_name, metric_data in history.items():
        for split_name in ["train", "validation"]:
            values = metric_data[split_name]
            for epoch, value in enumerate(values):
                rows.append(
                    {
                        "metric": metric_name,
                        "split": split_name,
                        "epoch": epoch,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def plot_metric_history(metric_data):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    has_data = False

    if metric_data["train"]:
        ax.plot(metric_data["train"], label="training", color="#2563eb", linewidth=2)
        has_data = True

    if metric_data["validation"]:
        ax.plot(metric_data["validation"], label="validation", color="#ef4444", linewidth=2)
        has_data = True

    ax.set_title(metric_data["title"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_data["ylabel"])
    ax.grid(True, alpha=0.25)
    if has_data:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "Not available in this checkpoint", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    return fig


def final_metric_value(values):
    return values[-1] if values else None


def samples_to_csv(samples):
    return trajectories_to_dataframe(samples, "generated").to_csv(index=False).encode("utf-8")


def figure_to_png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    buffer.seek(0)
    return buffer


def plot_trajectories(samples, references, guide, show_references, fixed_limits):
    fig, ax = plt.subplots(figsize=(7, 7))

    if show_references and references is not None:
        for idx, trajectory in enumerate(references):
            label = "real LASA" if idx == 0 else None
            ax.plot(trajectory[:, 0], trajectory[:, 1], color="#94a3b8", linewidth=1.5, alpha=0.7, label=label)

    for idx, trajectory in enumerate(samples):
        label = "generated" if idx == 0 else None
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="#2563eb", linewidth=2, alpha=0.85, label=label)

    if guide is not None:
        ax.plot(guide[:, 0], guide[:, 1], color="black", linewidth=3, linestyle="--", label="drawn guide")

    if fixed_limits:
        all_parts = [samples.reshape(-1, 2)]
        if show_references and references is not None:
            all_parts.append(references.reshape(-1, 2))
        if guide is not None:
            all_parts.append(guide.reshape(-1, 2))
        all_points = np.concatenate(all_parts, axis=0)
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max(maxs - mins) / 2.0
        radius = max(radius, 1e-6) * 1.15
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)

    ax.set_title("Trajectory Generation")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    return fig


st.title("SWE 599 Trajectory Diffusion Demo")
st.write(
    "Generate LASA-style 2D demonstration trajectories with a diffusion model, compare them "
    "with real demonstrations, and guide the displayed output with a sketch or start-goal pair."
)

with st.expander("How it works", expanded=False):
    st.markdown(
        """
        1. LASA handwriting demonstrations are resampled to a fixed sequence length.
        2. During training, Gaussian noise is added to normalized trajectories.
        3. A PyTorch denoising network learns to predict the added noise.
        4. Sampling starts from random noise and repeatedly denoises it into a trajectory.
        5. The reference overlays are loaded from the same LASA shapes stored in the selected checkpoint metadata.
        6. If a checkpoint was trained with start-goal conditioning, those values are passed into the model. Otherwise, this app applies geometric guidance after sampling.
        """
    )

checkpoint_options = list_checkpoints()
checkpoint_labels = [str(path.relative_to(ROOT)) for path in checkpoint_options]

with st.sidebar:
    st.header("Generation")
    if checkpoint_options:
        selected_label = st.selectbox("Checkpoint", checkpoint_labels, index=0)
        checkpoint_path = str(ROOT / selected_label)
    else:
        checkpoint_path = str(DEFAULT_CHECKPOINT)

    num_samples = st.slider("Generated trajectories", min_value=3, max_value=12, value=DEFAULT_NUM_SAMPLES)
    variation = st.slider("Variation", min_value=1, max_value=10, value=1)

    st.header("Guidance")
    drawing_strength = st.slider("Sketch influence", 0.0, 1.0, DEFAULT_DRAWING_STRENGTH, 0.05)

    st.header("Display")
    show_references = st.checkbox("Overlay real LASA demonstrations", value=True)
    generate = st.button("Generate", type="primary")

checkpoint = Path(checkpoint_path)

if not checkpoint.exists():
    st.error(f"Checkpoint not found: {checkpoint}")
    st.stop()

preview_metadata = load_checkpoint_metadata(str(checkpoint), checkpoint.stat().st_mtime)
trained_shape_names = checkpoint_shape_names(preview_metadata)
references, reference_labels, shape_counts = load_reference_trajectories(tuple(trained_shape_names), 256)

with st.sidebar:
    st.header("Training Shapes")
    st.caption("Choose which trained LASA shapes to show as real reference overlays.")
    selected_shape_names = []
    for shape in trained_shape_names:
        demo_count = int(shape_counts.get(shape, 0))
        if st.checkbox(f"{shape} ({demo_count})", value=True, key=f"shape_filter_{checkpoint}_{shape}"):
            selected_shape_names.append(shape)

    if not selected_shape_names:
        st.warning("Select at least one training shape. Showing all shapes for now.")
        selected_shape_names = list(trained_shape_names)

reference_mask = np.isin(np.asarray(reference_labels), selected_shape_names)
filtered_references = references[reference_mask]
filtered_reference_labels = [label for label, keep in zip(reference_labels, reference_mask) if keep]
if len(filtered_references) == 0:
    filtered_references = references
    filtered_reference_labels = list(reference_labels)

with st.sidebar:
    max_references = min(DEFAULT_REFERENCE_COUNT, len(filtered_references))
    st.caption(f"Showing {max_references} reference demonstrations from {len(selected_shape_names)} shape(s).")

reference_preview, reference_label_preview = select_reference_preview(
    filtered_references,
    filtered_reference_labels,
    selected_shape_names,
    max_references,
)
default_start = reference_preview[0, 0, :]
default_goal = reference_preview[0, -1, :]

draw_col, condition_col, plot_col, info_col = st.columns([1, 0.8, 1.45, 0.8])

with draw_col:
    st.subheader("Draw")
    st.write("Sketch one continuous path to guide the displayed generated trajectories.")

    if st_canvas is None:
        st.warning("Install `streamlit-drawable-canvas` to enable mouse drawing.")
        drawing = None
    else:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=4,
            stroke_color="#2563eb",
            background_color="#ffffff",
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            key="trajectory_canvas",
        )
        drawing = extract_drawn_points(canvas_result.json_data)

    if drawing is None:
        st.caption("No drawing detected.")
    else:
        st.caption(f"Captured {len(drawing)} drawing points.")

with condition_col:
    st.subheader("Start / Goal")

    if drawing is not None:
        guide_for_condition = map_drawing_to_sample_space(resample_points(drawing, 2), reference_preview)
        start = guide_for_condition[0]
        goal = guide_for_condition[-1]
        st.caption("Using endpoints from the sketch.")
    else:
        start = default_start.astype(np.float32)
        goal = default_goal.astype(np.float32)
        st.caption("Using endpoints from the first reference demonstration.")

    st.dataframe(
        pd.DataFrame(
            [
                {"point": "start", "x": float(start[0]), "y": float(start[1])},
                {"point": "goal", "x": float(goal[0]), "y": float(goal[1])},
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    start_goal = np.concatenate([start, goal]).astype(np.float32)

checkpoint_changed = st.session_state.get("checkpoint_path") != str(checkpoint)
settings = (str(checkpoint), int(num_samples), int(variation), tuple(selected_shape_names))
settings_changed = st.session_state.get("generation_settings") != settings

if generate or "samples" not in st.session_state or checkpoint_changed or settings_changed:
    with st.spinner("Generating trajectories..."):
        samples, metadata = generate_samples(
            checkpoint_path=checkpoint,
            num_samples=num_samples,
            timesteps=DEFAULT_TIMESTEPS,
            seed=int(variation),
            condition=start_goal,
        )
    samples = smooth_trajectories(samples)
    st.session_state["samples"] = samples
    st.session_state["metadata"] = metadata
    st.session_state["start_goal"] = start_goal
    st.session_state["checkpoint_path"] = str(checkpoint)
    st.session_state["generation_settings"] = settings
else:
    samples = st.session_state["samples"]
    metadata = st.session_state["metadata"]
    start_goal = st.session_state.get("start_goal", start_goal)

resampled_drawing = resample_points(drawing, samples.shape[1]) if drawing is not None else None
display_samples, guide = apply_drawing_guidance(samples, resampled_drawing, drawing_strength)

true_conditioning = metadata.get("cond_dim", 0) > 0 and start_goal is not None
if start_goal is not None and not true_conditioning:
    display_samples = apply_start_goal_guidance(display_samples, start_goal, DEFAULT_START_GOAL_STRENGTH)

with plot_col:
    fig = plot_trajectories(display_samples, reference_preview, guide, show_references, fixed_limits=True)
    st.pyplot(fig)

    png_buffer = figure_to_png(fig)
    st.download_button(
        "Download plot PNG",
        data=png_buffer,
        file_name="trajectory_generation.png",
        mime="image/png",
    )
    st.download_button(
        "Download trajectories CSV",
        data=samples_to_csv(display_samples),
        file_name="generated_trajectories.csv",
        mime="text/csv",
    )

with info_col:
    st.subheader("Metrics")
    endpoint = endpoint_error(display_samples, start_goal)
    nearest = nearest_reference_distance(display_samples, reference_preview)

    st.metric("Generated", len(display_samples))
    st.metric("Smoothness", f"{trajectory_smoothness(display_samples):.4f}")
    st.metric("Curvature", f"{mean_curvature(display_samples):.4f}")
    if endpoint is not None:
        st.metric("Endpoint error", f"{endpoint:.4f}")
    if nearest is not None:
        st.metric("Nearest demo dist.", f"{nearest:.4f}")

    st.subheader("Model")
    history = checkpoint_history(metadata)
    final_loss = final_metric_value(history["loss"]["train"])
    final_val_loss = final_metric_value(history["loss"]["validation"])
    st.json(
        {
            "shape_name": metadata.get("shape_name", "unknown"),
            "shape_names": checkpoint_shape_names(metadata),
            "seq_len": metadata.get("seq_len", samples.shape[1]),
            "timesteps": metadata.get("timesteps", "unknown"),
            "hidden": metadata.get("hidden", "unknown"),
            "conditioning": metadata.get("conditioning", "none"),
            "condition_used": "learned" if true_conditioning else "geometric" if start_goal is not None else "none",
            "final_loss": final_loss,
            "final_val_loss": final_val_loss,
            "final_accuracy": final_metric_value(history["accuracy"]["train"]),
            "final_val_accuracy": final_metric_value(history["accuracy"]["validation"]),
            "final_f1": final_metric_value(history["f1"]["train"]),
            "final_val_f1": final_metric_value(history["f1"]["validation"]),
        }
    )

training_metrics_tab, generated_data_tab, reference_data_tab = st.tabs(
    ["Training Metrics", "Generated Data", "Reference Data"]
)

with training_metrics_tab:
    st.subheader("Training and Validation Curves")
    st.write(
        "Loss is the main diffusion training metric. Accuracy and F1 are proxy "
        "classification metrics computed by checking whether the denoiser predicts "
        "the correct sign of each noise coordinate."
    )

    history = checkpoint_history(metadata)
    metric_definitions = metadata.get("metric_definitions") or {
        "loss": "Mean squared error between predicted and true diffusion noise.",
        "accuracy": "Proxy metric: fraction of noise coordinates with the correct predicted sign.",
        "f1": "Proxy metric: F1 score after binarizing each noise coordinate by sign.",
    }

    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "Final train loss",
        f"{final_metric_value(history['loss']['train']):.4f}" if history["loss"]["train"] else "n/a",
    )
    metric_cols[1].metric(
        "Final val accuracy",
        f"{final_metric_value(history['accuracy']['validation']):.4f}"
        if history["accuracy"]["validation"]
        else "n/a",
    )
    metric_cols[2].metric(
        "Final val F1",
        f"{final_metric_value(history['f1']['validation']):.4f}" if history["f1"]["validation"] else "n/a",
    )

    if not history["loss"]["validation"] or not history["accuracy"]["train"] or not history["f1"]["train"]:
        st.info(
            "This checkpoint was trained before the full metric history was added, "
            "so some plots may show only training loss. Retrain with the updated "
            "`train.py` to store validation loss, accuracy, and F1 curves."
        )

    loss_col, acc_col, f1_col = st.columns(3)
    with loss_col:
        st.pyplot(plot_metric_history(history["loss"]))
    with acc_col:
        st.pyplot(plot_metric_history(history["accuracy"]))
    with f1_col:
        st.pyplot(plot_metric_history(history["f1"]))

    st.write("Metric definitions")
    st.dataframe(
        pd.DataFrame(
            [{"metric": name, "definition": definition} for name, definition in metric_definitions.items()]
        ),
        width="stretch",
        hide_index=True,
    )

    history_df = metric_history_dataframe(history)
    st.write("Metric history DataFrame")
    if history_df.empty:
        st.warning("No training metric history was found in this checkpoint.")
    else:
        st.dataframe(history_df, width="stretch", height=360)
        st.download_button(
            "Download metric history CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="training_metric_history.csv",
            mime="text/csv",
        )

with generated_data_tab:
    st.subheader("Generated Trajectory DataFrame")
    st.write(
        "Each row is one point from one displayed generated trajectory. "
        "`sample` identifies which trajectory the point belongs to, `t` is the "
        "time-step index, and `x`, `y` are the 2D coordinates."
    )

    generated_df = trajectories_to_dataframe(display_samples, "generated")
    generated_summary_df = trajectory_summary_dataframe(display_samples, "generated")

    info_col_a, info_col_b, info_col_c, info_col_d = st.columns(4)
    info_col_a.metric("Rows", len(generated_df))
    info_col_b.metric("Columns", len(generated_df.columns))
    info_col_c.metric("Samples", generated_df["sample"].nunique())
    info_col_d.metric("Steps", generated_df["t"].nunique())

    visual_col, chart_col = st.columns([1.45, 1])
    with visual_col:
        st.write("Visual trajectory summary")
        st.dataframe(
            generated_summary_df,
            width="stretch",
            height=300,
            column_config=visual_table_config(),
        )
    with chart_col:
        st.write("Trajectory preview")
        st.pyplot(plot_dataframe_preview(display_samples, "Generated Trajectories", "#2563eb"))

    st.write("Point-level DataFrame")
    st.dataframe(generated_df, width="stretch", height=360)

    st.write("DataFrame summary")
    st.dataframe(dataframe_info(generated_df), width="stretch")

    st.write("Coordinate statistics")
    st.dataframe(generated_df[["x", "y"]].describe(), width="stretch")

    st.download_button(
        "Download displayed DataFrame CSV",
        data=generated_df.to_csv(index=False).encode("utf-8"),
        file_name="displayed_generated_trajectories.csv",
        mime="text/csv",
    )

with reference_data_tab:
    st.subheader("Real LASA Training-Shape Reference DataFrame")
    st.write(
        "This table shows real LASA demonstrations from the shapes stored in the selected "
        "checkpoint. Checkbox selections in the sidebar control which shape references "
        "are used for the gray overlay and for "
        "nearest-demonstration comparison."
    )

    shape_count_df = pd.DataFrame(
        [
            {
                "shape": shape,
                "demos": int(shape_counts.get(shape, 0)),
                "shown": shape in selected_shape_names,
            }
            for shape in trained_shape_names
        ]
    )
    st.write("Training shapes represented by the selected checkpoint")
    st.dataframe(shape_count_df, width="stretch", hide_index=True)

    reference_df = trajectories_to_dataframe(reference_preview, "real_lasa", reference_label_preview)
    reference_summary_df = trajectory_summary_dataframe(reference_preview, "real_lasa", reference_label_preview)

    ref_col_a, ref_col_b, ref_col_c, ref_col_d = st.columns(4)
    ref_col_a.metric("Rows", len(reference_df))
    ref_col_b.metric("Columns", len(reference_df.columns))
    ref_col_c.metric("Demos", reference_df["sample"].nunique())
    ref_col_d.metric("Steps", reference_df["t"].nunique())

    ref_visual_col, ref_chart_col = st.columns([1.45, 1])
    with ref_visual_col:
        st.write("Visual demonstration summary")
        st.dataframe(
            reference_summary_df,
            width="stretch",
            height=300,
            column_config=visual_table_config(),
        )
    with ref_chart_col:
        st.write("Demonstration preview")
        st.pyplot(plot_dataframe_preview(reference_preview, "Real LASA Demonstrations", "#94a3b8"))

    st.write("Point-level DataFrame")
    st.dataframe(reference_df, width="stretch", height=360)

    st.write("DataFrame summary")
    st.dataframe(dataframe_info(reference_df), width="stretch")

    st.write("Coordinate statistics")
    st.dataframe(reference_df[["x", "y"]].describe(), width="stretch")
