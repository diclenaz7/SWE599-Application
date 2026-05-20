import numpy as np
import torch
from torch.utils.data import Dataset
import pyLasaDataset as lasa


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


def normalize_shape_names(shape_name="Angle", shape_names=None):
    if shape_names is None:
        return [shape_name]

    if isinstance(shape_names, str):
        shape_names = [name.strip() for name in shape_names.split(",")]

    shape_names = [name for name in shape_names if name]
    if not shape_names:
        raise ValueError("At least one LASA shape name is required.")

    return shape_names


class LASATrajectoryDataset(Dataset):
    def __init__(self, shape_name="Angle", shape_names=None, seq_len=256, conditioning="none"):
        self.conditioning = conditioning
        self.shape_names = normalize_shape_names(shape_name=shape_name, shape_names=shape_names)

        trajectories = []
        shape_labels = []
        shape_counts = {}

        for current_shape in self.shape_names:
            data = getattr(lasa.DataSet, current_shape)
            shape_counts[current_shape] = len(data.demos)

            for demo in data.demos:
                pos = demo.pos.T  # original: (2, T), convert to (T, 2)

                # resample to fixed length
                old_idx = np.linspace(0, 1, pos.shape[0])
                new_idx = np.linspace(0, 1, seq_len)

                x = np.interp(new_idx, old_idx, pos[:, 0])
                y = np.interp(new_idx, old_idx, pos[:, 1])

                traj = np.stack([x, y], axis=-1)  # (seq_len, 2)
                trajectories.append(traj)
                shape_labels.append(current_shape)

        self.raw_data = np.stack(trajectories).astype(np.float32)
        self.shape_labels = np.asarray(shape_labels)
        self.shape_counts = shape_counts

        self.mean = self.raw_data.mean(axis=(0, 1), keepdims=True)
        self.std = self.raw_data.std(axis=(0, 1), keepdims=True) + 1e-8

        self.data = (self.raw_data - self.mean) / self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory = torch.tensor(self.data[idx])

        if self.conditioning == "start-goal":
            condition = torch.cat([trajectory[0], trajectory[-1]])
            return trajectory, condition

        return trajectory
