import numpy as np
import torch
from torch.utils.data import Dataset
import pyLasaDataset as lasa


class LASATrajectoryDataset(Dataset):
    def __init__(self, shape_name="Angle", seq_len=256):
        data = getattr(lasa.DataSet, shape_name)

        trajectories = []

        for demo in data.demos:
            pos = demo.pos.T  # original: (2, T), convert to (T, 2)

            # resample to fixed length
            old_idx = np.linspace(0, 1, pos.shape[0])
            new_idx = np.linspace(0, 1, seq_len)

            x = np.interp(new_idx, old_idx, pos[:, 0])
            y = np.interp(new_idx, old_idx, pos[:, 1])

            traj = np.stack([x, y], axis=-1)  # (seq_len, 2)
            trajectories.append(traj)

        self.data = np.stack(trajectories).astype(np.float32)

        self.mean = self.data.mean(axis=(0, 1), keepdims=True)
        self.std = self.data.std(axis=(0, 1), keepdims=True) + 1e-8

        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])