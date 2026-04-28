import torch
import torch.nn as nn


class TrajectoryDenoiser(nn.Module):
    def __init__(self, seq_len=256, dim=2, hidden=256):
        super().__init__()
        input_dim = seq_len * dim

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, input_dim),
        )

        self.seq_len = seq_len
        self.dim = dim

    def forward(self, x, t):
        b = x.shape[0]

        x_flat = x.reshape(b, -1)
        t = t.float().view(b, 1) / 1000.0
        t_emb = self.time_embed(t)

        inp = torch.cat([x_flat, t_emb], dim=1)
        out = self.net(inp)

        return out.reshape(b, self.seq_len, self.dim)