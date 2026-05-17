import torch
import torch.nn as nn


class TemporalConvBlock(nn.Module):
    def __init__(self, hidden, dilation=1):
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=padding, dilation=dilation),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=padding, dilation=dilation),
            nn.GroupNorm(8, hidden),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class TrajectoryDenoiser(nn.Module):
    def __init__(self, seq_len=256, dim=2, hidden=256, cond_dim=0, architecture="temporal-conv"):
        super().__init__()
        input_dim = seq_len * dim
        self.cond_dim = cond_dim
        self.architecture = architecture

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.cond_embed = nn.Linear(cond_dim, hidden) if cond_dim > 0 else None

        if architecture == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim + hidden + cond_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, input_dim),
            )
        elif architecture == "temporal-conv":
            self.input_proj = nn.Conv1d(dim, hidden, kernel_size=1)
            self.net = nn.Sequential(
                TemporalConvBlock(hidden, dilation=1),
                TemporalConvBlock(hidden, dilation=2),
                TemporalConvBlock(hidden, dilation=4),
                TemporalConvBlock(hidden, dilation=8),
            )
            self.output_proj = nn.Conv1d(hidden, dim, kernel_size=1)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.dim = dim

    def forward(self, x, t, condition=None):
        b = x.shape[0]
        t = t.float().view(b, 1) / 1000.0
        t_emb = self.time_embed(t)

        if self.architecture == "mlp":
            return self.forward_mlp(x, t_emb, condition)

        return self.forward_temporal_conv(x, t_emb, condition)

    def forward_mlp(self, x, t_emb, condition=None):
        b = x.shape[0]
        x_flat = x.reshape(b, -1)

        inputs = [x_flat, t_emb]
        if self.cond_dim > 0:
            if condition is None:
                condition = torch.zeros((b, self.cond_dim), device=x.device, dtype=x.dtype)
            inputs.append(condition)

        inp = torch.cat(inputs, dim=1)
        out = self.net(inp)

        return out.reshape(b, self.seq_len, self.dim)

    def forward_temporal_conv(self, x, t_emb, condition=None):
        b = x.shape[0]
        h = self.input_proj(x.transpose(1, 2))
        h = h + t_emb.unsqueeze(-1)

        if self.cond_embed is not None:
            if condition is None:
                condition = torch.zeros((b, self.cond_dim), device=x.device, dtype=x.dtype)
            h = h + self.cond_embed(condition).unsqueeze(-1)

        h = self.net(h)
        return self.output_proj(h).transpose(1, 2)
