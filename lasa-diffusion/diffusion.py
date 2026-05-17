import torch


class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)

        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)

        noisy = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

        return noisy, noise

    @torch.no_grad()
    def sample(self, model, shape, condition=None):
        x = torch.randn(shape).to(self.device)
        if condition is not None:
            condition = condition.to(self.device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)

            predicted_noise = model(x, t_batch, condition=condition)

            beta = self.betas[t]
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise
            ) + torch.sqrt(beta) * noise

        return x
