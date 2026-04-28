import torch
import matplotlib.pyplot as plt

from model import TrajectoryDenoiser
from diffusion import Diffusion


device = "cuda" if torch.cuda.is_available() else "cpu"

seq_len = 256
num_samples = 10

checkpoint = torch.load("outputs/lasa_diffusion.pt", map_location=device)

model = TrajectoryDenoiser(seq_len=seq_len).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

mean = checkpoint["mean"]
std = checkpoint["std"]

diffusion = Diffusion(device=device)

samples = diffusion.sample(model, shape=(num_samples, seq_len, 2))
samples = samples.cpu().numpy()

samples = samples * std + mean

for traj in samples:
    plt.plot(traj[:, 0], traj[:, 1])

plt.title("Generated LASA-like Trajectories")
plt.axis("equal")
plt.savefig("outputs/generated_trajectories.png")
plt.show()