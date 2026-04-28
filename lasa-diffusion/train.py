import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import LASATrajectoryDataset
from model import TrajectoryDenoiser
from diffusion import Diffusion


device = "cuda" if torch.cuda.is_available() else "cpu"

seq_len = 256
batch_size = 7
epochs = 3000

dataset = LASATrajectoryDataset(shape_name="Angle", seq_len=seq_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TrajectoryDenoiser(seq_len=seq_len).to(device)
diffusion = Diffusion(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

losses = []

for epoch in tqdm(range(epochs)):
    for x0 in loader:
        x0 = x0.to(device)

        t = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=device)

        noisy_x, noise = diffusion.add_noise(x0, t)
        predicted_noise = model(noisy_x, t)

        loss = loss_fn(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses.append(loss.item())

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save({
    "model": model.state_dict(),
    "mean": dataset.mean,
    "std": dataset.std,
}, "outputs/lasa_diffusion.pt")

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.savefig("outputs/loss.png")
plt.show()