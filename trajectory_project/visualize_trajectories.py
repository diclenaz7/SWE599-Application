import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.load("synthetic_trajectories.npz")
    trajectories = data["trajectories"]
    starts = data["starts"]
    goals = data["goals"]

    num_to_plot = 20
    indices = np.linspace(0, len(trajectories) - 1, num_to_plot, dtype=int)

    plt.figure(figsize=(8, 6))

    for idx in indices:
        traj = trajectories[idx]
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.8)

    plt.scatter(starts[indices, 0], starts[indices, 1], marker="o", s=40, label="starts")
    plt.scatter(goals[indices, 0], goals[indices, 1], marker="x", s=50, label="goals")

    plt.title("Sample Synthetic 2D Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
