import numpy as np


def generate_single_trajectory(num_steps: int = 60, noise_scale: float = 0.015):
    """Generate one synthetic 2D trajectory from start to goal with curvature."""
    start = np.random.uniform(0.0, 2.0, size=2)
    goal = np.random.uniform(8.0, 10.0, size=2)

    t = np.linspace(0.0, 1.0, num_steps)
    base = (1 - t)[:, None] * start + t[:, None] * goal

    direction = goal - start
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        direction = np.array([1.0, 0.0])
        norm = 1.0

    unit_dir = direction / norm
    perp = np.array([-unit_dir[1], unit_dir[0]])

    curve_strength = np.random.uniform(0.2, 1.2)
    curve_direction = np.random.choice([-1.0, 1.0])
    arc_profile = np.sin(np.pi * t)

    curved = base + (curve_direction * curve_strength * arc_profile)[:, None] * perp
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=curved.shape)
    trajectory = curved + noise

    trajectory[0] = start
    trajectory[-1] = goal

    return trajectory, start, goal, curve_strength, curve_direction


def generate_dataset(num_trajectories: int = 500, num_steps: int = 60):
    trajectories = []
    starts = []
    goals = []
    curve_strengths = []
    curve_directions = []

    for _ in range(num_trajectories):
        traj, start, goal, strength, direction = generate_single_trajectory(num_steps=num_steps)
        trajectories.append(traj)
        starts.append(start)
        goals.append(goal)
        curve_strengths.append(strength)
        curve_directions.append(direction)

    return {
        "trajectories": np.array(trajectories, dtype=np.float32),
        "starts": np.array(starts, dtype=np.float32),
        "goals": np.array(goals, dtype=np.float32),
        "curve_strengths": np.array(curve_strengths, dtype=np.float32),
        "curve_directions": np.array(curve_directions, dtype=np.float32),
    }


def main():
    np.random.seed(42)
    dataset = generate_dataset(num_trajectories=500, num_steps=60)
    output_path = "synthetic_trajectories.npz"
    np.savez(output_path, **dataset)

    print(f"Saved dataset to {output_path}")
    print(f"Trajectories shape: {dataset['trajectories'].shape}")
    print(f"Starts shape: {dataset['starts'].shape}")
    print(f"Goals shape: {dataset['goals'].shape}")


if __name__ == "__main__":
    main()
