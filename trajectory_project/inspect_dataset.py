import numpy as np


def main():
    data = np.load("synthetic_trajectories.npz")

    trajectories = data["trajectories"]
    starts = data["starts"]
    goals = data["goals"]
    strengths = data["curve_strengths"]
    directions = data["curve_directions"]

    print("Dataset keys:", list(data.keys()))
    print("Trajectories shape:", trajectories.shape)
    print("Single trajectory shape:", trajectories[0].shape)
    print()

    for i in range(3):
        print(f"Example {i}")
        print("  Start:", starts[i])
        print("  Goal:", goals[i])
        print("  Curve strength:", strengths[i])
        print("  Curve direction:", directions[i])
        print("  First 3 points:")
        print(trajectories[i][:3])
        print("  Last 3 points:")
        print(trajectories[i][-3:])
        print()


if __name__ == "__main__":
    main()
