# Synthetic Trajectory Starter Project

A tiny Python starter project for generating, saving, loading, and visualizing synthetic 2D robot-like trajectories.

## What is included?

- `generate_dataset.py` — creates a synthetic trajectory dataset and saves it as `.npz`
- `visualize_trajectories.py` — loads the dataset and plots sample trajectories
- `inspect_dataset.py` — prints dataset shape and a few examples
- `requirements.txt` — basic dependencies

## Project idea

This project simulates **demonstration trajectories** for a point robot moving in 2D from a start point to a goal point. Some trajectories curve left, some curve right, and a little noise is added so the data looks more realistic.

Each trajectory has shape:

- `(T, 2)` where:
  - `T` = number of time steps
  - `2` = `(x, y)` coordinates

The full dataset has shape:

- `(N, T, 2)` where:
  - `N` = number of trajectories

## 1. Create a project folder

You can either use the files in this folder directly, or recreate them in your own local folder.

Example:

```bash
mkdir trajectory_project
cd trajectory_project
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate   # Windows
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy matplotlib
```

## 3. Generate the dataset

```bash
python generate_dataset.py
```

This will create:

- `synthetic_trajectories.npz`

## 4. Inspect the dataset

```bash
python inspect_dataset.py
```

This will show:

- dataset shape
- one trajectory shape
- example start/goal points

## 5. Visualize sample trajectories

```bash
python visualize_trajectories.py
```

A plot window will open showing sample trajectories.

## Data format

The saved `.npz` file contains these arrays:

- `trajectories`: shape `(N, T, 2)`
- `starts`: shape `(N, 2)`
- `goals`: shape `(N, 2)`
- `curve_strengths`: shape `(N,)`
- `curve_directions`: shape `(N,)`

## How the synthetic trajectories are generated

Each path is built using:

1. A start point
2. A goal point
3. Linear interpolation between them
4. A perpendicular curve offset to create an arc-like path
5. Small Gaussian noise for realism

This gives you a simple but useful imitation-learning-style dataset.

## Good starting points for your project

### Starting point 1: Just understand the data

- generate the data
- plot 10–20 trajectories
- check how shape `(N, T, 2)` works

### Starting point 2: Add labels or conditions

You can condition later on:

- start point
- goal point
- curve direction
- obstacle layout

### Starting point 3: Build a very simple baseline

Before diffusion, try:

- linear interpolation baseline
- nearest-neighbor retrieval of similar trajectories
- simple MLP that reconstructs trajectories

### Starting point 4: Prepare for diffusion

Your model input can be:

- flattened trajectory: `(T * 2,)`
- or sequence form: `(T, 2)`

A first easy choice is to flatten trajectories and use an MLP.
