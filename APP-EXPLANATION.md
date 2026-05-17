# What Is Happening in the Web App

This document explains the SWE 599 trajectory diffusion demo in simple terms.

## Big Picture

The app demonstrates a diffusion-based trajectory generation pipeline.

It combines:

- a trained diffusion model
- real LASA handwriting trajectories
- user drawing input
- start/goal guidance
- visualization
- metrics
- DataFrame inspection

The main purpose of the app is to show how a model can generate 2D robot-like
demonstration trajectories and how those generated trajectories compare with real
demonstrations.

## The Actual Diffusion Model

The diffusion model was trained on LASA handwriting trajectories.

During training, the model saw clean 2D paths. Noise was added to those paths,
and the model learned to predict and remove that noise.

When the user presses **Generate**, the model starts from random noise and tries
to turn it into a trajectory.

The blue lines in the plot are the model's generated trajectories.

## The Drawing Input

The drawing canvas lets the user sketch a trajectory with the mouse.

The app extracts points from the drawing and resamples them into a trajectory-like
sequence.

The drawing is useful because it gives the app a user-provided trajectory shape
or direction. However, the drawing does not always mean the model fully understands
the sketch as an instruction.

There are two cases:

1. If the selected checkpoint was trained with start/goal conditioning, the app
   can pass the start and goal values into the model.
2. If the selected checkpoint is unconditional, the app applies geometric guidance
   after sampling.

So the drawing helps guide the result, but it may not completely control the
generated trajectories.

## What the Plot Shows

The plot contains three possible trajectory types:

- **Gray lines**: real LASA demonstrations from the dataset
- **Blue lines**: generated trajectories from the diffusion model
- **Black dashed line**: the user's drawn guide

Ideally, the blue generated trajectories should look smooth and similar to the
real LASA trajectories. If guidance is enabled, the generated trajectories should
also move toward the drawing or the selected start and goal points.

## Why the Generated Paths May Not Follow the Drawing Perfectly

The current main model is stronger than the first version. It uses a compact
temporal-convolution denoising network, so the trajectory is treated as an
ordered sequence instead of only as one flattened vector.

However, the model still has limited training data. A single LASA shape usually
contains only a small number of demonstrations, so the model can still produce
noisy, overly averaged, or blob-like trajectories when the sampling settings are
too fast or the requested drawing is far from the training examples.

A true sketch-conditioned model would be trained like this:

```text
input: user sketch / start point / goal point
output: trajectory that follows that condition
```

The current app supports start/goal conditioning when a compatible checkpoint is
selected, but full sketch conditioning would require a stronger training setup.

## What the Controls Do

## Samples

Controls how many blue generated trajectories are shown.

More samples gives more variety, but it can also make the plot crowded. For a
clearer visual result, 3 to 5 samples are often easier to inspect than 10 or more.

## Sampling Speed

Controls how many denoising steps are used.

- **Fast** uses fewer denoising steps. It is quicker but usually lower quality.
- **Balanced** is a middle option.
- **Full** uses more denoising steps. It is slower but usually gives better
  trajectories.

## Drawing Guide Strength

Controls how strongly the drawn sketch influences the displayed generated paths.

A higher value makes the generated display move closer to the drawn guide.

## Use Start-Goal Guidance

Uses a start point and goal point to guide generation.

The start and goal can come from:

- the endpoints of the drawing
- manually entered coordinate values

If the checkpoint supports learned start/goal conditioning, the start and goal
are passed into the model. Otherwise, the app applies geometric correction after
sampling.

## Overlay Real LASA Demonstrations

Shows real trajectories from the LASA dataset as gray lines.

This helps compare generated trajectories against the data that the model is
trying to imitate.

## Metrics

The metrics provide simple numerical summaries of the displayed generated
trajectories.

The app also has a **Training Metrics** tab. This tab reads the selected
checkpoint and plots the training history saved during model training.

The training plots are:

- **loss**: mean squared error between predicted and true diffusion noise
- **accuracy**: proxy score for whether the predicted noise direction has the
  correct sign
- **F1 score**: proxy F1 score after converting noise coordinates into positive
  or negative classes

Loss is the most important metric because the model is solving a continuous
denoising problem. Accuracy and F1 are helpful diagnostics for the report, but
they are not the same as classification accuracy in a normal classifier.

Older checkpoints may only show training loss because validation accuracy and F1
were added later.

## Smoothness

Measures how much the trajectory changes direction or acceleration between
neighboring points.

Lower values usually mean smoother trajectories.

## Curvature

Measures how sharply the path bends.

Higher curvature means sharper turns.

## Endpoint Error

Measures how far the generated trajectories are from the desired start and goal
points.

Lower endpoint error means the trajectories better respect the selected start and
goal.

## Nearest Demonstration Distance

Measures how close the generated trajectories are to the nearest real LASA
demonstrations shown in the app.

Lower values mean the generated paths are more similar to the displayed real
examples.

## DataFrame Tabs

The app also includes DataFrame tabs for inspecting trajectory data numerically.

## Generated Data

This tab shows the generated trajectories as a table.

Each row is one point from one trajectory.

The columns are:

- `source`: whether the row comes from generated or real data
- `sample`: which trajectory the point belongs to
- `t`: the timestep index in the trajectory
- `x`: x-coordinate
- `y`: y-coordinate

## Reference Data

This tab shows the real LASA trajectories currently used as reference overlays.

It is useful for comparing generated coordinate ranges and statistics with the
real demonstration data.

## What Is Good About the App

The app demonstrates the complete project workflow:

```text
LASA dataset
-> diffusion model
-> generated trajectories
-> user guidance
-> visualization
-> metrics
-> data inspection
```

It is useful for showing both the implementation and the current limitations of
the model.

## What Is Still Weak

The app works as an interactive demo, but the model quality can still be improved.

Current limitations include:

- the model is still compact
- the dataset for each LASA shape is small
- full sketch conditioning is not yet learned
- fast preview sampling can reduce trajectory quality
- generated trajectories may be noisy

## Possible Future Improvements

Useful next steps would be:

- condition directly on the full user sketch
- train on multiple LASA shapes
- add better quantitative evaluation
- improve trajectory smoothness
- compare the temporal-conv checkpoint against the older MLP checkpoint
- compare generated trajectories with nearest real demonstrations more clearly

These improvements would make the generated paths cleaner and make the drawing
input more meaningful.
