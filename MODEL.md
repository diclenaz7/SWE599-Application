# Model Training Details

This document describes the diffusion model currently implemented in
`lasa-diffusion/` for the SWE 599 trajectory generation demo.

## Objective

The goal is to learn a generative model over 2D demonstration trajectories from
the LASA Handwriting Dataset. The trained model should generate LASA-like
trajectories by starting from random Gaussian noise and iteratively denoising it.

The implementation follows a simple DDPM-style noise-prediction setup:

1. Normalize real trajectories.
2. Add Gaussian noise at a randomly selected diffusion timestep.
3. Train a neural network to predict the noise that was added.
4. At inference time, repeatedly apply the learned denoiser from pure noise.

## Dataset

Dataset loader:

- `lasa-diffusion/dataset.py`

Class:

- `LASATrajectoryDataset`

Default dataset configuration:

```text
shape_name = "Angle"
seq_len = 128
trajectory_dim = 2
```

Each LASA demonstration is originally loaded as 2D position data. The loader:

1. Reads one LASA shape from `pyLasaDataset`.
2. Converts each demonstration from shape `(2, T)` to `(T, 2)`.
3. Resamples each trajectory to a fixed length of `128` points by default.
4. Stacks all demonstrations into an array of shape `(N, 128, 2)`.
5. Computes global dataset mean and standard deviation over trajectories and time.
6. Normalizes the trajectories before training.

The checkpoint stores the dataset `mean` and `std` so generated samples can be
converted back to the original coordinate scale.

## Trajectory Representation

The model represents a trajectory as a fixed-length sequence:

```text
x = [(x_1, y_1), (x_2, y_2), ..., (x_128, y_128)]
```

Tensor shape:

```text
(batch_size, seq_len, 2)
```

The current default model keeps the sequence dimension while denoising, so nearby
trajectory points can influence each other through temporal convolution layers.

## Model Architecture

Model file:

- `lasa-diffusion/model.py`

Class:

- `TrajectoryDenoiser`

The current default denoising model is a temporal convolutional network. It
receives:

- A noisy trajectory `x_t`
- A diffusion timestep `t`
- An optional conditioning vector

The timestep is normalized by dividing by `1000.0`, then passed through a small
time-embedding network. The time embedding is added to every trajectory point,
and the optional condition embedding is added in the same way.

Default architecture:

```text
Input trajectory:          (batch_size, 128, 2)
Channel-first projection:  Conv1d(2, 256, kernel_size=1)
Time embedding:            Linear(1, 256) -> SiLU -> Linear(256, 256)
Condition embedding:       Linear(cond_dim, 256), when conditioning is enabled
Residual temporal blocks:  Conv1d blocks with dilations 1, 2, 4, and 8
Output projection:         Conv1d(256, 2, kernel_size=1)
Output noise:              (batch_size, 128, 2)
```

The output has the same shape as the input trajectory because the model predicts
the noise added to each coordinate at each timestep.

The older flattened MLP remains available with `--architecture mlp` so older
checkpoints can still be sampled and compared.

When `--conditioning start-goal` is used, the conditioning vector is:

```text
[start_x, start_y, goal_x, goal_y]
```

The start and goal values are taken from the normalized trajectory coordinates.
Older unconditional checkpoints remain supported with `cond_dim = 0`.

## Diffusion Process

Diffusion file:

- `lasa-diffusion/diffusion.py`

Class:

- `Diffusion`

Default diffusion hyperparameters:

```text
timesteps = 1000
beta_start = 1e-4
beta_end = 0.02
```

The implementation uses a linear beta schedule:

```text
beta_t = linearly spaced values from 1e-4 to 0.02
alpha_t = 1 - beta_t
alpha_bar_t = product(alpha_1 ... alpha_t)
```

During training, clean trajectories `x_0` are noised using:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

where:

```text
epsilon ~ N(0, I)
```

The model is trained to predict `epsilon`.

## Training Loop

Training file:

- `lasa-diffusion/train.py`

Recommended training command:

```bash
cd lasa-diffusion
python train.py --shape-name Angle --seq-len 128 --epochs 10000 --batch-size 7 --timesteps 1000 --hidden 256 --conditioning start-goal --architecture temporal-conv --output-dir outputs/temporal_conv_metrics
```

Older MLP training command:

```bash
cd lasa-diffusion
python train.py --shape-name Angle --epochs 3000 --batch-size 7 --conditioning start-goal --architecture mlp --output-dir outputs/start_goal
```

Fast smoke test:

```bash
cd lasa-diffusion
python train.py --shape-name Angle --epochs 5 --batch-size 7 --timesteps 100
```

Default training hyperparameters:

```text
shape_name = Angle
seq_len = 128
batch_size = 7
epochs = 10000 for the current trained temporal-conv checkpoint
timesteps = 1000
hidden = 256
learning_rate = 1e-4
optimizer = Adam
loss = mean squared error
seed = 42
conditioning = start-goal
architecture = temporal-conv
validation_split = 0.2
```

For each training batch:

1. Load normalized clean trajectories `x_0`.
2. Sample a random timestep `t` for each trajectory in the batch.
3. Add Gaussian noise using the forward diffusion process.
4. Optionally pass the normalized start/goal condition into `TrajectoryDenoiser`.
5. Predict the added noise with `TrajectoryDenoiser`.
6. Compute MSE between predicted noise and true noise.
7. Backpropagate and update the model with Adam.

The training script records training and validation metrics for each epoch.
Validation uses a held-out split of the LASA demonstrations.

Recorded metrics:

- `loss`: mean squared error between predicted and true diffusion noise. This is
  the main optimization metric.
- `accuracy`: proxy metric measuring whether each predicted noise coordinate has
  the same sign as the true noise coordinate.
- `f1`: proxy F1 score after binarizing noise coordinates by sign.

Accuracy and F1 are included because they are useful report-style diagnostic
plots, but they should be described as proxy metrics. This project is a
continuous generative modeling task, so loss and trajectory quality are more
important than classification accuracy.

## Checkpoint

Current recommended checkpoint path:

```text
lasa-diffusion/outputs/temporal_conv_metrics/lasa_diffusion.pt
```

The checkpoint stores:

```text
model       - PyTorch model state dictionary
mean        - dataset mean used for normalization
std         - dataset standard deviation used for normalization
shape_name  - LASA shape name used for training
seq_len     - trajectory sequence length
timesteps   - number of diffusion timesteps
hidden      - hidden layer width
conditioning - none or start-goal
cond_dim    - conditioning vector size
architecture - temporal-conv or mlp
losses      - epoch-level training losses
train_losses - epoch-level training MSE
val_losses - epoch-level validation MSE
train_accuracy - epoch-level training noise-sign accuracy
val_accuracy - epoch-level validation noise-sign accuracy
train_f1    - epoch-level training noise-sign F1
val_f1      - epoch-level validation noise-sign F1
metric_definitions - text definitions for the stored metrics
```

The checkpoint is used by:

- `lasa-diffusion/sample.py`
- `webapp/app.py`

## Training Outputs

The training script saves:

```text
lasa-diffusion/outputs/temporal_conv_metrics/lasa_diffusion.pt
lasa-diffusion/outputs/temporal_conv_metrics/loss.png
lasa-diffusion/outputs/temporal_conv_metrics/accuracy.png
lasa-diffusion/outputs/temporal_conv_metrics/f1.png
```

The metric plots can be included in the final report. Checkpoints trained before
these fields were added may only contain training loss.

## Sampling

Sampling file:

- `lasa-diffusion/sample.py`

Default sampling command:

```bash
cd lasa-diffusion
python sample.py --checkpoint outputs/temporal_conv_metrics/lasa_diffusion.pt --num-samples 10 --timesteps 1000
```

Conditional sampling command for a start/goal checkpoint:

```bash
cd lasa-diffusion
python sample.py --checkpoint outputs/temporal_conv_metrics/lasa_diffusion.pt --num-samples 10 --timesteps 1000 --start -40 -3 --goal 0 0
```

Sampling starts from Gaussian noise:

```text
x_T ~ N(0, I)
```

Then it iterates backward from timestep `T - 1` to `0`, applying the learned
denoising model at each step. The final generated trajectories are unnormalized
using the checkpoint's stored mean and standard deviation.

Default output:

```text
lasa-diffusion/outputs/generated_trajectories.png
```

## Web App Inference

The Streamlit app loads the same checkpoint and sampling function.

App file:

- `webapp/app.py`

Run command:

```bash
./lasa-diffusion/venv/bin/streamlit run webapp/app.py
```

The app allows the user to:

- Select a checkpoint path.
- Select a LASA reference shape.
- Choose the number of generated samples.
- Choose a random seed.
- Choose fast, balanced, or full sampling.
- Draw a trajectory sketch.
- Enter or sketch start and goal points.
- Overlay real LASA demonstrations.
- Export the trajectory plot as PNG.
- Export generated trajectories as CSV.
- Inspect smoothness, curvature, endpoint error, and nearest-demonstration distance.
- View generated trajectories and checkpoint metadata.

## Current Scope and Limitations

The project now uses a stronger temporal-convolution denoiser for the main demo,
while keeping the original MLP implementation for comparison and older
checkpoints.

Important limitations:

- The current model is still compact and trained on a very small number of LASA
  demonstrations for each shape.
- The default setup trains on one LASA shape at a time.
- The model supports learned start-goal conditioning, but it does not yet learn
  full sketch conditioning from the entire user drawing.
- Fast and balanced app sampling use fewer reverse denoising steps, so full
  `1000`-step sampling should be used for final-quality outputs.
- Quantitative evaluation is currently limited to training loss and visual
  inspection of generated trajectories.

Potential extensions:

- Train on more LASA shapes and condition on the shape label.
- Replace the temporal convolution with a trajectory U-Net or Transformer.
- Add trajectory metrics such as endpoint error, dynamic time warping distance,
  smoothness, and nearest-neighbor comparison to demonstrations.
- Add side-by-side comparison between real LASA demonstrations and generated
  trajectories in the web app.
