# Technology Stack

This project implements and demonstrates a diffusion-based learning-from-demonstration
model for 2D robot trajectory generation using LASA handwriting trajectories.

## Python

Python is the main implementation language for the project.

Used for:
- Loading and preprocessing trajectory data.
- Defining the diffusion process.
- Training the denoising neural network.
- Sampling generated trajectories from a trained checkpoint.
- Running the Streamlit web demo.

Main files:
- `lasa-diffusion/dataset.py`
- `lasa-diffusion/diffusion.py`
- `lasa-diffusion/model.py`
- `lasa-diffusion/train.py`
- `lasa-diffusion/sample.py`
- `webapp/app.py`

## PyTorch

PyTorch is used for the machine learning implementation.

Used for:
- Defining the trajectory denoising neural network.
- Running the forward diffusion noising process during training.
- Training with the mean squared error noise-prediction objective.
- Saving and loading model checkpoints.
- Sampling trajectories through the reverse denoising process.

Relevant files:
- `lasa-diffusion/model.py`
- `lasa-diffusion/diffusion.py`
- `lasa-diffusion/train.py`
- `lasa-diffusion/sample.py`

The current default model is a compact temporal-convolution denoiser. It keeps
the trajectory as a sequence of 2D points and uses `Conv1d` residual blocks with
different dilation rates so nearby and moderately distant trajectory points can
influence each other. The older flattened MLP denoiser is still available for
comparison and for loading old checkpoints.

## Diffusion Model

The project uses a DDPM-style diffusion formulation.

Used for:
- Adding Gaussian noise to clean trajectories at randomly sampled timesteps.
- Training the neural network to predict the added noise.
- Starting from random noise at inference time and iteratively denoising it into
  a generated trajectory.

Relevant file:
- `lasa-diffusion/diffusion.py`

The implementation keeps the method intentionally simple and readable so it can be
explained clearly in the final report.

## LASA Handwriting Dataset

The LASA Handwriting Dataset is used as the source of demonstration trajectories.

Used for:
- Providing 2D handwriting demonstrations.
- Resampling demonstrations to a fixed sequence length.
- Normalizing trajectories before training.

Relevant file:
- `lasa-diffusion/dataset.py`

Dependency:
- `pyLasaDataset`

## NumPy

NumPy is used for numerical preprocessing outside the neural network.

Used for:
- Resampling trajectories with interpolation.
- Stacking trajectory arrays.
- Computing dataset mean and standard deviation.
- Converting generated model outputs back to original coordinate scale.

Relevant files:
- `lasa-diffusion/dataset.py`
- `lasa-diffusion/sample.py`
- `trajectory_project/*.py`

## Matplotlib

Matplotlib is used for static visualization.

Used for:
- Plotting training loss.
- Plotting training/validation accuracy and F1 proxy metrics.
- Plotting generated trajectories.
- Producing figures that can be inserted into the LaTeX report.
- Supporting visualization inside the Streamlit app.

Relevant files:
- `lasa-diffusion/train.py`
- `lasa-diffusion/sample.py`
- `webapp/app.py`
- `trajectory_project/visualize_trajectories.py`
- `trajectory_project/plot_dataset_analysis.py`

The scripts use a non-interactive plotting backend so figures can be saved reliably
from command-line runs.

## Streamlit

Streamlit is used for the web application demo.

Used for:
- Loading a trained diffusion checkpoint.
- Letting the user choose sampling settings.
- Generating trajectories interactively.
- Drawing trajectory sketches through `streamlit-drawable-canvas`.
- Overlaying real LASA demonstrations with generated trajectories.
- Displaying trajectory plots, training curves, metrics, exports, and checkpoint metadata in the browser.

Relevant file:
- `webapp/app.py`

Run command:

```bash
./lasa-diffusion/venv/bin/streamlit run webapp/app.py
```

## tqdm

`tqdm` is used for progress bars during model training.

Relevant file:
- `lasa-diffusion/train.py`

## LaTeX

LaTeX is used for the final project report.

Used for:
- Writing the formal report.
- Managing figures, references, and thesis-style formatting.
- Producing the final PDF submission.

Current report folder:
- `report/SWE599-Report/`

Key files:
- `report/SWE599-Report/SWE599_Report.tex`
- `report/SWE599-Report/references.bib`
- `report/SWE599-Report/styles/`

## Git

Git is used for version control.

Used for:
- Tracking implementation changes.
- Keeping report, code, figures, and documentation in one workspace.
- Making it easier to recover or compare project states while finishing.

## Project Dependencies

The root dependency list is stored in:

- `requirements.txt`

Current dependencies:

```text
numpy>=1.24
matplotlib>=3.7
torch>=2.0
tqdm>=4.66
streamlit>=1.34
streamlit-drawable-canvas>=0.9.3
pyLasaDataset>=0.1.1
```

## Runtime Artifacts

The model workflow produces local artifacts such as:

- `lasa-diffusion/outputs/temporal_conv_metrics/lasa_diffusion.pt`
- `lasa-diffusion/outputs/temporal_conv_metrics/loss.png`
- `lasa-diffusion/outputs/temporal_conv_metrics/accuracy.png`
- `lasa-diffusion/outputs/temporal_conv_metrics/f1.png`
- `lasa-diffusion/outputs/generated_trajectories.png`

These files are useful for the demo and final report, but large checkpoints should
usually stay out of Git unless explicitly needed for submission.
