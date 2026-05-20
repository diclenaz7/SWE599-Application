# SWE 599 Application Demo

Implementation workspace for the SWE 599 project: learning from demonstration with
diffusion-based trajectory models.

## Project Layout

- `lasa-diffusion/` - LASA dataset loader, denoising model, DDPM training, and sampling.
- `webapp/` - Streamlit demo that loads a trained checkpoint and generates trajectories.
- `report/` - place the Overleaf LaTeX source here.
- `trajectory_project/` - earlier synthetic trajectory exploration utilities and figures.

## Setup

Fast path using the existing model virtualenv:

```bash
source lasa-diffusion/venv/bin/activate
```

Fresh environment option:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `pyLasaDataset` is already installed in `lasa-diffusion/venv`, you can also use
that existing environment for the model scripts.

## Train

Recommended multi-shape training:

```bash
cd lasa-diffusion
python train.py --curated-shapes --seq-len 128 --epochs 10000 --batch-size 16 --timesteps 1000 --hidden 256 --conditioning start-goal --architecture temporal-conv
```

The curated training subset is:

```text
Angle, CShape, GShape, JShape, LShape, Sine, Spoon, WShape
```

This command saves to `lasa-diffusion/outputs/multi_shape_temporal_conv/` by default.

Single-shape training is still available:

```bash
cd lasa-diffusion
python train.py --shape-name Angle --seq-len 128 --epochs 10000 --batch-size 7 --timesteps 1000 --hidden 256 --conditioning start-goal --architecture temporal-conv --output-dir outputs/temporal_conv_metrics
```

Useful fast smoke test:

```bash
cd lasa-diffusion
python train.py --shape-name Angle --epochs 5 --batch-size 7 --timesteps 100
```

Optional older MLP comparison training:

```bash
cd lasa-diffusion
python train.py --shape-name Angle --epochs 3000 --batch-size 7 --conditioning start-goal --architecture mlp --output-dir outputs/start_goal
```

Training saves:

- `lasa-diffusion/outputs/multi_shape_temporal_conv/lasa_diffusion.pt`
- `lasa-diffusion/outputs/multi_shape_temporal_conv/loss.png`
- `lasa-diffusion/outputs/multi_shape_temporal_conv/accuracy.png`
- `lasa-diffusion/outputs/multi_shape_temporal_conv/f1.png`
- `lasa-diffusion/outputs/temporal_conv_metrics/lasa_diffusion.pt`
- `lasa-diffusion/outputs/temporal_conv_metrics/loss.png`
- `lasa-diffusion/outputs/temporal_conv_metrics/accuracy.png`
- `lasa-diffusion/outputs/temporal_conv_metrics/f1.png`

## Sample

```bash
cd lasa-diffusion
python sample.py --checkpoint outputs/multi_shape_temporal_conv/lasa_diffusion.pt --num-samples 10 --timesteps 1000
```

Sampling saves:

- `lasa-diffusion/outputs/generated_trajectories.png`

## Run The Web App

```bash
./lasa-diffusion/venv/bin/streamlit run webapp/app.py
```

The app prefers `lasa-diffusion/outputs/multi_shape_temporal_conv/lasa_diffusion.pt`
when it exists, then falls back to older checkpoints.

The app includes:

- mouse drawing for trajectory guidance
- fixed start/goal guidance from the sketch endpoints or default reference endpoints
- real LASA demonstration overlays
- checkpoint selection
- checkbox filters for the LASA shapes used by the selected checkpoint
- simplified demo controls for sample count, variation, and sketch influence
- trajectory metrics
- training/validation metric plots from the selected checkpoint
- PNG and CSV export buttons
- model metadata display

## Report Workflow

Download the Overleaf project source as a zip and extract it into `report/`.
The current exported report is under `report/SWE599-Report/`.
Keep final generated figures in either `report/figures/` or reference the stable
paths under `lasa-diffusion/outputs/`.
