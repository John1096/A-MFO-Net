# A-MFO-Net source release draft

This repository contains the A-MFO-Net implementation for three-dimensional
server-room flow, pressure, and temperature forecasting. It is a local release
draft. Selected paper figures and CSV metrics from the completed, capped
three-seed campaign are included. The CFD dataset, trained weights, and
prediction trajectories are not included.

## Model and protocol

The model in `gnnfno_bc/train_gnn_fno_semibc.py` encodes a sampled mesh graph,
processes a regular 3-D latent grid with a Fourier neural operator, and decodes
the grid features on the mesh. Its graph encoder and decoder use learned
edge-aware attention. The state has five channels (three velocity components,
pressure, temperature), and the supplied boundary type has six channels.
The included configuration uses an eight-neighbor graph, a 32³ grid, 64 latent
channels, four FNO layers with 12 modes, and two graph processor layers on
each side of the FNO.

`gnnfno_bc/train_amfo.py` fixes the frozen 14-training/2-validation/1-test
case split, learned aligned attention, 100 epochs, batch size two, FP32,
three-step autoregressive training, 2,048 sampled training cells per item, and
a boundary-conditioned 400-step rollout on the held-out extended room. Loss
is computed on interior cells. The validation pass is one-step; the historical
`validation.rollout_steps` label is not a 20-step validation horizon.

Two temporal sampling choices are exposed:

| Protocol | Training windows per epoch | Validation windows | Use |
| --- | ---: | ---: | --- |
| `submitted` | 200 | 20 | Earlier capped experiment |
| `fullwindow` | 2,972 | 400 | Follow-up using every available start |

Both use the same case split and 400-step all-cell test. The `fullwindow`
choice does not change the 2,048-cell training query subsample. The
`submitted` choice keeps the original output policy of all 401 prediction
VTKs. The `fullwindow` choice keeps VTKs at steps 0, 20, 200, and 400 plus
`trajectory.npz` with the full predicted sequence. The metric CSV records all
400 forecast steps in either mode. Boundary values are imposed from the
prescribed test state at each step, so the rollout is boundary conditioned.

## Data and environment

The data are not publicly available in this draft, and no data license is
asserted here. Supply a local directory with the 17 case directories named in
`gnnfno_bc/c89_matched_benchmark_split.json`. Each case needs consecutive
`1_0.vtk`, `1_1.vtk`, … frames. The loader expects velocity, pressure, and
temperature arrays plus six-class `boundary_types` or `boundary_type` fields;
the accepted physical-array aliases are in `config_amfo.yaml`. The reader
supports cell and point arrays and converts boundary labels when necessary.

The original code ran in a PhysicsNeMo 26.03 GPU container. It imports
PyTorch, PhysicsNeMo, PyTorch Geometric, Hydra, OmegaConf, NumPy, SciPy,
PyVista, and VTK. Exact package versions and a clean-environment install have
not been verified for this draft. PhysicsNeMo's FNO implementation is an
external dependency; no third-party source is bundled.

From an environment with those dependencies and the dataset available:

```bash
python gnnfno_bc/train_amfo.py \
  --protocol submitted --seed 42 \
  --data-root /path/to/cfd-data --output-root /path/to/output --dry-run
python gnnfno_bc/train_amfo.py \
  --protocol submitted --seed 42 \
  --data-root /path/to/cfd-data --output-root /path/to/output
```

Choose `--protocol fullwindow` for the uncapped follow-up. The entry point
refuses an existing run directory. It writes
`amfo_<protocol>_seed<seed>/resolved_config.yaml`, the final checkpoint,
training and rollout metrics, and the test predictions beneath the output root.
The original trainer also writes `epoch_XXXX.pt` snapshots at the configured
checkpoint interval. Plan storage for those files when running this release.
Run the dry run first to inspect every resolved setting. Training requires
substantial GPU time and storage; the dry run does not load the CFD data.

To repeat the held-out boundary-conditioned rollout from a saved A-MFO-Net
checkpoint, use a new output directory:

```bash
python gnnfno_bc/infer_amfo.py \
  --run-dir /path/to/output/amfo_submitted_seed42 \
  --data-root /path/to/cfd-data --output-root /path/to/new-rollout
```

The trainer's `rollout_metrics_per_step.csv` covers all 400 steps. For extra
R² metrics on the retained VTK snapshots, run:

```bash
python gnnfno_bc/evaluate_rollout_r2.py \
  /path/to/new-rollout/test_rollout/big_room__cooling-090pct__server-100pct__frames-401 \
  /path/to/cfd-data/big_room__cooling-090pct__server-100pct__frames-401
```

That optional R² script evaluates only the VTK steps present; with the compact
setting it summarizes three prediction steps (20, 200, 400), not the full
trajectory.

## Source notes

The original trainer is retained intact for numerical fidelity. It contains
dormant attention and score-edge ablation switches used in earlier internal
studies. The public entry point pins the learned, aligned A-MFO-Net path and
does not expose another network. The C70 integrity module and score-edge
helpers are included because the original trainer imports them; their
separate study launcher and artifacts are not included.

The `metrics/` directory contains the 400-step A-MFO-Net rollout metrics from
the earlier 200-window campaign; its README states their scope and provenance.
The selected `figures/` directory is documented separately. Neither directory
contains results from the full-window follow-up. No third-party checkout,
baseline model implementation, checkpoint, or CFD frame is included. No
license has been selected for the project source yet.
