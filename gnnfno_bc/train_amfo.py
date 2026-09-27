"""Train A-MFO-Net on the frozen C89 split and evaluate its held-out rollout.

This entry point exposes only the learned, aligned attention configuration.
The shared trainer also contains historical ablation switches, which are not
selected here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
SPLIT_PATH = HERE / "c89_matched_benchmark_split.json"
SPLIT_SHA256 = "e05b5e5d1c8e5135f93f980e3ebb2c50633ffe38f2cd7076da5eb6c060e73c15"
HELD_OUT_CASE = "big_room__cooling-090pct__server-100pct__frames-401"


def check_split() -> None:
    raw = SPLIT_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPLIT_SHA256:
        raise ValueError("The frozen C89 case-split manifest has changed")
    split = json.loads(raw)
    train, valid, test = (split[key] for key in ("train_cases", "valid_cases", "test_cases"))
    if (len(train), len(valid), test) != (14, 2, [HELD_OUT_CASE]):
        raise ValueError("Expected the frozen 14/2/1 extended-room split")
    if len(set(train + valid + test)) != 17:
        raise ValueError("Case-split partitions overlap")


def load_config(protocol: str, seed: int, data_root: Path, output_root: Path):
    check_split()
    cfg = OmegaConf.load(HERE / "config_amfo.yaml")
    cfg.training.seed = seed
    cfg.training.max_epochs = 100
    cfg.training.batch_size = 2
    cfg.training.rollout_steps = 3
    cfg.training.use_amp = False
    cfg.training.train_interior_only = True
    cfg.training.semi_teacher_force_boundary = True
    cfg.training.teacher_forcing = False
    cfg.training.use_multistep_loss = True
    cfg.validation.rollout_steps = 1
    cfg.validation.validation_epochs = 20
    cfg.validation.batch_size = 1
    cfg.split.explicit_manifest = SPLIT_PATH.name
    cfg.arch.gnn.use_attention = True
    cfg.arch.gnn.attention_mode = "learned"
    cfg.arch.gnn.score_edge_mode = "aligned"
    cfg.test.rollout_steps = 400
    cfg.test.semi_teacher_force_boundary = True
    cfg.test.metric_scope = "all"
    cfg.test.rollout_after_training = True
    cfg.data.data_root = str(data_root)
    cfg.output.root_dir = str(output_root)
    cfg.output.run_name = f"amfo_{protocol}_seed{seed}"
    if protocol == "submitted":
        cfg.data.max_train_samples = 200
        cfg.data.max_valid_samples = 20
        cfg.training.rec_results_freq = 20
        cfg.test.use_in_memory_final_weights = False
        cfg.test.save_rollout_vtk_steps = None
        cfg.test.save_rollout_npz = False
    elif protocol == "fullwindow":
        cfg.data.max_train_samples = -1
        cfg.data.max_valid_samples = -1
        cfg.training.rec_results_freq = 50
        cfg.test.use_in_memory_final_weights = True
        cfg.test.save_rollout_vtk_steps = [0, 20, 200, 400]
        cfg.test.save_rollout_npz = True
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("submitted", "fullwindow"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    cfg = load_config(args.protocol, args.seed, data_root, output_root)
    run_dir = output_root / cfg.output.run_name
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"Refusing to reuse a run directory: {run_dir}")

    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    if not data_root.is_dir():
        raise FileNotFoundError(f"CFD data root is unavailable: {data_root}")

    import train_gnn_fno_semibc as experiment

    experiment.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
