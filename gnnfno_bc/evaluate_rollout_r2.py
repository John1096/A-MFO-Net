"""Evaluate saved VTK rollouts with RMSE, RelL2, R2, and relative SSE."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_gnn_fno_semibc import (
    list_time_vtk_files,
    read_vtk_boundary_features,
    read_vtk_state,
)


CHANNEL_NAMES = ("vx", "vy", "vz", "pressure", "temperature")
VELOCITY_KEYS = ("Velocity(m/s)", "velocity", "Velocity", "U")
PRESSURE_KEYS = ("Pressure(Pa)", "pressure", "Pressure", "p")
TEMPERATURE_KEYS = ("Temperature(C)", "temperature", "Temperature", "T")


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    sse = float(np.square(error).sum())
    target_mean = target.mean(axis=0, keepdims=True)
    sst = float(np.square(target - target_mean).sum())
    energy = float(np.square(target).sum())
    r2 = float(1.0 - sse / max(sst, 1.0e-24))
    return {
        "rmse": float(np.sqrt(np.square(error).mean())),
        "relative_l2": float(np.sqrt(sse / max(energy, 1.0e-24))),
        "r2": r2,
        "one_minus_r2": float(1.0 - r2),
        "relative_squared_error": float(sse / max(energy, 1.0e-24)),
    }


def read_state(path: str | Path) -> np.ndarray:
    _, state, _ = read_vtk_state(
        str(path), VELOCITY_KEYS, PRESSURE_KEYS, TEMPERATURE_KEYS
    )
    return state.astype(np.float64, copy=False)


def read_target_with_interior_mask(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    _, state, use_cell_data = read_vtk_state(
        str(path), VELOCITY_KEYS, PRESSURE_KEYS, TEMPERATURE_KEYS
    )
    boundary_features = read_vtk_boundary_features(
        str(path),
        use_cell_data=use_cell_data,
        expected_n=state.shape[0],
        use_boundary_features=True,
        boundary_onehot_key="boundary_types",
        boundary_class_key="boundary_type",
        num_boundary_classes=6,
    )
    interior_mask = boundary_features[:, 1] > 0.5
    if not np.any(interior_mask):
        raise ValueError(f"No boundary class 1 (flow/interior) entries in {path}")
    return state.astype(np.float64, copy=False), interior_mask


def empty_aggregate() -> dict[str, np.ndarray | int]:
    return {
        "sse": np.zeros(5, dtype=np.float64),
        "target_sum": np.zeros(5, dtype=np.float64),
        "target_sq_sum": np.zeros(5, dtype=np.float64),
        "count": 0,
    }


def update_aggregate(
    aggregate: dict[str, np.ndarray | int],
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    aggregate["sse"] += np.square(prediction - target).sum(axis=0)
    aggregate["target_sum"] += target.sum(axis=0)
    aggregate["target_sq_sum"] += np.square(target).sum(axis=0)
    aggregate["count"] += target.shape[0]


def add_global_summary(
    summary: dict[str, object],
    aggregate: dict[str, np.ndarray | int],
    scope_prefix: str,
) -> None:
    sse_by_channel = np.asarray(aggregate["sse"])
    target_sum = np.asarray(aggregate["target_sum"])
    target_sq_sum = np.asarray(aggregate["target_sq_sum"])
    count = max(int(aggregate["count"]), 1)
    global_sst = target_sq_sum - np.square(target_sum) / count
    prefix = f"global_{scope_prefix}" if scope_prefix else "global_"

    for channel, name in enumerate(CHANNEL_NAMES):
        sse = float(sse_by_channel[channel])
        r2 = float(1.0 - sse / max(float(global_sst[channel]), 1.0e-24))
        summary[f"{prefix}{name}_r2"] = r2
        summary[f"{prefix}{name}_one_minus_r2"] = float(1.0 - r2)
        summary[f"{prefix}{name}_rmse"] = float(np.sqrt(sse / count))
        relative_squared_error = float(
            sse / max(float(target_sq_sum[channel]), 1.0e-24)
        )
        summary[f"{prefix}{name}_relative_squared_error"] = (
            relative_squared_error
        )
        summary[f"{prefix}{name}_relative_l2"] = float(
            np.sqrt(relative_squared_error)
        )

    velocity_sse = float(sse_by_channel[:3].sum())
    velocity_r2 = float(
        1.0 - velocity_sse / max(float(global_sst[:3].sum()), 1.0e-24)
    )
    summary[f"{prefix}velocity_vector_r2"] = velocity_r2
    summary[f"{prefix}velocity_vector_one_minus_r2"] = float(
        1.0 - velocity_r2
    )
    summary[f"{prefix}velocity_vector_rmse"] = float(
        np.sqrt(velocity_sse / count)
    )
    velocity_relative_squared_error = float(
        velocity_sse / max(float(target_sq_sum[:3].sum()), 1.0e-24)
    )
    summary[f"{prefix}velocity_vector_relative_squared_error"] = (
        velocity_relative_squared_error
    )
    summary[f"{prefix}velocity_vector_relative_l2"] = float(
        np.sqrt(velocity_relative_squared_error)
    )
    summary[f"{prefix}sample_count"] = int(aggregate["count"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("ground_truth_dir", type=Path)
    parser.add_argument("--label", default="rollout")
    args = parser.parse_args()

    prediction_dir = args.prediction_dir.resolve()
    ground_truth_files = list_time_vtk_files(str(args.ground_truth_dir.resolve()))
    prediction_files = sorted(prediction_dir.glob("pred_*.vtk"))
    if not prediction_files:
        raise FileNotFoundError(f"No pred_*.vtk files in {prediction_dir}")

    rows = []
    aggregate = empty_aggregate()
    interior_aggregate = empty_aggregate()
    mesh_cell_count = None
    interior_cell_count = None
    for prediction_path in prediction_files:
        step = int(prediction_path.stem.rsplit("_", 1)[1])
        if step == 0:
            continue
        if step >= len(ground_truth_files):
            raise IndexError(f"No ground-truth frame {step} for {prediction_path}")
        prediction = read_state(prediction_path)
        target, interior_mask = read_target_with_interior_mask(
            ground_truth_files[step]
        )
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch at step {step}: {prediction.shape} vs {target.shape}"
            )

        row = {"step": step}
        velocity_metrics = metrics(prediction[:, :3], target[:, :3])
        velocity_metrics["rmse"] = float(
            np.sqrt(
                np.mean(
                    np.sum(np.square(prediction[:, :3] - target[:, :3]), axis=1)
                )
            )
        )
        for name, value in velocity_metrics.items():
            row[f"velocity_vector_{name}"] = value
        for channel, name in enumerate(CHANNEL_NAMES):
            channel_metrics = metrics(
                prediction[:, channel : channel + 1],
                target[:, channel : channel + 1],
            )
            for metric_name, value in channel_metrics.items():
                row[f"{name}_{metric_name}"] = value

        interior_prediction = prediction[interior_mask]
        interior_target = target[interior_mask]
        interior_velocity_metrics = metrics(
            interior_prediction[:, :3], interior_target[:, :3]
        )
        interior_velocity_metrics["rmse"] = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            interior_prediction[:, :3]
                            - interior_target[:, :3]
                        ),
                        axis=1,
                    )
                )
            )
        )
        for name, value in interior_velocity_metrics.items():
            row[f"interior_velocity_vector_{name}"] = value
        for channel, name in enumerate(CHANNEL_NAMES):
            channel_metrics = metrics(
                interior_prediction[:, channel : channel + 1],
                interior_target[:, channel : channel + 1],
            )
            for metric_name, value in channel_metrics.items():
                row[f"interior_{name}_{metric_name}"] = value
        rows.append(row)

        update_aggregate(aggregate, prediction, target)
        update_aggregate(
            interior_aggregate, interior_prediction, interior_target
        )
        mesh_cell_count = target.shape[0]
        interior_cell_count = interior_target.shape[0]

    rows.sort(key=lambda row: row["step"])
    csv_path = prediction_dir / "rollout_r2_metrics_per_step.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "label": args.label,
        "prediction_dir": str(prediction_dir),
        "ground_truth_dir": str(args.ground_truth_dir.resolve()),
        "steps": len(rows),
        "mesh_cells_per_step": mesh_cell_count,
        "interior_cells_per_step": interior_cell_count,
        "metric_scopes": {
            "unprefixed": "all mesh entries; includes exact imposed boundary values",
            "interior": "boundary_type == 1 only; primary forecast-quality scope",
        },
    }
    for key in rows[0]:
        if key != "step":
            summary[f"mean_step_{key}"] = float(
                np.mean([float(row[key]) for row in rows])
            )

    add_global_summary(summary, aggregate, scope_prefix="")
    add_global_summary(summary, interior_aggregate, scope_prefix="interior_")

    summary_path = prediction_dir / "rollout_r2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
