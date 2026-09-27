from __future__ import annotations

import os
import re
import glob
import random
import csv
import json
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from scipy.spatial import cKDTree
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, LaunchLogger

from case_split_controls import (
    C70_TRAINING_WINDOW_MANIFEST_FILENAME,
    load_c70_training_window_selection,
    load_explicit_case_split,
    resolve_c70_training_window_samples,
)
from c70_score_control_integrity import (
    ScoreControlIntegrityError,
    SOURCE_LOCK_FILENAME,
    canonical_run_name,
    reserve_fresh_run_directory,
    runtime_identity,
    verify_canonical_source_lock,
    write_completion_manifest,
)
from score_edge_controls import (
    destination_local_derangement,
    score_edge_attributes,
    validate_score_edge_mode,
)


# ============================================================
# VTK HELPERS
# ============================================================
def list_case_dirs(data_root: str) -> List[str]:
    # Supports the original cXXsYY folders and descriptive room/capacity names,
    # e.g. small_room__cooling-085pct__server-100pct__frames-201.
    case_name_pattern = re.compile(
        r"(?:c\d+s\d+|(?:small|big)_room__cooling-\d{3}pct__server-\d{3}pct__frames-\d+)"
    )
    case_dirs = []
    if not os.path.isdir(data_root):
        raise RuntimeError(f"Data root does not exist: {data_root}")
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if os.path.isdir(path) and case_name_pattern.fullmatch(name):
            case_dirs.append(path)
    if not case_dirs:
        raise RuntimeError(f"No recognized case folders found under {data_root}")
    return case_dirs



def split_case_dirs(case_dirs: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    train_cases, valid_cases, _ = split_case_dirs_three_way(case_dirs, seed, max(1, len(case_dirs)-1), 1 if len(case_dirs)>1 else 0, 0)
    return train_cases, valid_cases


def split_case_dirs_three_way(case_dirs: List[str], seed: int, train_count: int = 10, valid_count: int = 2, test_count: int = 1) -> Tuple[List[str], List[str], List[str]]:
    if not case_dirs:
        raise RuntimeError("No case directories available for splitting.")
    requested = int(train_count) + int(valid_count) + int(test_count)
    if len(case_dirs) < requested:
        raise RuntimeError(f"Need at least {requested} cases for split ({train_count} train, {valid_count} valid, {test_count} test), but found only {len(case_dirs)} cases.")
    rng = random.Random(seed)
    shuffled = list(case_dirs)
    rng.shuffle(shuffled)
    test_cases = sorted(shuffled[:test_count])
    valid_cases = sorted(shuffled[test_count:test_count + valid_count])
    train_cases = sorted(shuffled[test_count + valid_count:test_count + valid_count + train_count])
    return train_cases, valid_cases, test_cases


def list_time_vtk_files(case_dir: str) -> List[str]:
    files = []
    for path in glob.glob(os.path.join(case_dir, "*.vtk")):
        base = os.path.basename(path)
        m = re.fullmatch(r"1_(\d+)\.vtk", base)
        if m is not None:
            files.append((int(m.group(1)), path))
    files.sort(key=lambda x: x[0])
    vtk_files = [p for _, p in files]
    if len(vtk_files) < 2:
        raise RuntimeError(f"Need at least 2 timestep VTK files in {case_dir}")
    return vtk_files


def _find_array_key(point_keys, cell_keys, candidates):
    for k in candidates:
        if k in point_keys:
            return k, "point"
        if k in cell_keys:
            return k, "cell"
    return None, None


def read_vtk_state(
    vtk_path: str,
    velocity_keys: List[str],
    pressure_keys: List[str],
    temperature_keys: List[str],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Returns
    -------
    coords : (N, 3) float32
    state  : (N, 5) float32  [vx, vy, vz, pressure, temperature]
    use_cell_data : bool
    """
    mesh = pv.read(vtk_path)

    point_keys = set(mesh.point_data.keys())
    cell_keys = set(mesh.cell_data.keys())

    vel_key, vel_loc = _find_array_key(point_keys, cell_keys, velocity_keys)
    p_key, p_loc = _find_array_key(point_keys, cell_keys, pressure_keys)
    t_key, t_loc = _find_array_key(point_keys, cell_keys, temperature_keys)

    if vel_key is None:
        raise KeyError(f"No velocity array found in {vtk_path}. Tried {velocity_keys}")
    if p_key is None:
        raise KeyError(f"No pressure array found in {vtk_path}. Tried {pressure_keys}")
    if t_key is None:
        raise KeyError(f"No temperature array found in {vtk_path}. Tried {temperature_keys}")

    use_cell_data = (vel_loc == "cell") or (p_loc == "cell") or (t_loc == "cell")

    if use_cell_data:
        coords = mesh.cell_centers().points.astype(np.float32)
        source = mesh.cell_data
    else:
        coords = mesh.points.astype(np.float32)
        source = mesh.point_data

    vel = np.asarray(source[vel_key], dtype=np.float32)
    p = np.asarray(source[p_key], dtype=np.float32).reshape(-1, 1)
    t = np.asarray(source[t_key], dtype=np.float32).reshape(-1, 1)

    if vel.ndim != 2 or vel.shape[1] != 3:
        raise ValueError(f"Velocity field shape is {vel.shape}, expected (N,3)")

    state = np.concatenate([vel, p, t], axis=1).astype(np.float32)
    return coords, state, use_cell_data


def read_vtk_boundary_features(
    vtk_path: str,
    use_cell_data: bool,
    expected_n: int,
    use_boundary_features: bool = True,
    boundary_onehot_key: str = "boundary_onehot",
    boundary_class_key: str = "boundary_class",
    num_boundary_classes: int = 6,
) -> np.ndarray:
    """
    Read boundary features from the preprocessed VTK and align them with the
    training nodes.

    Your current physical fields are often cell_data, while the boundary labels
    produced by preprocessing may be point_data. If training uses cell centers,
    this function converts point_data labels to cell_data using PyVista's
    point_data_to_cell_data().

    Returns
    -------
    boundary_feats : (N, C) float32
        Usually C=6: [wall, flow, acu_inlet, acu_outlet, server_inlet, server_outlet].
        If disabled or missing, returns zeros with shape (N, num_boundary_classes).
    """
    if not use_boundary_features:
        return np.zeros((expected_n, num_boundary_classes), dtype=np.float32)

    mesh = pv.read(vtk_path)

    # Select arrays from the same association as the training state.
    if use_cell_data:
        if boundary_onehot_key in mesh.cell_data:
            source = mesh.cell_data
        elif boundary_class_key in mesh.cell_data:
            source = mesh.cell_data
        elif boundary_onehot_key in mesh.point_data or boundary_class_key in mesh.point_data:
            # Convert point labels/features to cell features so they align with cell centers.
            mesh = mesh.point_data_to_cell_data(pass_point_data=True)
            source = mesh.cell_data
        else:
            raise KeyError(
                f"Boundary arrays not found in {vtk_path}. Expected '{boundary_onehot_key}' "
                f"or '{boundary_class_key}' in point_data/cell_data."
            )
    else:
        if boundary_onehot_key in mesh.point_data or boundary_class_key in mesh.point_data:
            source = mesh.point_data
        elif boundary_onehot_key in mesh.cell_data or boundary_class_key in mesh.cell_data:
            mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
            source = mesh.point_data
        else:
            raise KeyError(
                f"Boundary arrays not found in {vtk_path}. Expected '{boundary_onehot_key}' "
                f"or '{boundary_class_key}' in point_data/cell_data."
            )

    if boundary_onehot_key in source:
        feats = np.asarray(source[boundary_onehot_key], dtype=np.float32)
        if feats.ndim == 1:
            # Defensive fallback if a writer flattened the array.
            if feats.size == expected_n * num_boundary_classes:
                feats = feats.reshape(expected_n, num_boundary_classes)
            else:
                raise ValueError(
                    f"{boundary_onehot_key} has shape {feats.shape}, cannot reshape to "
                    f"({expected_n}, {num_boundary_classes})"
                )
        if feats.shape[1] != num_boundary_classes:
            raise ValueError(
                f"{boundary_onehot_key} has {feats.shape[1]} channels, expected {num_boundary_classes}"
            )
    elif boundary_class_key in source:
        cls = np.asarray(source[boundary_class_key]).astype(np.int64).reshape(-1)
        cls = np.clip(cls, 0, num_boundary_classes - 1)
        feats = np.zeros((cls.shape[0], num_boundary_classes), dtype=np.float32)
        feats[np.arange(cls.shape[0]), cls] = 1.0
    else:
        raise KeyError(f"Could not find boundary feature arrays in {vtk_path}")

    if feats.shape[0] != expected_n:
        raise ValueError(
            f"Boundary feature count mismatch in {vtk_path}: got {feats.shape[0]}, expected {expected_n}. "
            f"Check whether physical fields are point_data or cell_data and whether boundary arrays were saved correctly."
        )

    return feats.astype(np.float32)


def compute_acu_outlet_geometry_features(
    mesh: pv.DataSet,
    query_points_world: np.ndarray,
    scale: np.ndarray,
    boundary_onehot_key: str = "boundary_types",
    boundary_class_key: str = "boundary_type",
    acu_outlet_class_index: int = 3,
) -> np.ndarray:
    """Return normalized distance and direction to the nearest ACU outlet.

    These four static geometry channels use only the fixed mesh and boundary
    labels; no future target state is used.
    """
    if boundary_onehot_key in mesh.point_data:
        labels, locations = np.asarray(mesh.point_data[boundary_onehot_key]), mesh.points
    elif boundary_class_key in mesh.point_data:
        labels, locations = np.asarray(mesh.point_data[boundary_class_key]).reshape(-1), mesh.points
    elif boundary_onehot_key in mesh.cell_data or boundary_class_key in mesh.cell_data:
        locations = mesh.cell_centers().points
        labels = np.asarray(mesh.cell_data[boundary_onehot_key] if boundary_onehot_key in mesh.cell_data else mesh.cell_data[boundary_class_key])
    else:
        raise KeyError("ACU outlet geometry requires a boundary-type array")
    classes = labels.argmax(axis=1) if labels.ndim == 2 else labels.astype(np.int64).reshape(-1)
    outlet_points = np.asarray(locations, dtype=np.float32)[classes == int(acu_outlet_class_index)]
    if outlet_points.size == 0:
        raise ValueError("ACU outlet geometry requested but no ACU outlet points were found")
    distance, nearest = cKDTree(outlet_points).query(query_points_world)
    norm_scale = max(float(scale[0]), 1e-8)
    return np.concatenate([
        (distance / norm_scale).reshape(-1, 1),
        (outlet_points[nearest] - query_points_world) / norm_scale,
    ], axis=1).astype(np.float32)


# ============================================================
# GEOMETRY HELPERS
# ============================================================
def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pmin = points.min(axis=0, keepdims=True)
    pmax = points.max(axis=0, keepdims=True)
    center = 0.5 * (pmin + pmax)
    scale = np.max(pmax - pmin) / 2.0
    scale = max(float(scale), 1e-8)
    points_norm = (points - center) / scale
    return points_norm.astype(np.float32), center.astype(np.float32), np.array([scale], dtype=np.float32)


def build_knn_graph(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(points)
    _, idxs = tree.query(points, k=k + 1)

    src_list = []
    dst_list = []
    edge_attr_list = []

    for i in range(len(points)):
        neigh = idxs[i, 1:]
        for j in neigh:
            rel = points[j] - points[i]
            dist = np.linalg.norm(rel)
            src_list.append(j)
            dst_list.append(i)
            edge_attr_list.append([rel[0], rel[1], rel[2], dist])

    edge_index = np.stack([np.asarray(src_list), np.asarray(dst_list)], axis=0).astype(np.int64)
    edge_attr = np.asarray(edge_attr_list, dtype=np.float32)
    return edge_index, edge_attr


def build_uniform_grid(resolution: int = 32):
    xs = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    zs = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return grid_points, np.array([resolution, resolution, resolution], dtype=np.int32)


def compute_grid_sdf_from_surface(mesh: pv.DataSet, center: np.ndarray, scale: np.ndarray, grid_resolution: int) -> np.ndarray:
    # Pin PyVista's current surface-extraction behavior so future default
    # changes cannot silently alter the signed-distance feature geometry.
    surface = mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()
    surf_pts = surface.points.astype(np.float32)

    surf_pts_norm = (surf_pts - center) / scale[0]
    surface_norm = surface.copy()
    surface_norm.points = surf_pts_norm

    grid_points, _ = build_uniform_grid(grid_resolution)
    grid_cloud = pv.PolyData(grid_points)
    grid_cloud = grid_cloud.compute_implicit_distance(surface_norm)
    sdf = np.asarray(grid_cloud["implicit_distance"], dtype=np.float32).reshape(-1, 1)
    return sdf


# ============================================================
# GRID PROJECTION
# ============================================================
def trilinear_weights_and_indices(points: torch.Tensor, grid_shape: torch.Tensor):
    nx, ny, nz = [int(v.item()) for v in grid_shape]

    gx = (points[:, 0] + 1.0) * 0.5 * (nx - 1)
    gy = (points[:, 1] + 1.0) * 0.5 * (ny - 1)
    gz = (points[:, 2] + 1.0) * 0.5 * (nz - 1)

    x0 = torch.floor(gx).long().clamp(0, nx - 1)
    y0 = torch.floor(gy).long().clamp(0, ny - 1)
    z0 = torch.floor(gz).long().clamp(0, nz - 1)

    x1 = (x0 + 1).clamp(0, nx - 1)
    y1 = (y0 + 1).clamp(0, ny - 1)
    z1 = (z0 + 1).clamp(0, nz - 1)

    dx = gx - x0.float()
    dy = gy - y0.float()
    dz = gz - z0.float()

    corners = torch.stack([
        torch.stack([x0, y0, z0], dim=-1),
        torch.stack([x0, y0, z1], dim=-1),
        torch.stack([x0, y1, z0], dim=-1),
        torch.stack([x0, y1, z1], dim=-1),
        torch.stack([x1, y0, z0], dim=-1),
        torch.stack([x1, y0, z1], dim=-1),
        torch.stack([x1, y1, z0], dim=-1),
        torch.stack([x1, y1, z1], dim=-1),
    ], dim=1)

    weights = torch.stack([
        (1 - dx) * (1 - dy) * (1 - dz),
        (1 - dx) * (1 - dy) * dz,
        (1 - dx) * dy * (1 - dz),
        (1 - dx) * dy * dz,
        dx * (1 - dy) * (1 - dz),
        dx * (1 - dy) * dz,
        dx * dy * (1 - dz),
        dx * dy * dz,
    ], dim=1)

    return corners, weights


def splat_to_grid(points: torch.Tensor, feats: torch.Tensor, grid_shape: torch.Tensor) -> torch.Tensor:
    points = points.float()
    feats = feats.float()

    device = points.device
    _, c = feats.shape
    nx, ny, nz = [int(v.item()) for v in grid_shape]

    corners, weights = trilinear_weights_and_indices(points, grid_shape)

    grid = torch.zeros((c, nx, ny, nz), device=device, dtype=torch.float32)
    norm = torch.zeros((1, nx, ny, nz), device=device, dtype=torch.float32)

    for k in range(8):
        idx = corners[:, k]
        w = weights[:, k:k + 1].float()

        x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
        contrib = feats * w

        for ch in range(c):
            grid[ch].index_put_((x, y, z), contrib[:, ch], accumulate=True)

        norm[0].index_put_((x, y, z), w[:, 0], accumulate=True)

    grid = grid / (norm + 1e-8)
    return grid.unsqueeze(0)


def sample_from_grid(points: torch.Tensor, grid_feats: torch.Tensor, grid_shape: torch.Tensor) -> torch.Tensor:
    points = points.float()
    grid_feats = grid_feats.float().squeeze(0)
    c = grid_feats.shape[0]

    corners, weights = trilinear_weights_and_indices(points, grid_shape)
    out = torch.zeros((points.shape[0], c), device=points.device, dtype=torch.float32)

    for k in range(8):
        idx = corners[:, k]
        w = weights[:, k:k + 1].float()
        x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
        vals = grid_feats[:, x, y, z].permute(1, 0)
        out += vals * w

    return out


# ============================================================
# DATASET
# ============================================================
class VTKMGNHybridDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        val_ratio: float,
        seed: int,
        grid_resolution: int,
        k_neighbors: int,
        velocity_keys: List[str],
        pressure_keys: List[str],
        temperature_keys: List[str],
        max_cases: int = -1,
        train_query_subsample: int = -1,
        rollout_steps: int = 1,
        use_boundary_features: bool = True,
        boundary_onehot_key: str = "boundary_onehot",
        boundary_class_key: str = "boundary_class",
        boundary_feature_dim: int = 6,
        outlet_geometry_features: bool = False,
        acu_outlet_class_index: int = 3,
        compute_grid_sdf: bool = True,
        train_count: int = 10,
        valid_count: int = 2,
        test_count: int = 1,
        explicit_case_split_path: str = "",
        score_edge_mode: str = "aligned",
    ):
        super().__init__()
        assert split in ["train", "valid", "test"]

        self.split = split
        self.grid_resolution = int(grid_resolution)
        self.k_neighbors = int(k_neighbors)
        self.velocity_keys = list(velocity_keys)
        self.pressure_keys = list(pressure_keys)
        self.temperature_keys = list(temperature_keys)
        self.train_query_subsample = int(train_query_subsample)
        self.rollout_steps = int(rollout_steps)
        self.use_boundary_features = bool(use_boundary_features)
        self.boundary_onehot_key = str(boundary_onehot_key)
        self.boundary_class_key = str(boundary_class_key)
        self.boundary_feature_dim = int(boundary_feature_dim)
        self.outlet_geometry_features = bool(outlet_geometry_features)
        self.acu_outlet_class_index = int(acu_outlet_class_index)
        self.compute_grid_sdf = bool(compute_grid_sdf)
        self.train_count = int(train_count)
        self.valid_count = int(valid_count)
        self.test_count = int(test_count)
        self.explicit_case_split_path = str(explicit_case_split_path)
        self.score_edge_mode = validate_score_edge_mode(score_edge_mode)

        case_dirs = list_case_dirs(data_root)
        if max_cases > 0:
            case_dirs = case_dirs[:max_cases]

        if self.explicit_case_split_path:
            train_case_dirs, valid_case_dirs, test_case_dirs, _ = load_explicit_case_split(
                case_dirs,
                self.explicit_case_split_path,
                {
                    "train_cases": self.train_count,
                    "valid_cases": self.valid_count,
                    "test_cases": self.test_count,
                },
            )
        else:
            train_case_dirs, valid_case_dirs, test_case_dirs = split_case_dirs_three_way(
                case_dirs=case_dirs, seed=seed, train_count=self.train_count, valid_count=self.valid_count, test_count=self.test_count
            )
        if split == "train":
            self.case_dirs = train_case_dirs
        elif split == "valid":
            self.case_dirs = valid_case_dirs
        else:
            self.case_dirs = test_case_dirs

        rng = random.Random(seed)
        self.samples: List[Tuple[str, int]] = []

        for case_dir in self.case_dirs:
            vtk_files = list_time_vtk_files(case_dir)
            max_start = len(vtk_files) - (self.rollout_steps + 1)
            if max_start < 0:
                continue

            start_ids = list(range(max_start + 1))
            rng.shuffle(start_ids)

            for t in start_ids:
                self.samples.append((case_dir, t))

        rng.shuffle(self.samples)
        self.case_cache: Dict[str, Dict] = {}
        self.state_cache: Dict[Tuple[str, int], Dict] = {}

    def __len__(self):
        return len(self.samples)

    def _load_case_static(self, case_dir: str) -> Dict:
        if case_dir in self.case_cache:
            return self.case_cache[case_dir]

        vtk_files = list_time_vtk_files(case_dir)
        ref_mesh = pv.read(vtk_files[0])
        ref_coords, _, use_cell_data = read_vtk_state(
            vtk_files[0],
            self.velocity_keys,
            self.pressure_keys,
            self.temperature_keys,
        )

        boundary_features = read_vtk_boundary_features(
            vtk_files[0],
            use_cell_data=use_cell_data,
            expected_n=ref_coords.shape[0],
            use_boundary_features=self.use_boundary_features,
            boundary_onehot_key=self.boundary_onehot_key,
            boundary_class_key=self.boundary_class_key,
            num_boundary_classes=self.boundary_feature_dim,
        )

        query_points_norm, center, scale = normalize_points(ref_coords)
        if self.outlet_geometry_features:
            outlet_features = compute_acu_outlet_geometry_features(
                ref_mesh,
                ref_coords,
                scale,
                boundary_onehot_key=self.boundary_onehot_key,
                boundary_class_key=self.boundary_class_key,
                acu_outlet_class_index=self.acu_outlet_class_index,
            )
            boundary_features = np.concatenate([boundary_features, outlet_features], axis=1)
        if (
            self.split == "train"
            and self.train_query_subsample > 0
            and len(query_points_norm) > self.train_query_subsample
        ):
            # Every training __getitem__ below samples nodes and rebuilds KNN.
            # Avoid constructing a 100k-node graph here only to discard it.
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, 4), dtype=np.float32)
        else:
            edge_index, edge_attr = build_knn_graph(
                query_points_norm, self.k_neighbors
            )
        _, grid_shape = build_uniform_grid(self.grid_resolution)
        if self.compute_grid_sdf:
            grid_sdf = compute_grid_sdf_from_surface(
                ref_mesh, center, scale, self.grid_resolution
            )
        else:
            # Graph-only baselines do not consume the regular-grid SDF. Keep a
            # lightweight placeholder so the shared sample contract is intact.
            grid_sdf = np.zeros((1,), dtype=np.float32)

        self.case_cache[case_dir] = {
            "vtk_files": vtk_files,
            "query_points_world": ref_coords.astype(np.float32),
            "query_points_norm": query_points_norm.astype(np.float32),
            "center": center.astype(np.float32),
            "scale": scale.astype(np.float32),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "grid_shape": grid_shape.astype(np.int32),
            "grid_sdf": grid_sdf.astype(np.float32),
            "boundary_features": boundary_features.astype(np.float32),
            "use_cell_data": use_cell_data,
        }
        return self.case_cache[case_dir]

    def _load_state(self, case_dir: str, t: int) -> np.ndarray:
        key = (case_dir, t)
        if key in self.state_cache:
            return self.state_cache[key]["state"]

        static = self._load_case_static(case_dir)
        coords, state, _ = read_vtk_state(
            static["vtk_files"][t],
            self.velocity_keys,
            self.pressure_keys,
            self.temperature_keys,
        )

        if coords.shape[0] != static["query_points_world"].shape[0]:
            raise ValueError(f"Point/cell count mismatch in {case_dir} at time {t}")

        self.state_cache[key] = {"state": state.astype(np.float32)}
        return self.state_cache[key]["state"]

    def __getitem__(self, idx):
        case_dir, t = self.samples[idx]
        static = self._load_case_static(case_dir)

        q = static["query_points_norm"]
        # These immutable IDs survive the sampled-graph reindexing below and
        # make A10S independent of incidental KNN edge ordering.
        node_ids = np.arange(len(q), dtype=np.int64)
        gs = static["grid_shape"]
        sdf = static["grid_sdf"]
        edge_index = static["edge_index"]
        edge_attr = static["edge_attr"]
        boundary_features = static["boundary_features"]

        states = [self._load_state(case_dir, t + k) for k in range(self.rollout_steps + 1)]
        s0 = states[0]
        future_states = states[1:]  # list of rollout_steps arrays

        if self.split == "train" and self.train_query_subsample > 0 and len(q) > self.train_query_subsample:
            idx_q = np.random.choice(len(q), self.train_query_subsample, replace=False)
            idx_q = np.sort(idx_q)

            q = q[idx_q]
            node_ids = node_ids[idx_q]
            boundary_features = boundary_features[idx_q]
            s0 = s0[idx_q]
            future_states = [s[idx_q] for s in future_states]

            # Rebuild KNN on the sampled points so each sampled graph remains
            # well-formed and has a predictable neighborhood structure.
            edge_index, edge_attr = build_knn_graph(q, self.k_neighbors)

        if self.score_edge_mode == "deranged":
            score_edge_permutation = destination_local_derangement(
                edge_index,
                node_ids,
                os.path.basename(os.path.normpath(case_dir)),
            )
        else:
            score_edge_permutation = np.arange(edge_attr.shape[0], dtype=np.int64)

        future_states = np.stack(future_states, axis=0)  # (K, N, 5)

        return {
            "query_points": torch.from_numpy(q),
            "grid_shape": torch.from_numpy(gs),
            "grid_sdf": torch.from_numpy(sdf),
            "edge_index": torch.from_numpy(edge_index).long(),
            "edge_attr": torch.from_numpy(edge_attr).float(),
            "score_edge_permutation": torch.from_numpy(score_edge_permutation).long(),
            "boundary_features": torch.from_numpy(boundary_features).float(),
            "state_t": torch.from_numpy(s0).float(),
            "future_states": torch.from_numpy(future_states).float(),
        }


# ============================================================
# COLLATE / NORMALIZER
# ============================================================
def collate_list(batch):
    """
    Keep each sample as an independent graph.

    This lets DataLoader use batch_size > 1 even when different cases have
    different numbers of nodes / edges. We then loop over the samples inside
    each mini-batch and average their losses before backpropagation.
    """
    if len(batch) == 0:
        raise ValueError("Empty batch received.")
    return batch


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's per-worker deterministic seed."""

    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class ChannelNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mean = mean.float()
        self.std = std.float()
        self.eps = eps

    def to(self, device: torch.device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean


def compute_normalization_stats(dataset: Dataset, max_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    n = min(max_samples, len(dataset))
    if n == 0:
        raise RuntimeError("Training dataset is empty, cannot compute normalization stats.")

    states = []
    for i in range(n):
        sample = dataset[i]
        states.append(sample["state_t"].float())

    stacked = torch.cat(states, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    std = torch.clamp(std, min=1e-6)
    return mean, std


# ============================================================
# MODEL
# ============================================================


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 2, activation: str = "silu", layer_norm: bool = False):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        act_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }
        if activation not in act_map:
            raise ValueError(f"Unsupported activation: {activation}")
        act_cls = act_map[activation]

        dims = [in_dim]
        if num_layers == 1:
            dims.append(out_dim)
        else:
            dims.extend([hidden_dim] * (num_layers - 1))
            dims.append(out_dim)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleGNNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        mlp_layers: int = 2,
        activation: str = "silu",
        layer_norm: bool = False,
        residual: bool = True,
        aggregation: str = "mean",
        use_edge_features: bool = True,
        use_attention: bool = False,
        attention_mode: str | None = None,
        attention_hidden_dim: int = 32,
        attention_temperature: float = 1.0,
        score_edge_mode: str = "aligned",
    ):
        super().__init__()
        self.residual = residual
        self.aggregation = aggregation
        self.use_edge_features = use_edge_features
        if attention_mode is None:
            attention_mode = "learned" if use_attention else "none"
        self.attention_mode = str(attention_mode)
        if self.attention_mode not in {"none", "uniform", "learned"}:
            raise ValueError("attention_mode must be one of: none, uniform, learned")
        if bool(use_attention) != (self.attention_mode != "none"):
            raise ValueError("use_attention and attention_mode disagree")
        self.use_attention = self.attention_mode != "none"
        self.attention_temperature = max(float(attention_temperature), 1e-6)
        self.score_edge_mode = validate_score_edge_mode(score_edge_mode)
        if self.score_edge_mode != "aligned" and (
            self.attention_mode != "learned" or not self.use_edge_features
        ):
            raise ValueError(
                "non-aligned score-edge controls require learned attention and edge features"
            )

        msg_in_dim = hidden_dim + (edge_feat_dim if use_edge_features else 0)
        self.message_mlp = MLP(
            in_dim=msg_in_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=mlp_layers,
            activation=activation,
            layer_norm=layer_norm,
        )
        if self.use_attention:
            att_in_dim = hidden_dim * 2 + (edge_feat_dim if use_edge_features else 0)
            self.attn_mlp = MLP(
                in_dim=att_in_dim,
                out_dim=1,
                hidden_dim=attention_hidden_dim,
                num_layers=2,
                activation=activation,
                layer_norm=False,
            )
        self.node_mlp = MLP(
            in_dim=hidden_dim * 2,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=mlp_layers,
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        score_edge_permutation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]

        if self.use_edge_features:
            msg_in = torch.cat([x[src], edge_attr], dim=-1)
            if self.attention_mode == "learned":
                if self.score_edge_mode == "deranged":
                    if score_edge_permutation is None:
                        raise ValueError("deranged score mode requires a score-edge permutation")
                    # The dataset and rollout constructor construct and
                    # verify the full destination-local permutation once.
                    # Re-sorting E edges in every GNN block would make
                    # room-scale rollouts prohibitively expensive and
                    # condition-dependent.
                score_edge_attr = score_edge_attributes(
                    edge_attr,
                    self.score_edge_mode,
                    score_edge_permutation,
                    validate_permutation=False,
                )
                att_in = torch.cat([x[src], x[dst], score_edge_attr], dim=-1)
        else:
            msg_in = x[src]
            if self.attention_mode == "learned":
                att_in = torch.cat([x[src], x[dst]], dim=-1)
        messages = self.message_mlp(msg_in)

        if self.attention_mode == "learned":
            att_logits = self.attn_mlp(att_in) / self.attention_temperature
            att_logits = att_logits.squeeze(-1)
            # Some CUDA autocast kernels promote exp to fp32. Keep the
            # scatter source consistent with the fp16/bf16 node accumulator.
            att_exp = torch.exp(att_logits - att_logits.detach().amax()).to(x.dtype)
            att_denom = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
            att_denom.index_add_(0, dst, att_exp)
            att_weights = (att_exp / att_denom[dst].clamp_min(1e-8)).unsqueeze(-1)
            messages = messages * att_weights
        elif self.attention_mode == "uniform":
            # This is the exact fixed-score counterpart of learned attention:
            # incoming weights are 1/degree before the shared aggregation
            # rule.  The scorer module remains instantiated so its parameter
            # count matches the learned score-control arms.
            degree = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            messages = messages * (1.0 / degree[dst].clamp_min(1.0)).unsqueeze(-1)

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages)

        if self.aggregation == "mean":
            deg = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
            deg.index_add_(0, dst, torch.ones((dst.shape[0], 1), device=x.device, dtype=x.dtype))
            agg = agg / deg.clamp_min(1.0)
        elif self.aggregation != "sum":
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")

        dx = self.node_mlp(torch.cat([x, agg], dim=-1))
        if self.residual:
            return x + dx
        return dx


class SimpleGraphNet(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        node_out_dim: int,
        hidden_dim: int = 128,
        processor_layers: int = 4,
        mlp_layers: int = 2,
        activation: str = "silu",
        layer_norm: bool = False,
        residual: bool = True,
        aggregation: str = "mean",
        use_edge_features: bool = True,
        use_attention: bool = False,
        attention_mode: str | None = None,
        attention_hidden_dim: int = 32,
        attention_temperature: float = 1.0,
        score_edge_mode: str = "aligned",
    ):
        super().__init__()
        self.node_encoder = MLP(
            in_dim=node_in_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=mlp_layers,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.processor = nn.ModuleList([
            SimpleGNNLayer(
                hidden_dim=hidden_dim,
                edge_feat_dim=edge_in_dim,
                mlp_layers=mlp_layers,
                activation=activation,
                layer_norm=layer_norm,
                residual=residual,
                aggregation=aggregation,
                use_edge_features=use_edge_features,
                use_attention=use_attention,
                attention_mode=attention_mode,
                attention_hidden_dim=attention_hidden_dim,
                attention_temperature=attention_temperature,
                score_edge_mode=score_edge_mode,
            )
            for _ in range(processor_layers)
        ])
        self.node_decoder = MLP(
            in_dim=hidden_dim,
            out_dim=node_out_dim,
            hidden_dim=hidden_dim,
            num_layers=mlp_layers,
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(
        self,
        node_x: torch.Tensor,
        edge_attr: torch.Tensor,
        graph: Data,
        score_edge_permutation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.node_encoder(node_x)
        edge_index = graph.edge_index
        for block in self.processor:
            x = block(x, edge_index, edge_attr, score_edge_permutation)
        return self.node_decoder(x)


class GNNFNOGNNHybrid(nn.Module):
    def __init__(
        self,
        state_dim: int = 5,
        boundary_feature_dim: int = 6,
        outlet_geometry_feature_dim: int = 0,
        latent_dim: int = 64,
        fno_modes: int = 12,
        fno_layers: int = 4,
        fno_padding: int = 8,
        fno_decoder_layers: int = 1,
        fno_decoder_layer_size: int = 64,
        encoder_hidden_dim: int = 128,
        encoder_processor_layers: int = 4,
        encoder_mlp_layers: int = 2,
        decoder_hidden_dim: int = 128,
        decoder_processor_layers: int = 4,
        decoder_mlp_layers: int = 2,
        gnn_activation: str = "silu",
        gnn_layer_norm: bool = False,
        gnn_residual: bool = True,
        gnn_aggregation: str = "mean",
        gnn_use_edge_features: bool = True,
        gnn_use_attention: bool = False,
        gnn_attention_mode: str | None = None,
        gnn_attention_hidden_dim: int = 32,
        gnn_attention_temperature: float = 1.0,
        gnn_score_edge_mode: str = "aligned",
    ):
        super().__init__()

        self.boundary_feature_dim = int(boundary_feature_dim) + int(outlet_geometry_feature_dim)

        self.encoder = SimpleGraphNet(
            node_in_dim=state_dim + 3 + self.boundary_feature_dim,
            edge_in_dim=4,
            node_out_dim=latent_dim,
            hidden_dim=encoder_hidden_dim,
            processor_layers=encoder_processor_layers,
            mlp_layers=encoder_mlp_layers,
            activation=gnn_activation,
            layer_norm=gnn_layer_norm,
            residual=gnn_residual,
            aggregation=gnn_aggregation,
            use_edge_features=gnn_use_edge_features,
            use_attention=gnn_use_attention,
            attention_mode=gnn_attention_mode,
            attention_hidden_dim=gnn_attention_hidden_dim,
            attention_temperature=gnn_attention_temperature,
            score_edge_mode=gnn_score_edge_mode,
        )

        self.fno = FNO(
            in_channels=latent_dim + 1,
            out_channels=latent_dim,
            decoder_layers=fno_decoder_layers,
            decoder_layer_size=fno_decoder_layer_size,
            dimension=3,
            latent_channels=latent_dim,
            num_fno_layers=fno_layers,
            num_fno_modes=fno_modes,
            padding=fno_padding,
        )

        self.decoder = SimpleGraphNet(
            node_in_dim=latent_dim + 3,
            edge_in_dim=4,
            node_out_dim=state_dim,
            hidden_dim=decoder_hidden_dim,
            processor_layers=decoder_processor_layers,
            mlp_layers=decoder_mlp_layers,
            activation=gnn_activation,
            layer_norm=gnn_layer_norm,
            residual=gnn_residual,
            aggregation=gnn_aggregation,
            use_edge_features=gnn_use_edge_features,
            use_attention=gnn_use_attention,
            attention_mode=gnn_attention_mode,
            attention_hidden_dim=gnn_attention_hidden_dim,
            attention_temperature=gnn_attention_temperature,
            score_edge_mode=gnn_score_edge_mode,
        )

    def _forward_single(
        self,
        query_points: torch.Tensor,
        state_t: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        grid_shape: torch.Tensor,
        grid_sdf: torch.Tensor,
        boundary_features: torch.Tensor,
        score_edge_permutation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        graph = Data(edge_index=edge_index, num_nodes=query_points.shape[0])

        enc_x = torch.cat([state_t, query_points.float(), boundary_features.float()], dim=-1)
        latent_nodes = self.encoder(
            enc_x, edge_attr, graph, score_edge_permutation
        )

        grid_latent = splat_to_grid(query_points, latent_nodes, grid_shape)

        nx, ny, nz = [int(v.item()) for v in grid_shape]
        sdf_grid = grid_sdf.float().view(nx, ny, nz, 1).permute(3, 0, 1, 2).unsqueeze(0)

        fno_in = torch.cat([grid_latent, sdf_grid], dim=1)
        grid_out = self.fno(fno_in)
        sampled_nodes = sample_from_grid(query_points, grid_out, grid_shape)

        dec_x = torch.cat([sampled_nodes, query_points.float()], dim=-1)
        pred = self.decoder(dec_x, edge_attr, graph, score_edge_permutation)
        return pred

    def forward(
        self,
        query_points: torch.Tensor,
        state_t: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        grid_shape: torch.Tensor,
        grid_sdf: torch.Tensor,
        boundary_features: torch.Tensor,
        score_edge_permutation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_points.dim() == 2:
            return self._forward_single(
                query_points,
                state_t,
                edge_index,
                edge_attr,
                grid_shape,
                grid_sdf,
                boundary_features,
                score_edge_permutation,
            )
        raise ValueError("This model expects single-sample tensors; batch over samples in the training loop.")

# ============================================================
# CKPT / LOSS
# ============================================================
def rel_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = torch.sum((pred - target) ** 2)
    den = torch.sum(target ** 2)
    return torch.sqrt(num / (den + eps))


def make_interior_mask(boundary_features: torch.Tensor, interior_class_index: int = 1) -> torch.Tensor:
    """Return boolean mask for interior / flow nodes.

    Default assumes boundary_features one-hot channels are:
    [wall, flow/interior, acu_inlet, acu_outlet, server_inlet, server_outlet].
    Thus interior_class_index=1.
    """
    if boundary_features is None or boundary_features.numel() == 0:
        return None
    idx = int(interior_class_index)
    if boundary_features.dim() != 2 or idx < 0 or idx >= boundary_features.shape[1]:
        return None
    return boundary_features[:, idx] > 0.5


def rel_l2_loss_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Relative L2 loss optionally computed only on selected nodes."""
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if int(mask.sum().item()) > 0:
            pred = pred[mask]
            target = target[mask]
    return rel_l2_loss(pred, target, eps=eps)


def impose_boundary_from_target(
    pred_phys: torch.Tensor,
    target_phys: torch.Tensor,
    boundary_features: torch.Tensor,
    interior_class_index: int = 1,
) -> torch.Tensor:
    """Semi-teacher-force boundary nodes by copying target/GT boundary values.

    Interior nodes are kept from prediction. Non-interior nodes are overwritten
    with target_phys. This mirrors TIGNN-style hard boundary enforcement during rollout.
    """
    interior_mask = make_interior_mask(boundary_features, interior_class_index)
    if interior_mask is None:
        return pred_phys
    boundary_mask = ~interior_mask
    if int(boundary_mask.sum().item()) == 0:
        return pred_phys
    out = pred_phys.clone()
    out[boundary_mask] = target_phys[boundary_mask]
    return out


def edge_channel_gradient_loss(
    pred_phys: torch.Tensor,
    target_phys: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    channels,
    eps: float = 1e-8,
    relative: bool = True,
) -> torch.Tensor:
    """
    Edge-based gradient matching loss along KNN edges.

    For each edge (src, dst), this compares field differences divided by
    edge distance. This is a directional finite-difference gradient proxy:

        grad_edge(u) = (u_src - u_dst) / ||x_src - x_dst||

    channels can be a single int, e.g. 3 for pressure, or a list/tuple,
    e.g. [0, 1, 2] for velocity vector components.
    """
    if isinstance(channels, int):
        channel_idx = torch.tensor([channels], device=pred_phys.device, dtype=torch.long)
    else:
        channel_idx = torch.tensor(list(channels), device=pred_phys.device, dtype=torch.long)

    src, dst = edge_index[0], edge_index[1]
    dist = edge_attr[:, 3].clamp_min(eps).unsqueeze(-1)

    pred_vals = pred_phys[:, channel_idx]
    target_vals = target_phys[:, channel_idx]

    pred_grad = (pred_vals[src] - pred_vals[dst]) / dist
    target_grad = (target_vals[src] - target_vals[dst]) / dist

    mse = torch.mean((pred_grad - target_grad) ** 2)
    if relative:
        denom = torch.mean(target_grad ** 2).clamp_min(eps)
        return mse / denom
    return mse


def add_optional_gradient_losses(
    base_loss: torch.Tensor,
    pred_phys: torch.Tensor,
    target_phys: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    use_pressure_grad_loss: bool = False,
    pressure_grad_weight: float = 0.05,
    pressure_grad_relative: bool = True,
    use_velocity_grad_loss: bool = False,
    velocity_grad_weight: float = 0.01,
    velocity_grad_relative: bool = True,
    use_temperature_grad_loss: bool = False,
    temperature_grad_weight: float = 0.01,
    temperature_grad_relative: bool = True,
) -> torch.Tensor:
    """Add independently configurable pressure/velocity/temperature gradient losses."""
    loss = base_loss

    if use_pressure_grad_loss and float(pressure_grad_weight) > 0.0:
        grad_loss = edge_channel_gradient_loss(
            pred_phys, target_phys, edge_index, edge_attr,
            channels=3, relative=pressure_grad_relative
        )
        loss = loss + float(pressure_grad_weight) * grad_loss

    if use_velocity_grad_loss and float(velocity_grad_weight) > 0.0:
        grad_loss = edge_channel_gradient_loss(
            pred_phys, target_phys, edge_index, edge_attr,
            channels=[0, 1, 2], relative=velocity_grad_relative
        )
        loss = loss + float(velocity_grad_weight) * grad_loss

    if use_temperature_grad_loss and float(temperature_grad_weight) > 0.0:
        grad_loss = edge_channel_gradient_loss(
            pred_phys, target_phys, edge_index, edge_attr,
            channels=4, relative=temperature_grad_relative
        )
        loss = loss + float(temperature_grad_weight) * grad_loss

    return loss

def per_channel_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=0))


def save_plain_checkpoint(path, epoch, model, optimizer, scheduler, best_val=None, norm_mean=None, norm_std=None):
    os.makedirs(path, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val": best_val,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }
    torch.save(ckpt, os.path.join(path, "latest.pt"))
    torch.save(ckpt, os.path.join(path, f"epoch_{epoch:04d}.pt"))


def load_plain_checkpoint(path, device, model, optimizer=None, scheduler=None):
    latest_path = os.path.join(path, "latest.pt")
    if not os.path.isfile(latest_path):
        return 0, None, None, None

    ckpt = torch.load(latest_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return (
        int(ckpt.get("epoch", 0)),
        ckpt.get("best_val", None),
        ckpt.get("norm_mean", None),
        ckpt.get("norm_std", None),
    )


# ============================================================
# TRAIN / VALID
# ============================================================
def train_one_epoch(
    model, train_loader, optimizer, scaler, dist, use_amp, normalizer, logger,
    rollout_steps: int,
    teacher_forcing: bool,
    use_multistep_loss: bool,
    train_interior_only: bool = False,
    interior_class_index: int = 1,
    semi_teacher_force_boundary: bool = False,
    use_pressure_grad_loss: bool = False,
    pressure_grad_weight: float = 0.05,
    pressure_grad_relative: bool = True,
    use_velocity_grad_loss: bool = False,
    velocity_grad_weight: float = 0.01,
    velocity_grad_relative: bool = True,
    use_temperature_grad_loss: bool = False,
    temperature_grad_weight: float = 0.01,
    temperature_grad_relative: bool = True,
):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_samples in train_loader:
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0

        with torch.amp.autocast("cuda", enabled=use_amp):
            for sample in batch_samples:
                q = sample["query_points"].to(dist.device).float()
                gs = sample["grid_shape"].to(dist.device)
                sdf = sample["grid_sdf"].to(dist.device).float()
                edge_index = sample["edge_index"].to(dist.device)
                edge_attr = sample["edge_attr"].to(dist.device).float()
                score_edge_permutation = sample["score_edge_permutation"].to(dist.device)
                boundary_features = sample["boundary_features"].to(dist.device).float()
                interior_mask = make_interior_mask(boundary_features, interior_class_index) if train_interior_only else None
                s0 = sample["state_t"].to(dist.device).float()
                future_states = sample["future_states"].to(dist.device).float()  # (K, N, 5)

                current_state_phys = s0
                if not use_multistep_loss:
                    target_phys = future_states[0]
                    current_state_norm = normalizer.encode(current_state_phys)
                    target_norm = normalizer.encode(target_phys)
                    pred_norm = model(
                        q, current_state_norm, edge_index, edge_attr, gs, sdf,
                        boundary_features, score_edge_permutation,
                    )
                    sample_loss = rel_l2_loss_masked(pred_norm, target_norm, interior_mask)
                    pred_phys = normalizer.decode(pred_norm)
                    sample_loss = add_optional_gradient_losses(
                        sample_loss, pred_phys, target_phys, edge_index, edge_attr,
                        use_pressure_grad_loss, pressure_grad_weight, pressure_grad_relative,
                        use_velocity_grad_loss, velocity_grad_weight, velocity_grad_relative,
                        use_temperature_grad_loss, temperature_grad_weight, temperature_grad_relative,
                    )
                else:
                    sample_total_loss = 0.0
                    for k in range(rollout_steps):
                        target_phys = future_states[k]
                        current_state_norm = normalizer.encode(current_state_phys)
                        target_norm = normalizer.encode(target_phys)

                        pred_norm = model(
                            q, current_state_norm, edge_index, edge_attr, gs, sdf,
                            boundary_features, score_edge_permutation,
                        )
                        step_loss = rel_l2_loss_masked(pred_norm, target_norm, interior_mask)
                        pred_phys = normalizer.decode(pred_norm)
                        step_loss = add_optional_gradient_losses(
                            step_loss, pred_phys, target_phys, edge_index, edge_attr,
                            use_pressure_grad_loss, pressure_grad_weight, pressure_grad_relative,
                            use_velocity_grad_loss, velocity_grad_weight, velocity_grad_relative,
                            use_temperature_grad_loss, temperature_grad_weight, temperature_grad_relative,
                        )
                        sample_total_loss = sample_total_loss + step_loss

                        if teacher_forcing:
                            current_state_phys = target_phys
                        else:
                            if semi_teacher_force_boundary:
                                current_state_phys = impose_boundary_from_target(
                                    pred_phys, target_phys, boundary_features, interior_class_index
                                )
                            else:
                                current_state_phys = pred_phys
                    sample_loss = sample_total_loss / rollout_steps

                batch_loss = batch_loss + sample_loss

            loss = batch_loss / len(batch_samples)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches += 1
        logger.log_minibatch({"loss": loss.detach()})

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate_one_epoch(model, valid_loader, dist, use_amp, normalizer):
    model.eval()

    total_rel = 0.0
    total_mse_norm = 0.0
    total_rmse_phys = None
    n_samples = 0

    for batch_samples in valid_loader:
        for sample in batch_samples:
            q = sample["query_points"].to(dist.device).float()
            gs = sample["grid_shape"].to(dist.device)
            sdf = sample["grid_sdf"].to(dist.device).float()
            edge_index = sample["edge_index"].to(dist.device)
            edge_attr = sample["edge_attr"].to(dist.device).float()
            score_edge_permutation = sample["score_edge_permutation"].to(dist.device)
            boundary_features = sample["boundary_features"].to(dist.device).float()
            s0 = sample["state_t"].to(dist.device).float()
            s1 = sample["future_states"][0].to(dist.device).float()

            s0_norm = normalizer.encode(s0)
            s1_norm = normalizer.encode(s1)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_norm = model(
                    q, s0_norm, edge_index, edge_attr, gs, sdf,
                    boundary_features, score_edge_permutation,
                )
                mse_norm = torch.mean((pred_norm - s1_norm) ** 2)
                rel = rel_l2_loss(pred_norm, s1_norm)

            pred_phys = normalizer.decode(pred_norm)
            rmse_phys = per_channel_rmse(pred_phys, s1)

            total_rel += rel.item()
            total_mse_norm += mse_norm.item()
            n_samples += 1

            if total_rmse_phys is None:
                total_rmse_phys = rmse_phys.detach().cpu()
            else:
                total_rmse_phys += rmse_phys.detach().cpu()

    n_samples = max(n_samples, 1)
    return total_rel / n_samples, total_mse_norm / n_samples, total_rmse_phys / n_samples



# ============================================================
# LOGGING / ROLLOUT HELPERS
# ============================================================
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def _cfg_get(cfg, path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not hasattr(cur, p):
            return default
        cur = getattr(cur, p)
    return cur


def score_control_run_spec(cfg) -> Optional[Tuple[str, int]]:
    """Return the canonical C70 control cell, or ``None`` for ordinary runs."""

    raw_condition = _cfg_get(cfg, "score_control.condition", None)
    if raw_condition is None:
        return None
    condition = str(raw_condition)
    seed = int(_cfg_get(cfg, "training.seed", -1))
    # This also rejects names outside the fixed condition/seed matrix.
    canonical_run_name(condition, seed)
    return condition, seed


def verify_score_control_source_lock(cfg):
    """Require a launcher-pinned, still-current C70S100 source lock.

    The production locked launcher verifies the five source members before it
    invokes this trainer and injects the raw lock digest into the resolved
    config.  Repeating that check here prevents a direct trainer invocation
    from fabricating a provenance field after bypassing the launcher.
    """

    try:
        source_lock = verify_canonical_source_lock()
    except ScoreControlIntegrityError as exc:
        raise ValueError("C70S100 source lock verification failed: {}".format(exc))
    configured_filename = str(
        _cfg_get(cfg, "score_control.source_lock_filename", "")
    )
    configured_sha256 = str(_cfg_get(cfg, "score_control.source_lock_sha256", ""))
    if configured_filename != SOURCE_LOCK_FILENAME:
        raise ValueError("C70S100 source lock filename is missing or differs")
    if configured_sha256 != source_lock.raw_sha256:
        raise ValueError("C70S100 source lock SHA-256 is missing or differs")
    return source_lock


def validate_score_control_execution_contract(cfg) -> Optional[Tuple[str, int]]:
    """Reject resume or abbreviated execution only for the locked C70 study."""

    specification = score_control_run_spec(cfg)
    if specification is None:
        return None
    verify_score_control_source_lock(cfg)
    condition, seed = specification
    expected_name = canonical_run_name(condition, seed)
    actual_name = str(_cfg_get(cfg, "output.run_name", ""))
    if actual_name != expected_name:
        raise ValueError(
            "score-control output.run_name must be {}, got {}".format(
                expected_name, actual_name
            )
        )
    if bool(_cfg_get(cfg, "training.resume", False)):
        raise ValueError("C70S100 score-control runs are fresh-only and cannot resume")
    if not bool(_cfg_get(cfg, "score_control.require_fresh_run", True)):
        raise ValueError("C70S100 score-control runs require fresh-run enforcement")
    required = {
        "training.max_epochs": (int(_cfg_get(cfg, "training.max_epochs", -1)), 100),
        "training.rollout_steps": (int(_cfg_get(cfg, "training.rollout_steps", -1)), 3),
        "data.max_train_samples": (int(_cfg_get(cfg, "data.max_train_samples", -1)), 200),
        "data.training_window_manifest": (
            str(_cfg_get(cfg, "data.training_window_manifest", "")),
            C70_TRAINING_WINDOW_MANIFEST_FILENAME,
        ),
        "test.rollout_steps": (int(_cfg_get(cfg, "test.rollout_steps", -1)), 200),
        "test.rollout_after_training": (
            bool(_cfg_get(cfg, "test.rollout_after_training", False)),
            True,
        ),
    }
    mismatch = [
        "{}={}, expected {}".format(name, actual, expected)
        for name, (actual, expected) in required.items()
        if actual != expected
    ]
    if mismatch:
        raise ValueError(
            "C70S100 score-control completion contract mismatch: " + "; ".join(mismatch)
        )
    return specification


def make_experiment_dir(cfg: DictConfig) -> str:
    root = to_absolute_path(str(_cfg_get(cfg, "output.root_dir", "./runs")))
    run_name = str(_cfg_get(cfg, "output.run_name", "simple_gnn_fno_boundary_10train_2val_1test"))
    save_dir = os.path.join(root, run_name)
    specification = validate_score_control_execution_contract(cfg)
    if specification is not None:
        condition, seed = specification
        try:
            return str(reserve_fresh_run_directory(Path(save_dir), condition, seed))
        except ScoreControlIntegrityError as exc:
            raise RuntimeError(str(exc))
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def write_split_log(path: str, train_cases, valid_cases, test_cases, cfg, split_metadata=None):
    data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(cfg.training.seed),
        "train_cases": [os.path.basename(c) for c in train_cases],
        "valid_cases": [os.path.basename(c) for c in valid_cases],
        "test_cases": [os.path.basename(c) for c in test_cases],
    }
    if split_metadata:
        data.update(dict(split_metadata))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_score_control_run_manifest(
    save_dir: str,
    cfg,
    split_info: Dict,
    training_window_selection: Optional[Dict] = None,
) -> None:
    """Record immutable inputs for an auditable matched score-control run."""

    condition = _cfg_get(cfg, "score_control.condition", None)
    if condition is None:
        return
    if not isinstance(training_window_selection, dict):
        raise ValueError("C70S100 score-control run lacks a frozen training-window selection")
    window_keys = {
        "filename",
        "raw_sha256",
        "semantic_sha256",
        "sample_sha256",
        "summary",
        "records",
    }
    if set(training_window_selection) != window_keys:
        raise ValueError("C70S100 training-window selection has an unexpected inventory")
    if training_window_selection["filename"] != C70_TRAINING_WINDOW_MANIFEST_FILENAME:
        raise ValueError("C70S100 training-window manifest filename differs")
    if training_window_selection["raw_sha256"] != str(
        _cfg_get(cfg, "score_control.training_window_manifest_raw_sha256", "")
    ):
        raise ValueError("C70S100 training-window manifest raw digest differs")
    if training_window_selection["semantic_sha256"] != str(
        _cfg_get(cfg, "score_control.training_window_semantic_sha256", "")
    ):
        raise ValueError("C70S100 training-window manifest semantic digest differs")
    source_lock = verify_score_control_source_lock(cfg)
    resolved_config = os.path.join(save_dir, "resolved_config.yaml")
    payload = {
        "schema_version": 3,
        "condition": str(condition),
        "seed": int(cfg.training.seed),
        "base_config_sha256": str(
            _cfg_get(cfg, "score_control.base_config_sha256", "")
        ),
        "resolved_config_sha256": _sha256_file(resolved_config),
        "source_lock": {
            "filename": SOURCE_LOCK_FILENAME,
            "raw_sha256": source_lock.raw_sha256,
        },
        "source_sha256": dict(source_lock.source_sha256),
        "case_split": split_info,
        "training_windows": {
            "filename": training_window_selection["filename"],
            "raw_sha256": training_window_selection["raw_sha256"],
            "semantic_sha256": training_window_selection["semantic_sha256"],
            "sample_sha256": training_window_selection["sample_sha256"],
            "summary": training_window_selection["summary"],
        },
    }
    with open(os.path.join(save_dir, "score_control_run_manifest.json"), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_prediction_like_input(step_idx, output_dir, template_mesh, state_phys, use_cell_data):
    out_mesh = template_mesh.copy(deep=True)
    target = out_mesh.cell_data if use_cell_data else out_mesh.point_data
    target.clear()
    target["Velocity(m/s)"] = np.asarray(state_phys[:, 0:3], dtype=np.float32)
    target["Pressure(Pa)"] = np.asarray(state_phys[:, 3], dtype=np.float32)
    target["Temperature(C)"] = np.asarray(state_phys[:, 4], dtype=np.float32)
    out_path = os.path.join(output_dir, f"pred_{step_idx:04d}.vtk")
    out_mesh.save(out_path)
    return out_path


def compute_rmse_np(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2, axis=0))


def compute_relative_l2_np(pred, gt, eps=1e-12):
    diff_norm = np.linalg.norm(pred - gt, axis=0)
    gt_norm = np.linalg.norm(gt, axis=0)
    return diff_norm / np.maximum(gt_norm, eps)


def compute_velocity_vector_rmse_np(pred, gt):
    return float(np.sqrt(np.mean(np.sum((pred[:, 0:3] - gt[:, 0:3]) ** 2, axis=1))))


def compute_velocity_vector_relative_l2_np(pred, gt, eps=1e-12):
    diff_norm = np.linalg.norm(pred[:, 0:3] - gt[:, 0:3])
    gt_norm = np.linalg.norm(gt[:, 0:3])
    return float(diff_norm / max(gt_norm, eps))


@torch.no_grad()
def rollout_test_case(model, normalizer, cfg, test_dataset, test_case_dir: str, output_root: str, dist):
    model.eval()
    case_name = os.path.basename(os.path.normpath(test_case_dir))
    output_dir = os.path.join(output_root, "test_rollout", case_name)
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rollout_wall_start = time.perf_counter()
    model_forward_time = 0.0

    static = test_dataset._load_case_static(test_case_dir)
    vtk_files = static["vtk_files"]
    template_mesh = pv.read(vtk_files[0])
    use_cell_data = bool(static["use_cell_data"])

    q = torch.from_numpy(static["query_points_norm"]).float().to(dist.device)
    gs = torch.from_numpy(static["grid_shape"]).to(dist.device)
    sdf = torch.from_numpy(static["grid_sdf"]).float().to(dist.device)
    edge_index = torch.from_numpy(static["edge_index"]).long().to(dist.device)
    edge_attr = torch.from_numpy(static["edge_attr"]).float().to(dist.device)
    node_ids = np.arange(static["query_points_norm"].shape[0], dtype=np.int64)
    if str(_cfg_get(cfg, "arch.gnn.score_edge_mode", "aligned")) == "deranged":
        score_edge_permutation = torch.from_numpy(
            destination_local_derangement(edge_index.detach().cpu().numpy(), node_ids, case_name)
        ).long().to(dist.device)
    else:
        score_edge_permutation = torch.arange(
            edge_attr.shape[0], device=dist.device, dtype=torch.long
        )
    boundary_features = torch.from_numpy(static["boundary_features"]).float().to(dist.device)
    metric_scope = str(_cfg_get(cfg, "test.metric_scope", "all"))
    if metric_scope not in {"all", "interior"}:
        raise ValueError("test.metric_scope must be 'all' or 'interior'")
    if metric_scope == "interior":
        metric_mask = make_interior_mask(
            boundary_features, int(_cfg_get(cfg, "training.interior_class_index", 1))
        )
        if metric_mask is None or not bool(metric_mask.any()):
            raise ValueError("interior metric scope requires a nonempty interior mask")
        metric_mask_np = metric_mask.detach().cpu().numpy()
    else:
        metric_mask_np = None
    metric_node_count = (
        int(metric_mask_np.sum()) if metric_mask_np is not None else int(q.shape[0])
    )

    current_np = test_dataset._load_state(test_case_dir, 0)
    current = torch.from_numpy(current_np).float().to(dist.device)
    max_steps = min(int(_cfg_get(cfg, "test.rollout_steps", 200)), len(vtk_files) - 1)
    configured_vtk_steps = _cfg_get(cfg, "test.save_rollout_vtk_steps", None)
    if configured_vtk_steps is None:
        vtk_steps = None  # Legacy behavior: save every prediction VTK.
    else:
        vtk_steps = {int(step) for step in configured_vtk_steps}
        if not {0, max_steps}.issubset(vtk_steps) or any(
            step < 0 or step > max_steps for step in vtk_steps
        ):
            raise ValueError("Sparse VTK steps must include 0 and final rollout step")
    if vtk_steps is None or 0 in vtk_steps:
        save_prediction_like_input(0, output_dir, template_mesh, current_np, use_cell_data)
    save_trajectory_npz = bool(_cfg_get(cfg, "test.save_rollout_npz", False))
    trajectory = None
    if save_trajectory_npz:
        trajectory = np.empty((max_steps + 1, len(current_np), 5), dtype=np.float32)
        trajectory[0] = current_np
    channel_names = ["vx", "vy", "vz", "pressure", "temperature"]
    rmse_sum = np.zeros(5, dtype=np.float64)
    rel_sum = np.zeros(5, dtype=np.float64)
    vel_vec_rmse_sum = 0.0
    vel_vec_rel_sum = 0.0
    rows = []

    for step in range(1, max_steps + 1):
        step_wall_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_start = time.perf_counter()
        pred_norm = model(
            q, normalizer.encode(current), edge_index, edge_attr, gs, sdf,
            boundary_features, score_edge_permutation,
        )
        pred_phys = normalizer.decode(pred_norm)
        # Synchronize before loading targets or imposing boundary values so the
        # reported model-forward time excludes ground-truth access, forcing,
        # metric computation, and VTK I/O.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - forward_start
        model_forward_time += forward_time
        gt_np = test_dataset._load_state(test_case_dir, step)
        if bool(_cfg_get(cfg, "test.semi_teacher_force_boundary", False)):
            gt_phys = torch.from_numpy(gt_np).float().to(dist.device)
            pred_phys = impose_boundary_from_target(
                pred_phys, gt_phys, boundary_features, int(_cfg_get(cfg, "training.interior_class_index", 1))
            )
        pred_np = pred_phys.detach().cpu().numpy()
        if vtk_steps is None or step in vtk_steps:
            save_prediction_like_input(step, output_dir, template_mesh, pred_np, use_cell_data)
        if trajectory is not None:
            trajectory[step] = pred_np

        if metric_mask_np is None:
            pred_metric, gt_metric = pred_np, gt_np
        else:
            pred_metric, gt_metric = pred_np[metric_mask_np], gt_np[metric_mask_np]
        rmse = compute_rmse_np(pred_metric, gt_metric)
        rel = compute_relative_l2_np(pred_metric, gt_metric)
        vel_vec_rmse = compute_velocity_vector_rmse_np(pred_metric, gt_metric)
        vel_vec_rel = compute_velocity_vector_relative_l2_np(pred_metric, gt_metric)
        rmse_sum += rmse
        rel_sum += rel
        vel_vec_rmse_sum += vel_vec_rmse
        vel_vec_rel_sum += vel_vec_rel
        step_total_time = time.perf_counter() - step_wall_start
        row = {
            "step": step,
            "metric_scope": metric_scope,
            "metric_node_count": metric_node_count,
            "vel_vec_rmse": vel_vec_rmse,
            "vel_vec_rel_l2": vel_vec_rel,
            "model_forward_time_s": float(forward_time),
            "step_total_time_s": float(step_total_time),
        }
        for i, name in enumerate(channel_names):
            row[f"{name}_rmse"] = float(rmse[i])
            row[f"{name}_rel_l2"] = float(rel[i])
        rows.append(row)
        print(
            f"[test rollout {case_name}] step {step}/{max_steps} | "
            f"vel_vec RelL2={vel_vec_rel:.6e} | T RelL2={rel[4]:.6e} | "
            f"forward={forward_time:.4f}s | total_step={step_total_time:.4f}s"
        )
        current = pred_phys

    if trajectory is not None:
        trajectory_path = os.path.join(output_dir, "trajectory.npz")
        temporary_path = trajectory_path + f".{os.getpid()}.tmp"
        try:
            with open(temporary_path, "xb") as stream:
                np.savez_compressed(
                    stream,
                    predictions=trajectory,
                    steps=np.arange(max_steps + 1, dtype=np.int32),
                )
            os.replace(temporary_path, trajectory_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    n = max(len(rows), 1)
    avg_rmse = rmse_sum / n
    avg_rel = rel_sum / n
    avg_vel_vec_rmse = vel_vec_rmse_sum / n
    avg_vel_vec_rel = vel_vec_rel_sum / n

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rollout_wall_time = time.perf_counter() - rollout_wall_start
    avg_forward_time = model_forward_time / n
    avg_wall_step_time = rollout_wall_time / n

    summary_path = os.path.join(output_dir, "rollout_metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Test case: {case_name}\n")
        f.write(f"Rollout steps: {max_steps}\n")
        f.write(f"Metric scope: {metric_scope}\n")
        f.write(f"Metric nodes: {metric_node_count}\n")
        f.write(
            "Boundary conditioned: "
            f"{bool(_cfg_get(cfg, 'test.semi_teacher_force_boundary', False))}\n"
        )
        f.write("=== Timing ===\n")
        f.write(f"Full rollout wall time (seconds): {rollout_wall_time:.6f}\n")
        f.write(f"Full rollout wall time (minutes): {rollout_wall_time / 60.0:.6f}\n")
        f.write(f"Pure model forward time total (seconds): {model_forward_time:.6f}\n")
        f.write(f"Average model forward time per step (seconds): {avg_forward_time:.6f}\n")
        f.write(f"Average full wall time per step incl. IO/metrics (seconds): {avg_wall_step_time:.6f}\n")
        f.write("=== Average rollout metrics across timesteps ===\n")
        f.write(f"{'vel_vec':>12s} | Avg RMSE = {avg_vel_vec_rmse:.6e} | Avg RelL2 = {avg_vel_vec_rel:.6e}\n")
        for i, name in enumerate(channel_names):
            f.write(f"{name:>12s} | Avg RMSE = {avg_rmse[i]:.6e} | Avg RelL2 = {avg_rel[i]:.6e}\n")

    csv_path = os.path.join(output_dir, "rollout_metrics_per_step.csv")
    fieldnames = [
        "step", "metric_scope", "metric_node_count", "vel_vec_rmse",
        "vel_vec_rel_l2", "model_forward_time_s", "step_total_time_s",
    ]
    for name in channel_names:
        fieldnames.extend([f"{name}_rmse", f"{name}_rel_l2"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    timing_path = os.path.join(output_dir, "timing.txt")
    with open(timing_path, "w") as f:
        f.write(f"Rollout steps: {max_steps}\n")
        f.write(f"Full rollout wall time (seconds): {rollout_wall_time:.6f}\n")
        f.write(f"Full rollout wall time (minutes): {rollout_wall_time / 60.0:.6f}\n")
        f.write(f"Pure model forward time total (seconds): {model_forward_time:.6f}\n")
        f.write(f"Average model forward time per step (seconds): {avg_forward_time:.6f}\n")
        f.write(f"Average full wall time per step incl. IO/metrics (seconds): {avg_wall_step_time:.6f}\n")

    print("\n=== Test rollout timing ===")
    print(f"Full rollout wall time: {rollout_wall_time:.2f} seconds ({rollout_wall_time / 60.0:.2f} minutes)")
    print(f"Pure model forward time: {model_forward_time:.2f} seconds total, {avg_forward_time:.4f} seconds/step")
    print(f"Average full step time incl. IO/metrics: {avg_wall_step_time:.4f} seconds/step")

    print("\n=== Test rollout summary ===")
    print(f"Test case: {case_name}")
    print(f"Saved rollout VTKs to: {output_dir}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved per-step metrics: {csv_path}")
    print(f"Saved timing: {timing_path}")
    print(f"vel_vec Avg RMSE={avg_vel_vec_rmse:.6e}, Avg RelL2={avg_vel_vec_rel:.6e}")
    for i, name in enumerate(channel_names):
        print(f"{name:>12s} | Avg RMSE={avg_rmse[i]:.6e} | Avg RelL2={avg_rel[i]:.6e}")
    return {
        "test_case": case_name,
        "rollout_steps": max_steps,
        "metric_scope": metric_scope,
        "metric_node_count": metric_node_count,
        "rollout_metrics_path": csv_path,
        "rollout_summary_path": summary_path,
    }


# ============================================================
# MAIN
# ============================================================
@hydra.main(version_base="1.3", config_path=".", config_name="config_simple_gnn")
def main(cfg: DictConfig) -> None:
    evaluation_only = bool(_cfg_get(cfg, "training.evaluation_only", False))
    save_dir = make_experiment_dir(cfg)
    score_control_spec = score_control_run_spec(cfg)
    cfg.training.checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    if score_control_spec is not None and os.listdir(cfg.training.checkpoint_dir):
        raise RuntimeError(
            "C70S100 score-control run has checkpoint artifacts before epoch 0; resume is refused"
        )

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(os.path.join(save_dir, "training.log"), "a", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"\n=== Experiment save dir ===\n{save_dir}\n")
    if not evaluation_only:
        with open(os.path.join(save_dir, "resolved_config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

    DistributedManager.initialize()
    dist = DistributedManager()
    log = PythonLogger(name="simple_gnn_fno_boundary")
    log.file_logging()
    LaunchLogger.initialize()

    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    data_root_abs = to_absolute_path(str(cfg.data.data_root))
    all_cases = list_case_dirs(data_root_abs)
    if int(cfg.data.max_cases) > 0:
        all_cases = all_cases[:int(cfg.data.max_cases)]

    train_count = int(_cfg_get(cfg, "split.train_cases", 10))
    valid_count = int(_cfg_get(cfg, "split.valid_cases", 2))
    test_count = int(_cfg_get(cfg, "split.test_cases", 1))
    explicit_case_split_path = str(_cfg_get(cfg, "split.explicit_manifest", "") or "")
    if explicit_case_split_path:
        if not os.path.isabs(explicit_case_split_path):
            local_manifest = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), explicit_case_split_path
            )
            explicit_case_split_path = (
                local_manifest if os.path.isfile(local_manifest)
                else to_absolute_path(explicit_case_split_path)
            )
        train_case_dirs, valid_case_dirs, test_case_dirs, split_metadata = load_explicit_case_split(
            all_cases,
            explicit_case_split_path,
            {
                "train_cases": train_count,
                "valid_cases": valid_count,
                "test_cases": test_count,
            },
        )
    else:
        train_case_dirs, valid_case_dirs, test_case_dirs = split_case_dirs_three_way(
            all_cases, int(cfg.training.seed), train_count, valid_count, test_count
        )
        split_metadata = {"split_mode": "seeded-random"}
    if evaluation_only:
        with open(os.path.join(save_dir, "case_split.json"), encoding="utf-8") as f:
            split_info = json.load(f)
        if (
            split_info.get("train_cases") != [os.path.basename(c) for c in train_case_dirs]
            or split_info.get("valid_cases") != [os.path.basename(c) for c in valid_case_dirs]
            or split_info.get("test_cases") != [os.path.basename(c) for c in test_case_dirs]
        ):
            raise ValueError("Evaluation-only saved case split differs from frozen manifest")
    else:
        split_info = write_split_log(
            os.path.join(save_dir, "case_split.json"),
            train_case_dirs, valid_case_dirs, test_case_dirs, cfg, split_metadata,
        )
    print("=== Case split ===")
    print(json.dumps(split_info, indent=2))

    common_dataset_kwargs = dict(
        data_root=data_root_abs,
        val_ratio=float(getattr(cfg.data, "val_ratio", 0.0)),
        seed=int(cfg.training.seed),
        grid_resolution=int(cfg.data.grid_resolution),
        k_neighbors=int(cfg.data.k_neighbors),
        velocity_keys=list(cfg.data.velocity_keys),
        pressure_keys=list(cfg.data.pressure_keys),
        temperature_keys=list(cfg.data.temperature_keys),
        max_cases=int(cfg.data.max_cases),
        use_boundary_features=bool(getattr(cfg.data, "use_boundary_features", True)),
        boundary_onehot_key=str(getattr(cfg.data, "boundary_onehot_key", "boundary_onehot")),
        boundary_class_key=str(getattr(cfg.data, "boundary_class_key", "boundary_class")),
        boundary_feature_dim=int(getattr(cfg.arch, "boundary_feature_dim", 6)),
        outlet_geometry_features=bool(_cfg_get(cfg, "data.outlet_geometry_features", False)),
        acu_outlet_class_index=int(_cfg_get(cfg, "data.acu_outlet_class_index", 3)),
        compute_grid_sdf=bool(_cfg_get(cfg, "data.compute_grid_sdf", True)),
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        explicit_case_split_path=explicit_case_split_path,
        score_edge_mode=str(_cfg_get(cfg, "arch.gnn.score_edge_mode", "aligned")),
    )
    train_dataset = VTKMGNHybridDataset(split="train", train_query_subsample=int(cfg.data.train_query_subsample), rollout_steps=int(cfg.training.rollout_steps), **common_dataset_kwargs)
    valid_dataset = VTKMGNHybridDataset(split="valid", train_query_subsample=-1, rollout_steps=int(cfg.validation.rollout_steps), **common_dataset_kwargs)
    test_dataset = VTKMGNHybridDataset(split="test", train_query_subsample=-1, rollout_steps=int(_cfg_get(cfg, "test.rollout_steps", 200)), **common_dataset_kwargs)

    training_window_selection = None
    if score_control_spec is not None:
        window_manifest_path = str(_cfg_get(cfg, "data.training_window_manifest", ""))
        if not os.path.isabs(window_manifest_path):
            window_manifest_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), window_manifest_path
            )
        training_window_selection = load_c70_training_window_selection(
            window_manifest_path,
            split_info["train_cases"],
            str(split_info["split_semantic_sha256"]),
            int(cfg.training.seed),
        )
        train_dataset.samples = resolve_c70_training_window_samples(
            training_window_selection["records"], train_dataset.samples
        )

    # Optional deterministic caps are useful for controlled architecture
    # pilots without changing case membership or sample ordering.
    max_train_samples = int(_cfg_get(cfg, "data.max_train_samples", -1))
    max_valid_samples = int(_cfg_get(cfg, "data.max_valid_samples", -1))
    if max_train_samples > 0:
        if score_control_spec is not None:
            if len(train_dataset.samples) != max_train_samples:
                raise ValueError(
                    "C70S100 frozen training-window selection does not match the locked sample budget"
                )
        else:
            train_dataset.samples = train_dataset.samples[:max_train_samples]
    if max_valid_samples > 0:
        valid_dataset.samples = valid_dataset.samples[:max_valid_samples]
    write_score_control_run_manifest(
        save_dir, cfg, split_info, training_window_selection
    )

    loader_kwargs = dict(
        num_workers=int(cfg.training.num_workers),
        pin_memory=True,
        persistent_workers=True if int(cfg.training.num_workers) > 0 else False,
        collate_fn=collate_list,
    )
    if bool(_cfg_get(cfg, "training.deterministic_data_order", False)):
        train_generator = torch.Generator()
        train_generator.manual_seed(int(cfg.training.seed))
        valid_generator = torch.Generator()
        valid_generator.manual_seed(int(cfg.training.seed) + 1)
        loader_kwargs["worker_init_fn"] = seed_dataloader_worker
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=True,
            generator=train_generator,
            **loader_kwargs,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(getattr(cfg.validation, "batch_size", cfg.training.batch_size)),
            shuffle=False,
            generator=valid_generator,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=True,
            **loader_kwargs,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(getattr(cfg.validation, "batch_size", cfg.training.batch_size)),
            shuffle=False,
            **loader_kwargs,
        )

    model = GNNFNOGNNHybrid(
        state_dim=int(cfg.arch.state_dim),
        boundary_feature_dim=int(getattr(cfg.arch, "boundary_feature_dim", 6)),
        outlet_geometry_feature_dim=int(_cfg_get(cfg, "arch.outlet_geometry_feature_dim", 0)),
        latent_dim=int(cfg.arch.latent_dim),
        fno_modes=int(cfg.arch.fno.fno_modes),
        fno_layers=int(cfg.arch.fno.fno_layers),
        fno_padding=int(cfg.arch.fno.padding),
        fno_decoder_layers=int(cfg.arch.fno.decoder_layers),
        fno_decoder_layer_size=int(cfg.arch.fno.decoder_layer_size),
        encoder_hidden_dim=int(cfg.arch.gnn.encoder.hidden_dim),
        encoder_processor_layers=int(cfg.arch.gnn.encoder.processor_layers),
        encoder_mlp_layers=int(cfg.arch.gnn.encoder.mlp_layers),
        decoder_hidden_dim=int(cfg.arch.gnn.decoder.hidden_dim),
        decoder_processor_layers=int(cfg.arch.gnn.decoder.processor_layers),
        decoder_mlp_layers=int(cfg.arch.gnn.decoder.mlp_layers),
        gnn_activation=str(cfg.arch.gnn.activation),
        gnn_layer_norm=bool(cfg.arch.gnn.layer_norm),
        gnn_residual=bool(cfg.arch.gnn.residual),
        gnn_aggregation=str(cfg.arch.gnn.aggregation),
        gnn_use_edge_features=bool(getattr(cfg.arch.gnn, "use_edge_features", True)),
        gnn_use_attention=bool(getattr(cfg.arch.gnn, "use_attention", False)),
        gnn_attention_mode=_cfg_get(cfg, "arch.gnn.attention_mode", None),
        gnn_attention_hidden_dim=int(getattr(cfg.arch.gnn, "attention_hidden_dim", 32)),
        gnn_attention_temperature=float(getattr(cfg.arch.gnn, "attention_temperature", 1.0)),
        gnn_score_edge_mode=str(_cfg_get(cfg, "arch.gnn.score_edge_mode", "aligned")),
    ).to(dist.device)

    optimizer = AdamW(model.parameters(), lr=float(cfg.scheduler.initial_lr), weight_decay=float(cfg.scheduler.weight_decay))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(cfg.scheduler.decay_epochs), gamma=float(cfg.scheduler.decay_rate))

    if score_control_spec is not None:
        # The fresh directory reservation above is the only permitted entry
        # point.  Never call the generic checkpoint loader for a score-control
        # cell, even if an unexpected artifact appears after reservation.
        loaded_epoch, best_val, ckpt_mean, ckpt_std = 0, None, None, None
    else:
        loaded_epoch, best_val, ckpt_mean, ckpt_std = load_plain_checkpoint(
            cfg.training.checkpoint_dir, dist.device, model, optimizer, scheduler
        )
    if best_val is None:
        best_val = float("inf")

    if ckpt_mean is not None and ckpt_std is not None:
        mean = torch.as_tensor(ckpt_mean).float()
        std = torch.as_tensor(ckpt_std).float()
        log.warning("Loaded normalization stats from checkpoint.")
    else:
        log.info("Computing normalization statistics from training set...")
        mean, std = compute_normalization_stats(train_dataset, max_samples=int(getattr(cfg.training, "norm_stat_samples", 50)))
        log.info(f"Normalization mean: {mean.tolist()}")
        log.info(f"Normalization std: {std.tolist()}")

    normalizer = ChannelNormalizer(mean, std).to(dist.device)
    use_amp = bool(cfg.training.use_amp) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if loaded_epoch == 0:
        log.info("Training started...")
    else:
        log.warning(f"Resuming from epoch {loaded_epoch + 1}")

    epoch_csv = os.path.join(save_dir, "epoch_metrics.csv")
    if not os.path.exists(epoch_csv):
        with open(epoch_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["epoch", "train_rel", "val_rel", "val_mse_norm", "lr", "best_val"]).writeheader()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_wall_start = time.perf_counter()

    for epoch in range(max(1, loaded_epoch + 1), int(cfg.training.max_epochs) + 1):
        with LaunchLogger(name_space="train", epoch=epoch, num_mini_batch=len(train_loader), epoch_alert_freq=1) as train_logger:
            train_rel = train_one_epoch(
                model, train_loader, optimizer, scaler, dist, use_amp, normalizer, train_logger,
                int(cfg.training.rollout_steps),
                bool(getattr(cfg.training, "teacher_forcing", False)),
                bool(getattr(cfg.training, "use_multistep_loss", True)),
                bool(_cfg_get(cfg, "training.train_interior_only", False)),
                int(_cfg_get(cfg, "training.interior_class_index", 1)),
                bool(_cfg_get(cfg, "training.semi_teacher_force_boundary", False)),
                bool(_cfg_get(cfg, "loss.use_pressure_grad_loss", False)),
                float(_cfg_get(cfg, "loss.pressure_grad_weight", 0.05)),
                bool(_cfg_get(cfg, "loss.pressure_grad_relative", True)),
                bool(_cfg_get(cfg, "loss.use_velocity_grad_loss", False)),
                float(_cfg_get(cfg, "loss.velocity_grad_weight", 0.01)),
                bool(_cfg_get(cfg, "loss.velocity_grad_relative", True)),
                bool(_cfg_get(cfg, "loss.use_temperature_grad_loss", False)),
                float(_cfg_get(cfg, "loss.temperature_grad_weight", 0.01)),
                bool(_cfg_get(cfg, "loss.temperature_grad_relative", True)),
            )
            train_logger.log_epoch({"Train RelL2": train_rel, "Learning Rate": optimizer.param_groups[0]["lr"]})

        val_rel = None
        val_mse_norm = None
        if epoch % int(cfg.validation.validation_epochs) == 0:
            with LaunchLogger("valid", epoch=epoch) as valid_logger:
                val_rel, val_mse_norm, val_rmse_phys = validate_one_epoch(model, valid_loader, dist, use_amp, normalizer)
                valid_logger.log_epoch({
                    "Validation RelL2": val_rel,
                    "Validation MSE (norm)": val_mse_norm,
                    "Validation RMSE vx": float(val_rmse_phys[0]),
                    "Validation RMSE vy": float(val_rmse_phys[1]),
                    "Validation RMSE vz": float(val_rmse_phys[2]),
                    "Validation RMSE p": float(val_rmse_phys[3]),
                    "Validation RMSE T": float(val_rmse_phys[4]),
                })
                if val_rel < best_val:
                    best_val = val_rel
                    save_plain_checkpoint(cfg.training.checkpoint_dir, epoch, model, optimizer, scheduler, best_val=best_val, norm_mean=normalizer.mean.detach().cpu(), norm_std=normalizer.std.detach().cpu())
                    print(f"Saved new best checkpoint at epoch {epoch}, val_rel={val_rel:.6e}")

        if epoch % int(cfg.training.rec_results_freq) == 0:
            save_plain_checkpoint(cfg.training.checkpoint_dir, epoch, model, optimizer, scheduler, best_val=best_val, norm_mean=normalizer.mean.detach().cpu(), norm_std=normalizer.std.detach().cpu())

        with open(epoch_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=["epoch", "train_rel", "val_rel", "val_mse_norm", "lr", "best_val"]).writerow({"epoch": epoch, "train_rel": train_rel, "val_rel": val_rel, "val_mse_norm": val_mse_norm, "lr": optimizer.param_groups[0]["lr"], "best_val": best_val})
        scheduler.step()

    save_plain_checkpoint(cfg.training.checkpoint_dir, int(cfg.training.max_epochs), model, optimizer, scheduler, best_val=best_val, norm_mean=normalizer.mean.detach().cpu(), norm_std=normalizer.std.detach().cpu())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_training_time = time.perf_counter() - training_wall_start

    log.info("Training completed.")
    print("\n=== Training timing ===")
    print(f"Total training wall time: {total_training_time:.2f} seconds ({total_training_time / 3600.0:.2f} hours)")

    training_timing_path = os.path.join(save_dir, "training_timing.txt")
    if not evaluation_only:
        with open(training_timing_path, "w") as f:
            f.write(f"Start epoch: {max(1, loaded_epoch + 1)}\n")
            f.write(f"End epoch: {int(cfg.training.max_epochs)}\n")
            f.write(f"Total training wall time (seconds): {total_training_time:.6f}\n")
            f.write(f"Total training wall time (hours): {total_training_time / 3600.0:.6f}\n")
        print(f"Saved training timing: {training_timing_path}")
    else:
        print("Evaluation-only run: original training timing was not modified")

    rollout_result = None
    if bool(_cfg_get(cfg, "test.rollout_after_training", True)):
        print("\n=== Running final rollout on held-out test case ===")
        latest_path = os.path.join(cfg.training.checkpoint_dir, "latest.pt")
        if not os.path.isfile(latest_path):
            raise RuntimeError("Final rollout checkpoint is missing")
        if bool(_cfg_get(cfg, "test.use_in_memory_final_weights", False)):
            # The just-completed epoch and final checkpoint contain the same
            # model and normalization tensors. Avoid a second full model plus
            # Adam state allocation on GPU for capacity-sized follow-ups.
            print("Using in-memory final-epoch model for rollout")
        else:
            ckpt = torch.load(latest_path, map_location=dist.device)
            model.load_state_dict(ckpt["model_state_dict"])
            if ckpt.get("norm_mean") is not None and ckpt.get("norm_std") is not None:
                normalizer = ChannelNormalizer(torch.as_tensor(ckpt["norm_mean"]).float(), torch.as_tensor(ckpt["norm_std"]).float()).to(dist.device)
        rollout_result = rollout_test_case(
            model, normalizer, cfg, test_dataset, test_case_dirs[0], save_dir, dist
        )

    if score_control_spec is not None:
        condition, seed = score_control_spec
        if loaded_epoch != 0:
            raise RuntimeError("C70S100 score-control completion did not start at epoch 0")
        if rollout_result is None:
            raise RuntimeError("C70S100 score-control completion requires the final rollout")
        if int(rollout_result["rollout_steps"]) != 200:
            raise RuntimeError(
                "C70S100 score-control completion requires exactly 200 rollout transitions"
            )
        latest_path = os.path.join(cfg.training.checkpoint_dir, "latest.pt")
        completion_path = write_completion_manifest(
            run_dir=Path(save_dir),
            condition=condition,
            seed=seed,
            start_epoch=loaded_epoch,
            end_epoch=int(cfg.training.max_epochs),
            checkpoint_path=Path(latest_path),
            epoch_metrics_path=Path(epoch_csv),
            rollout_metrics_path=Path(rollout_result["rollout_metrics_path"]),
            rollout_summary_path=Path(rollout_result["rollout_summary_path"]),
            test_case=str(rollout_result["test_case"]),
            runtime=runtime_identity(torch, sys.modules.get("physicsnemo")),
            input_run_manifest_path=Path(
                os.path.join(save_dir, "score_control_run_manifest.json")
            ),
        )
        print("Saved atomic score-control completion manifest: {}".format(completion_path))

    # Restore the process streams before closing their Tee target.  Otherwise
    # Python's shutdown flush writes into a closed file and turns a successful
    # container run into exit code 120.
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


if __name__ == "__main__":
    main()
