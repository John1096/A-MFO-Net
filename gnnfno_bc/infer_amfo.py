"""Run the boundary-conditioned A-MFO-Net held-out rollout from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

import train_gnn_fno_semibc as experiment
from train_amfo import HELD_OUT_CASE, SPLIT_PATH, check_split


def build_model(cfg):
    return experiment.GNNFNOGNNHybrid(
        state_dim=int(cfg.arch.state_dim),
        boundary_feature_dim=int(cfg.arch.boundary_feature_dim),
        outlet_geometry_feature_dim=int(getattr(cfg.arch, "outlet_geometry_feature_dim", 0)),
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
        gnn_use_edge_features=bool(cfg.arch.gnn.use_edge_features),
        gnn_use_attention=True,
        gnn_attention_mode="learned",
        gnn_attention_hidden_dim=int(cfg.arch.gnn.attention_hidden_dim),
        gnn_attention_temperature=float(cfg.arch.gnn.attention_temperature),
        gnn_score_edge_mode="aligned",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    check_split()
    run_dir = args.run_dir.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_case = output_root / "test_rollout" / HELD_OUT_CASE
    if output_case.exists() or output_case.is_symlink():
        raise FileExistsError(f"Refusing to overwrite a rollout: {output_case}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"CFD data root is unavailable: {data_root}")

    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    if not bool(cfg.arch.gnn.use_attention) or str(getattr(cfg.arch.gnn, "attention_mode", "learned")) != "learned":
        raise ValueError("Saved run is not the learned-attention A-MFO-Net")
    if str(getattr(cfg.arch.gnn, "score_edge_mode", "aligned")) != "aligned":
        raise ValueError("Saved run is not the aligned-score A-MFO-Net")
    cfg.data.data_root = str(data_root)
    cfg.split.explicit_manifest = SPLIT_PATH.name
    cfg.test.rollout_steps = 400
    cfg.test.metric_scope = "all"
    cfg.test.semi_teacher_force_boundary = True
    cfg.test.save_rollout_vtk_steps = [0, 20, 200, 400]
    cfg.test.save_rollout_npz = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        run_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=True
    )
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    normalizer = experiment.ChannelNormalizer(
        torch.as_tensor(checkpoint["norm_mean"]).float(),
        torch.as_tensor(checkpoint["norm_std"]).float(),
    )
    del checkpoint
    model = model.to(device)
    normalizer = normalizer.to(device)

    dataset = experiment.VTKMGNHybridDataset(
        data_root=str(data_root),
        split="test",
        val_ratio=float(cfg.data.val_ratio),
        seed=int(cfg.training.seed),
        grid_resolution=int(cfg.data.grid_resolution),
        k_neighbors=int(cfg.data.k_neighbors),
        velocity_keys=list(cfg.data.velocity_keys),
        pressure_keys=list(cfg.data.pressure_keys),
        temperature_keys=list(cfg.data.temperature_keys),
        max_cases=int(cfg.data.max_cases),
        train_query_subsample=-1,
        rollout_steps=400,
        use_boundary_features=bool(cfg.data.use_boundary_features),
        boundary_onehot_key=str(cfg.data.boundary_onehot_key),
        boundary_class_key=str(cfg.data.boundary_class_key),
        boundary_feature_dim=int(cfg.arch.boundary_feature_dim),
        outlet_geometry_features=bool(getattr(cfg.data, "outlet_geometry_features", False)),
        acu_outlet_class_index=int(getattr(cfg.data, "acu_outlet_class_index", 3)),
        compute_grid_sdf=bool(cfg.data.compute_grid_sdf),
        train_count=int(cfg.split.train_cases),
        valid_count=int(cfg.split.valid_cases),
        test_count=int(cfg.split.test_cases),
        explicit_case_split_path=str(SPLIT_PATH),
        score_edge_mode="aligned",
    )
    if [Path(case).name for case in dataset.case_dirs] != [HELD_OUT_CASE]:
        raise ValueError("Held-out case differs from the frozen split")

    experiment.rollout_test_case(
        model, normalizer, cfg, dataset, dataset.case_dirs[0],
        str(output_root), SimpleNamespace(device=device),
    )


if __name__ == "__main__":
    main()
