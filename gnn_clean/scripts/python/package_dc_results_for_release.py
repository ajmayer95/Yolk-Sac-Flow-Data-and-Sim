#!/usr/bin/env python
"""Stage repo-ready and release-ready DC results bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "dc"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "publish"


@dataclass(frozen=True)
class CopySpec:
    source: Path
    destination_relative: Path
    description: str


@dataclass(frozen=True)
class ArchiveSpec:
    name: str
    description: str
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-step0-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-step1-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-step2-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-step3-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-step4-raw", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {description}: {path}")
    return path


def resolve_step_output_dir(outputs_root: Path, step_name: str) -> Path | None:
    exact = outputs_root / step_name
    if exact.exists():
        return exact
    matches = sorted(
        path for path in outputs_root.glob(f"{step_name}*") if path.is_dir()
    )
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Multiple candidate directories found for {step_name} under {outputs_root}: "
        + ", ".join(str(path.name) for path in matches)
    )


def maybe_copy_spec(path: Path, destination_relative: str, description: str) -> CopySpec | None:
    return CopySpec(path, Path(destination_relative), description) if path.exists() else None


def file_or_tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_repo_copy_specs(outputs_root: Path) -> list[CopySpec]:
    step0_root = resolve_step_output_dir(outputs_root, "00_ideal_models")
    step1_root = resolve_step_output_dir(outputs_root, "01_boundary_parameter_calibration")
    step2_root = resolve_step_output_dir(outputs_root, "02_physics_weight_sweep")
    step3_root = resolve_step_output_dir(outputs_root, "03_pressure_constraint_sensitivity")
    step4_root = resolve_step_output_dir(outputs_root, "04_message_passing_sensitivity")
    specs: list[CopySpec | None] = [
        maybe_copy_spec(
            (step0_root / "poiseuille_only_baseline" / "default_partitioned" / "summary.csv") if step0_root else Path("__missing__"),
            "dc/00_ideal_models/poiseuille_only_baseline/default_partitioned/summary.csv",
            "DC Step 0 Poiseuille baseline summary table",
        ),
        maybe_copy_spec(
            (step0_root / "poiseuille_only_baseline" / "default_partitioned" / "summary.yaml") if step0_root else Path("__missing__"),
            "dc/00_ideal_models/poiseuille_only_baseline/default_partitioned/summary.yaml",
            "DC Step 0 Poiseuille baseline summary YAML",
        ),
        maybe_copy_spec(
            (step0_root / "poiseuille_only_baseline" / "default_partitioned" / "poiseuille_summary.csv") if step0_root else Path("__missing__"),
            "dc/00_ideal_models/poiseuille_only_baseline/default_partitioned/poiseuille_summary.csv",
            "DC Step 0 Poiseuille-only baseline summary",
        ),
        maybe_copy_spec(
            (step0_root / "poiseuille_only_baseline" / "default_partitioned" / "figures") if step0_root else Path("__missing__"),
            "dc/00_ideal_models/poiseuille_only_baseline/default_partitioned/figures",
            "DC Step 0 Poiseuille baseline figures",
        ),
        maybe_copy_spec(
            (step1_root / "boundary_weight_summary.csv") if step1_root else Path("__missing__"),
            "dc/01_boundary_parameter_calibration/boundary_weight_summary.csv",
            "DC Step 1 boundary-weight summary table",
        ),
        maybe_copy_spec(
            (step1_root / "boundary_weight_summary.yaml") if step1_root else Path("__missing__"),
            "dc/01_boundary_parameter_calibration/boundary_weight_summary.yaml",
            "DC Step 1 boundary-weight summary YAML",
        ),
        maybe_copy_spec(
            (step1_root / "figures") if step1_root else Path("__missing__"),
            "dc/01_boundary_parameter_calibration/figures",
            "DC Step 1 boundary-parameter calibration figures",
        ),
        maybe_copy_spec(
            (step2_root / "launcher_manifest.csv") if step2_root else Path("__missing__"),
            "dc/02_physics_weight_sweep/launcher_manifest.csv",
            "DC Step 2 launcher manifest",
        ),
        maybe_copy_spec(
            (step2_root / "physics_weight_all_runs.csv") if step2_root else Path("__missing__"),
            "dc/02_physics_weight_sweep/physics_weight_all_runs.csv",
            "DC Step 2 all-runs summary",
        ),
        maybe_copy_spec(
            (step2_root / "physics_weight_analysis.yaml") if step2_root else Path("__missing__"),
            "dc/02_physics_weight_sweep/physics_weight_analysis.yaml",
            "DC Step 2 analysis YAML",
        ),
        maybe_copy_spec(
            (step2_root / "physics_weight_gnn_summary.csv") if step2_root else Path("__missing__"),
            "dc/02_physics_weight_sweep/physics_weight_gnn_summary.csv",
            "DC Step 2 GNN summary",
        ),
        maybe_copy_spec(
            (step2_root / "physics_weight_poiseuille_summary.csv") if step2_root else Path("__missing__"),
            "dc/02_physics_weight_sweep/physics_weight_poiseuille_summary.csv",
            "DC Step 2 Poiseuille summary",
        ),
        maybe_copy_spec(
            (step2_root / "representative_configurations.csv") if step2_root else Path("__missing__"),
            "dc/02_physics_weight_sweep/representative_configurations.csv",
            "DC Step 2 representative configurations",
        ),
        maybe_copy_spec(
            (step2_root / "figures") if step2_root else Path("__missing__"),
            "dc/02_physics_weight_sweep/figures",
            "DC Step 2 figures",
        ),
        maybe_copy_spec(
            (step3_root / "pressure_constraint_all_runs.csv") if step3_root else Path("__missing__"),
            "dc/03_pressure_constraint_sensitivity/pressure_constraint_all_runs.csv",
            "DC Step 3 all-runs summary",
        ),
        maybe_copy_spec(
            (step3_root / "pressure_constraint_gnn_summary.csv") if step3_root else Path("__missing__"),
            "dc/03_pressure_constraint_sensitivity/pressure_constraint_gnn_summary.csv",
            "DC Step 3 GNN summary",
        ),
        maybe_copy_spec(
            (step3_root / "pressure_constraint_poiseuille_summary.csv") if step3_root else Path("__missing__"),
            "dc/03_pressure_constraint_sensitivity/pressure_constraint_poiseuille_summary.csv",
            "DC Step 3 Poiseuille summary",
        ),
        maybe_copy_spec(
            (step3_root / "pressure_field_pairwise_metrics.csv") if step3_root else Path("__missing__"),
            "dc/03_pressure_constraint_sensitivity/pressure_field_pairwise_metrics.csv",
            "DC Step 3 pressure field pairwise metrics",
        ),
        maybe_copy_spec(
            (step3_root / "correction_field_pairwise_metrics.csv") if step3_root else Path("__missing__"),
            "dc/03_pressure_constraint_sensitivity/correction_field_pairwise_metrics.csv",
            "DC Step 3 correction field pairwise metrics",
        ),
        maybe_copy_spec(
            (step3_root / "pressure_correlation_matrix.csv") if step3_root else Path("__missing__"),
            "dc/03_pressure_constraint_sensitivity/pressure_correlation_matrix.csv",
            "DC Step 3 pressure correlation matrix table",
        ),
        maybe_copy_spec(
            (step3_root / "correction_correlation_matrix.csv") if step3_root else Path("__missing__"),
            "dc/03_pressure_constraint_sensitivity/correction_correlation_matrix.csv",
            "DC Step 3 correction correlation matrix table",
        ),
        maybe_copy_spec(
            (step3_root / "figures") if step3_root else Path("__missing__"),
            "dc/03_pressure_constraint_sensitivity/figures",
            "DC Step 3 figures",
        ),
        maybe_copy_spec(
            (step4_root / "summary.csv") if step4_root else Path("__missing__"),
            "dc/04_message_passing_sensitivity/summary.csv",
            "DC Step 4 summary table",
        ),
        maybe_copy_spec(
            (step4_root / "summary.yaml") if step4_root else Path("__missing__"),
            "dc/04_message_passing_sensitivity/summary.yaml",
            "DC Step 4 summary YAML",
        ),
        maybe_copy_spec(
            (step4_root / "flow_rmse_vs_K.png") if step4_root else Path("__missing__"),
            "dc/04_message_passing_sensitivity/flow_rmse_vs_K.png",
            "DC Step 4 flow RMSE plot",
        ),
        maybe_copy_spec(
            (step4_root / "kirchhoff_rms_vs_K.png") if step4_root else Path("__missing__"),
            "dc/04_message_passing_sensitivity/kirchhoff_rms_vs_K.png",
            "DC Step 4 Kirchhoff RMS plot",
        ),
        maybe_copy_spec(
            (step4_root / "pressure_maps_by_K.png") if step4_root else Path("__missing__"),
            "dc/04_message_passing_sensitivity/pressure_maps_by_K.png",
            "DC Step 4 pressure maps by K",
        ),
        maybe_copy_spec(
            (step4_root / "flow_residual_maps_by_K.png") if step4_root else Path("__missing__"),
            "dc/04_message_passing_sensitivity/flow_residual_maps_by_K.png",
            "DC Step 4 flow residual maps by K",
        ),
        maybe_copy_spec(
            (step4_root / "kirchhoff_residual_maps_by_K.png") if step4_root else Path("__missing__"),
            "dc/04_message_passing_sensitivity/kirchhoff_residual_maps_by_K.png",
            "DC Step 4 Kirchhoff residual maps by K",
        ),
        maybe_copy_spec(
            (step4_root / "conductance_correction_maps_by_K.png") if step4_root else Path("__missing__"),
            "dc/04_message_passing_sensitivity/conductance_correction_maps_by_K.png",
            "DC Step 4 conductance correction maps by K",
        ),
    ]
    return [spec for spec in specs if spec is not None]


def build_archive_specs(outputs_root: Path, args: argparse.Namespace) -> list[ArchiveSpec]:
    step0_root = resolve_step_output_dir(outputs_root, "00_ideal_models")
    step1_root = resolve_step_output_dir(outputs_root, "01_boundary_parameter_calibration")
    step2_root = resolve_step_output_dir(outputs_root, "02_physics_weight_sweep")
    step3_root = resolve_step_output_dir(outputs_root, "03_pressure_constraint_sensitivity")
    step4_root = resolve_step_output_dir(outputs_root, "04_message_passing_sensitivity")
    candidate_specs = [
        (args.include_step0_raw, "dc_step00_raw.tar.gz", "Full raw outputs for DC Step 0.", step0_root),
        (args.include_step1_raw, "dc_step01_raw.tar.gz", "Full raw outputs for DC Step 1.", step1_root),
        (args.include_step2_raw, "dc_step02_raw.tar.gz", "Full raw outputs for DC Step 2.", step2_root),
        (args.include_step3_raw, "dc_step03_raw.tar.gz", "Full raw outputs for DC Step 3.", step3_root),
        (args.include_step4_raw, "dc_step04_raw.tar.gz", "Full raw outputs for DC Step 4.", step4_root),
    ]
    specs: list[ArchiveSpec] = []
    for enabled, name, description, path in candidate_specs:
        if enabled and path is not None and path.exists():
            specs.append(ArchiveSpec(name=name, description=description, source=path.resolve()))
    return specs


def stage_copy_specs(specs: list[CopySpec], repo_bundle_root: Path, dry_run: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        destination = repo_bundle_root / spec.destination_relative
        row = {
            "artifact_name": str(spec.destination_relative),
            "bundle_type": "repo",
            "source_path": str(spec.source),
            "destination_path": str(destination),
            "size_bytes": file_or_tree_size(spec.source),
            "description": spec.description,
        }
        rows.append(row)
        print(f"[repo] {spec.source} -> {destination}")
        if dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if spec.source.is_dir():
            shutil.copytree(spec.source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(spec.source, destination)
    return rows


def create_tarball(archive_path: Path, source: Path, *, arcname_root: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source, arcname=str(source.relative_to(arcname_root)))


def stage_archives(specs: list[ArchiveSpec], release_bundle_root: Path, dry_run: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        archive_path = release_bundle_root / "dc" / spec.name
        row = {
            "artifact_name": spec.name,
            "bundle_type": "release",
            "source_path": str(spec.source),
            "destination_path": str(archive_path),
            "size_bytes": "",
            "description": spec.description,
        }
        print(f"[release] {spec.name}")
        print(f"  - {spec.source}")
        if not dry_run:
            create_tarball(archive_path, spec.source, arcname_root=PROJECT_ROOT)
            row["size_bytes"] = archive_path.stat().st_size
        rows.append(row)
    return rows


def create_repo_bundle_tar(repo_bundle_root: Path, release_bundle_root: Path) -> dict[str, object]:
    archive_path = release_bundle_root / "dc" / "dc_repo_bundle.tar.gz"
    create_tarball(archive_path, repo_bundle_root / "dc", arcname_root=repo_bundle_root)
    return {
        "artifact_name": "dc_repo_bundle.tar.gz",
        "bundle_type": "release",
        "source_path": str(repo_bundle_root / "dc"),
        "destination_path": str(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "description": "Convenience tarball of the repo-ready DC bundle.",
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["artifact_name", "bundle_type", "source_path", "destination_path", "size_bytes", "description"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_checksums(release_bundle_root: Path) -> None:
    dc_release_root = release_bundle_root / "dc"
    lines = []
    for artifact in sorted(dc_release_root.glob("*.tar.gz")):
        lines.append(f"{sha256sum(artifact)}  {artifact.name}")
    (release_bundle_root / "SHA256SUMS").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_bundle_readme(repo_bundle_dc_root: Path) -> None:
    readme = """# Somite21 DC Release Bundle

This folder contains the lightweight DC results bundle for the Somite21 dataset.

## Contents

- `00_ideal_models/`
  Poiseuille-only baseline summaries and figures.
- `01_boundary_parameter_calibration/`
  Boundary-weight calibration summaries and figures.
- `02_physics_weight_sweep/`
  Full sweep summary tables, representative-configuration table, and figures.
- `03_pressure_constraint_sensitivity/`
  Pressure-constraint summary tables, pairwise/correlation tables, and figures.
- `04_message_passing_sensitivity/`
  Message-passing depth summary files and K-sweep figures.

## Notes

- This is the lightweight bundle intended for GitHub release upload.
- It includes summary tables, metadata tables, and final figures.
- It does not include the full raw run directories for the DC sweeps.
- For Step 3 and Step 4, the packaged outputs correspond to the Somite21 run that used:
  - `lambda_q = 100`
  - `lambda_k = 0.1`
  - `lambda_delta = 0.1`

## Key files

- Step 2 representative configuration table:
  `02_physics_weight_sweep/representative_configurations.csv`
- Step 3 GNN summary:
  `03_pressure_constraint_sensitivity/pressure_constraint_gnn_summary.csv`
- Step 4 summary:
  `04_message_passing_sensitivity/summary.csv`
"""
    (repo_bundle_dc_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dc_publish_root = output_root / "dc"
    repo_bundle_root = dc_publish_root / "repo_bundle"
    release_bundle_root = dc_publish_root / "release_bundle"

    if not outputs_root.exists():
        raise FileNotFoundError(f"DC outputs root not found: {outputs_root}")

    copy_specs = build_repo_copy_specs(outputs_root)
    archive_specs = build_archive_specs(outputs_root, args)

    print(f"[info] staging repo bundle under {repo_bundle_root}")
    print(f"[info] staging release bundle under {release_bundle_root}")

    if args.dry_run:
        repo_rows = stage_copy_specs(copy_specs, repo_bundle_root, dry_run=True)
        release_rows = stage_archives(archive_specs, release_bundle_root, dry_run=True)
        print(f"[dry-run] repo artifacts: {len(repo_rows)}")
        print(f"[dry-run] release archives: {len(release_rows) + 1} (including dc_repo_bundle.tar.gz)")
        return

    if dc_publish_root.exists():
        shutil.rmtree(dc_publish_root)
    (repo_bundle_root / "dc").mkdir(parents=True, exist_ok=True)
    (release_bundle_root / "dc").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    rows.extend(stage_copy_specs(copy_specs, repo_bundle_root, dry_run=False))
    write_bundle_readme(repo_bundle_root / "dc")
    rows.extend(stage_archives(archive_specs, release_bundle_root, dry_run=False))
    rows.append(create_repo_bundle_tar(repo_bundle_root, release_bundle_root))

    manifest_path = dc_publish_root / "manifest.csv"
    write_manifest(manifest_path, rows)
    write_checksums(release_bundle_root)

    print(f"[ok] Wrote DC publish bundles under {dc_publish_root}")
    print(f"[ok] Manifest: {manifest_path}")
    print(f"[ok] Checksums: {release_bundle_root / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()
