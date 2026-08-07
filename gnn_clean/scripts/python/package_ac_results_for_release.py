#!/usr/bin/env python
"""Stage repo-ready and release-ready AC results bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "ac"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "publish"

STEP0_METRICS = (
    "complex_flow_rmse_nl_s",
    "kirchhoff_rms_per_internal_node_nl_s",
    "arterial_pressure_phase_difference_deg",
)


@dataclass(frozen=True)
class CopySpec:
    source: Path
    destination_relative: Path
    description: str


@dataclass(frozen=True)
class ArchiveSpec:
    name: str
    description: str
    sources: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-step0-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-step3-h1-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-step3-h2-raw", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


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


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {description}: {path}")
    return path


def build_repo_copy_specs(outputs_root: Path) -> list[CopySpec]:
    return [
        CopySpec(
            source=require_path(
                outputs_root / "00_ideal_models" / "distensibility_sweep" / "distensibility_sweep_metrics.csv",
                "AC Step 0 distensibility summary CSV",
            ),
            destination_relative=Path("ac/00_ideal_models/distensibility_sweep/distensibility_sweep_metrics.csv"),
            description="AC Step 0 distensibility sweep summary table",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "00_ideal_models" / "distensibility_sweep" / "figures",
                "AC Step 0 distensibility figures",
            ),
            destination_relative=Path("ac/00_ideal_models/distensibility_sweep/figures"),
            description="AC Step 0 distensibility sweep figures",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "00_ideal_models" / "harmonic_stage1_admittance_model_comparison" / "H1" / "figures",
                "AC Step 0 harmonic H1 figures",
            ),
            destination_relative=Path("ac/00_ideal_models/harmonic_stage1_admittance_model_comparison/H1/figures"),
            description="AC Step 0 harmonic-stage comparison figures for H1",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "00_ideal_models" / "harmonic_stage1_admittance_model_comparison" / "H2" / "figures",
                "AC Step 0 harmonic H2 figures",
            ),
            destination_relative=Path("ac/00_ideal_models/harmonic_stage1_admittance_model_comparison/H2/figures"),
            description="AC Step 0 harmonic-stage comparison figures for H2",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "01_boundary_parameter_calibration" / "boundary_parameter_calibration_summary_H1.csv",
                "AC Step 1 H1 summary CSV",
            ),
            destination_relative=Path("ac/01_boundary_parameter_calibration/boundary_parameter_calibration_summary_H1.csv"),
            description="AC Step 1 H1 boundary-parameter calibration summary",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "01_boundary_parameter_calibration" / "boundary_parameter_calibration_summary_H2.csv",
                "AC Step 1 H2 summary CSV",
            ),
            destination_relative=Path("ac/01_boundary_parameter_calibration/boundary_parameter_calibration_summary_H2.csv"),
            description="AC Step 1 H2 boundary-parameter calibration summary",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "01_boundary_parameter_calibration" / "figures",
                "AC Step 1 figures",
            ),
            destination_relative=Path("ac/01_boundary_parameter_calibration/figures"),
            description="AC Step 1 boundary-parameter calibration figures",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "02_physics_weight_sweep" / "H1" / "ac_physics_weight_analysis.yaml",
                "AC Step 2 H1 analysis YAML",
            ),
            destination_relative=Path("ac/02_physics_weight_sweep/H1/ac_physics_weight_analysis.yaml"),
            description="AC Step 2 H1 physics-weight analysis summary",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "02_physics_weight_sweep" / "H1" / "ac_physics_weight_representatives.csv",
                "AC Step 2 H1 representatives CSV",
            ),
            destination_relative=Path("ac/02_physics_weight_sweep/H1/ac_physics_weight_representatives.csv"),
            description="AC Step 2 H1 representative configurations",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "02_physics_weight_sweep" / "H1" / "figures",
                "AC Step 2 H1 figures",
            ),
            destination_relative=Path("ac/02_physics_weight_sweep/H1/figures"),
            description="AC Step 2 H1 figures",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "02_physics_weight_sweep" / "H2" / "ac_physics_weight_analysis.yaml",
                "AC Step 2 H2 analysis YAML",
            ),
            destination_relative=Path("ac/02_physics_weight_sweep/H2/ac_physics_weight_analysis.yaml"),
            description="AC Step 2 H2 physics-weight analysis summary",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "02_physics_weight_sweep" / "H2" / "ac_physics_weight_representatives.csv",
                "AC Step 2 H2 representatives CSV",
            ),
            destination_relative=Path("ac/02_physics_weight_sweep/H2/ac_physics_weight_representatives.csv"),
            description="AC Step 2 H2 representative configurations",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "02_physics_weight_sweep" / "H2" / "figures",
                "AC Step 2 H2 figures",
            ),
            destination_relative=Path("ac/02_physics_weight_sweep/H2/figures"),
            description="AC Step 2 H2 figures",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "03_distensibility_alpha_profiles" / "H1" / "combined_results.csv",
                "AC Step 3 H1 combined results CSV",
            ),
            destination_relative=Path("ac/03_distensibility_alpha_profiles/H1/combined_results.csv"),
            description="AC Step 3 H1 combined results table",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "03_distensibility_alpha_profiles" / "H1" / "metric_minima.csv",
                "AC Step 3 H1 minima CSV",
            ),
            destination_relative=Path("ac/03_distensibility_alpha_profiles/H1/metric_minima.csv"),
            description="AC Step 3 H1 metric minima table",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "03_distensibility_alpha_profiles" / "H1" / "representative_configurations.csv",
                "AC Step 3 H1 representatives CSV",
            ),
            destination_relative=Path("ac/03_distensibility_alpha_profiles/H1/representative_configurations.csv"),
            description="AC Step 3 H1 representative configurations",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "03_distensibility_alpha_profiles" / "H1" / "figures",
                "AC Step 3 H1 figures",
            ),
            destination_relative=Path("ac/03_distensibility_alpha_profiles/H1/figures"),
            description="AC Step 3 H1 figures, including curated representative fields",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "03_distensibility_alpha_profiles" / "H2" / "combined_results.csv",
                "AC Step 3 H2 combined results CSV",
            ),
            destination_relative=Path("ac/03_distensibility_alpha_profiles/H2/combined_results.csv"),
            description="AC Step 3 H2 combined results table",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "03_distensibility_alpha_profiles" / "H2" / "metric_minima.csv",
                "AC Step 3 H2 minima CSV",
            ),
            destination_relative=Path("ac/03_distensibility_alpha_profiles/H2/metric_minima.csv"),
            description="AC Step 3 H2 metric minima table",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "03_distensibility_alpha_profiles" / "H2" / "representative_configurations.csv",
                "AC Step 3 H2 representatives CSV",
            ),
            destination_relative=Path("ac/03_distensibility_alpha_profiles/H2/representative_configurations.csv"),
            description="AC Step 3 H2 representative configurations",
        ),
        CopySpec(
            source=require_path(
                outputs_root / "03_distensibility_alpha_profiles" / "H2" / "figures",
                "AC Step 3 H2 figures",
            ),
            destination_relative=Path("ac/03_distensibility_alpha_profiles/H2/figures"),
            description="AC Step 3 H2 figures",
        ),
    ]


def select_step0_raw_dirs(outputs_root: Path) -> list[Path]:
    metrics_path = require_path(
        outputs_root / "00_ideal_models" / "distensibility_sweep" / "distensibility_sweep_metrics.csv",
        "AC Step 0 distensibility summary CSV",
    )
    df = pd.read_csv(metrics_path)
    required_columns = {"source_summary_path", "harmonic_number", "model_name", "alpha"}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required Step 0 columns in {metrics_path}: {missing}")

    selections: set[Path] = set()
    for (harmonic_number, model_name, alpha), group in df.groupby(["harmonic_number", "model_name", "alpha"], dropna=False):
        del harmonic_number, model_name, alpha
        for metric in STEP0_METRICS:
            if metric not in group.columns:
                continue
            metric_values = pd.to_numeric(group[metric], errors="coerce")
            if metric == "arterial_pressure_phase_difference_deg":
                metric_values = metric_values.abs()
            valid = group[metric_values.notna()].copy()
            if valid.empty:
                continue
            valid_metric_values = metric_values[metric_values.notna()]
            best_index = valid_metric_values.idxmin()
            summary_path = Path(str(group.loc[best_index, "source_summary_path"])).expanduser().resolve()
            selections.add(require_path(summary_path.parent, f"Step 0 raw run directory for {metric}"))
    return sorted(selections)


def select_step3_raw_dirs(outputs_root: Path, harmonic_label: str) -> list[Path]:
    csv_path = require_path(
        outputs_root / "03_distensibility_alpha_profiles" / harmonic_label / "representative_configurations.csv",
        f"AC Step 3 {harmonic_label} representatives CSV",
    )
    df = pd.read_csv(csv_path)
    if "profile_run_dir" not in df.columns:
        raise ValueError(f"Missing profile_run_dir column in {csv_path}")
    selections = {
        require_path(Path(str(value)).expanduser().resolve(), f"Step 3 {harmonic_label} raw run directory")
        for value in df["profile_run_dir"].dropna().astype(str).unique()
    }
    return sorted(selections)


def build_archive_specs(outputs_root: Path, args: argparse.Namespace) -> list[ArchiveSpec]:
    specs: list[ArchiveSpec] = []
    if args.include_step0_raw:
        specs.append(
            ArchiveSpec(
                name="ac_step00_representative_raw.tar.gz",
                description=(
                    "Representative AC Step 0 raw runs selected from distensibility-sweep minima "
                    "over flow RMSE, Kirchhoff RMS, and arterial phase difference."
                ),
                sources=tuple(select_step0_raw_dirs(outputs_root)),
            )
        )
    if args.include_step3_h1_raw:
        specs.append(
            ArchiveSpec(
                name="ac_step03_H1_representative_raw.tar.gz",
                description="Representative AC Step 3 raw runs for H1, taken from representative_configurations.csv.",
                sources=tuple(select_step3_raw_dirs(outputs_root, "H1")),
            )
        )
    if args.include_step3_h2_raw:
        specs.append(
            ArchiveSpec(
                name="ac_step03_H2_representative_raw.tar.gz",
                description="Representative AC Step 3 raw runs for H2, taken from representative_configurations.csv.",
                sources=tuple(select_step3_raw_dirs(outputs_root, "H2")),
            )
        )
    return specs


def stage_copy_specs(
    specs: list[CopySpec],
    repo_bundle_root: Path,
    dry_run: bool,
) -> list[dict[str, object]]:
    manifest_rows: list[dict[str, object]] = []
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
        manifest_rows.append(row)
        print(f"[repo] {spec.source} -> {destination}")
        if dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if spec.source.is_dir():
            shutil.copytree(spec.source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(spec.source, destination)
    return manifest_rows


def create_tarball(archive_path: Path, sources: tuple[Path, ...], *, arcname_root: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for source in sources:
            arcname = source.relative_to(arcname_root)
            tar.add(source, arcname=str(arcname))


def stage_archives(
    specs: list[ArchiveSpec],
    release_bundle_root: Path,
    dry_run: bool,
) -> list[dict[str, object]]:
    manifest_rows: list[dict[str, object]] = []
    for spec in specs:
        if not spec.sources:
            raise ValueError(f"Archive {spec.name} has no selected sources.")
        archive_path = release_bundle_root / "ac" / spec.name
        joined_sources = ";".join(str(path) for path in spec.sources)
        row = {
            "artifact_name": spec.name,
            "bundle_type": "release",
            "source_path": joined_sources,
            "destination_path": str(archive_path),
            "size_bytes": "",
            "description": spec.description,
        }
        print(f"[release] {spec.name}")
        for source in spec.sources:
            print(f"  - {source}")
        if not dry_run:
            create_tarball(archive_path, spec.sources, arcname_root=PROJECT_ROOT)
            row["size_bytes"] = archive_path.stat().st_size
        manifest_rows.append(row)
    return manifest_rows


def create_repo_bundle_tar(repo_bundle_root: Path, release_bundle_root: Path) -> dict[str, object]:
    archive_path = release_bundle_root / "ac" / "ac_repo_bundle.tar.gz"
    create_tarball(archive_path, (repo_bundle_root / "ac",), arcname_root=repo_bundle_root)
    return {
        "artifact_name": "ac_repo_bundle.tar.gz",
        "bundle_type": "release",
        "source_path": str(repo_bundle_root / "ac"),
        "destination_path": str(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "description": "Convenience tarball of the repo-ready AC bundle.",
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["artifact_name", "bundle_type", "source_path", "destination_path", "size_bytes", "description"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_checksums(release_bundle_root: Path) -> None:
    ac_release_root = release_bundle_root / "ac"
    entries: list[str] = []
    for artifact in sorted(ac_release_root.glob("*.tar.gz")):
        entries.append(f"{sha256sum(artifact)}  {artifact.name}")
    (release_bundle_root / "SHA256SUMS").write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    ac_publish_root = output_root / "ac"
    repo_bundle_root = ac_publish_root / "repo_bundle"
    release_bundle_root = ac_publish_root / "release_bundle"

    if not outputs_root.exists():
        raise FileNotFoundError(f"AC outputs root not found: {outputs_root}")

    copy_specs = build_repo_copy_specs(outputs_root)
    archive_specs = build_archive_specs(outputs_root, args)

    print(f"[info] staging repo bundle under {repo_bundle_root}")
    print(f"[info] staging release bundle under {release_bundle_root}")

    if args.dry_run:
        repo_rows = stage_copy_specs(copy_specs, repo_bundle_root, dry_run=True)
        release_rows = stage_archives(archive_specs, release_bundle_root, dry_run=True)
        print(f"[dry-run] repo artifacts: {len(repo_rows)}")
        print(f"[dry-run] release archives: {len(release_rows) + 1} (including ac_repo_bundle.tar.gz)")
        return

    if ac_publish_root.exists():
        shutil.rmtree(ac_publish_root)
    (repo_bundle_root / "ac").mkdir(parents=True, exist_ok=True)
    (release_bundle_root / "ac").mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    manifest_rows.extend(stage_copy_specs(copy_specs, repo_bundle_root, dry_run=False))
    manifest_rows.extend(stage_archives(archive_specs, release_bundle_root, dry_run=False))
    manifest_rows.append(create_repo_bundle_tar(repo_bundle_root, release_bundle_root))

    manifest_path = ac_publish_root / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    write_checksums(release_bundle_root)

    print(f"[ok] Wrote AC publish bundles under {ac_publish_root}")
    print(f"[ok] Manifest: {manifest_path}")
    print(f"[ok] Checksums: {release_bundle_root / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()
