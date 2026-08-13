# Conversion Workflows

This directory documents the dataset-conversion workflows that matter for this repo.

There are three distinct workflow families:

1. `harmonized_scaled_dataset.gpickle`
2. the Somite21 dataset under `datasets/quail-flow-share`
3. the canonical mosaic dataset family around `mosaic_graph_canonical.gpickle`

## 1. emb1 harmonized workflow

This is the clean scripted workflow for the emb1 mosaic dataset.

Input:
- `datasets/emb1_mosaic_graph_analyzed.gpickle`

Output:
- `datasets/harmonized_scaled_dataset.gpickle`

Command:

```bash
python scripts/conversion/harmonize_emb1_mosaic_dataset.py \
  --input-graph datasets/emb1_mosaic_graph_analyzed.gpickle \
  --output-graph datasets/harmonized_scaled_dataset.gpickle \
  --overwrite
```

What it does:
- loads the analyzed emb1 mosaic graph
- fits one flow scale per tile from overlap consistency
- applies those scales to the graph measurements
- writes the harmonized repo-level dataset

DC flow convention in the converted output:
- `Q_DC`, `mean_Q`, and related top-level magnitude fields are written as unsigned magnitudes
- direction is carried by `flow_from` and `flow_to`
- the signed scalar is preserved separately in `Q_DC_signed_nl_s` and per-measurement `mean_Q_signed_nL_s`

Script:
- `scripts/conversion/harmonize_emb1_mosaic_dataset.py`

## 2. Somite21 quail-share workflow

This is the complete conversion workflow for the Somite21 dataset in:

- `datasets/quail-flow-share/data/somite21_mosaic.gpickle`

Final repo-level output:
- `datasets/somite21_mosaic_cut_pipeline_ready.gpickle`

Run the workflow in this order.

### Step 1. Build the cut-ready graph

```bash
python scripts/conversion/prepare_somite21_cut_for_pipeline.py \
  --input-graph datasets/quail-flow-share/data/somite21_mosaic.gpickle \
  --output-graph datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --overwrite
```

### Step 2. Backfill geometry compatibility aliases

```bash
python scripts/conversion/repair_somite21_cut_geometry_compat.py \
  --graph-path datasets/somite21_mosaic_cut_pipeline_ready.gpickle
```

### Step 3. Backfill synthetic-boundary AC phasors

```bash
python scripts/conversion/backfill_somite21_cut_boundary_phasors.py \
  --graph-path datasets/somite21_mosaic_cut_pipeline_ready.gpickle
```

What it does:
- converts the quail-share raw mosaic graph into the repo’s cut-ready DC/AC workflow graph
- repairs geometry aliases expected by older geometry readers
- repairs synthetic-boundary harmonic aliases expected by AC steps

Scripts:
- `scripts/conversion/prepare_somite21_cut_for_pipeline.py`
- `scripts/conversion/repair_somite21_cut_geometry_compat.py`
- `scripts/conversion/backfill_somite21_cut_boundary_phasors.py`

Optional alternative product:

```bash
python scripts/conversion/export_somite21_demo_conservative_graph.py \
  --input-graph datasets/quail-flow-share/data/somite21_mosaic.gpickle \
  --output-graph datasets/somite21_mosaic_ml_conservative.gpickle \
  --overwrite
```

This creates `datasets/somite21_mosaic_ml_conservative.gpickle`, which is separate from the cut-pipeline workflow.

## 3. Canonical mosaic workflow family

This family is different from the two workflows above.

Repo evidence shows these canonical-family datasets:

- `datasets/mosaic_graph_canonical.gpickle`
- `datasets/mosaic_graph_norm_canonical.gpickle`
- `datasets/mosaic_graph_canonical_harmonized/mosaic_graph_canonical_harmonized.gpickle`

What the repo clearly preserves:

- `mosaic_graph_canonical.gpickle` exists as a canonical source graph
- `mosaic_graph_norm_canonical.gpickle` is a normalized slim derivative for sharing
- `mosaic_graph_canonical_harmonized.gpickle` is the harmonized canonical workflow graph used by `scripts/canonical_rerun`

### What is inside `mosaic_graph_canonical.gpickle`

Inspection of `datasets/mosaic_graph_canonical.gpickle` and the uploaded
`datasets/mosaic_graph_canonical.gpickle.zip` shows:

- the zip is just a packaged copy of the same graph file
- node/edge counts are `4393` nodes and `6456` edges
- graph-level metadata is minimal:
  - `tile_f0_piv`
  - `arm_lengths_normalized = True`
  - `arm_lengths_normalize_mode = 'per_type_avg'`
  - `arm_L_target_source_px`
  - `arm_L_target_sink_px`
  - `bc_harmonics_convention = 'solver_ready'`
- boundary nodes are already present as 2 sources + 2 sinks
- many edges still carry raw per-tile measurement payloads under `measurements_piv`
- edges also carry pre-aggregated per-edge helper fields such as:
  - `mean_Q_piv`
  - `amp_Q_piv`
  - `phase_piv`
  - `flow_from_piv` / `flow_to_piv`
  - `_h_amp_H1`, `_h_phase_H1`, `_h_Z_H1` and corresponding H2/H3 fields
  - `f0_hz_piv`
  - `Q_t_piv`, `v_t_piv`
- this graph does **not** yet expose the later normalized/harmonized top-level fields like:
  - `Q_H1`, `Q_H2`, `Q_H3`
  - top-level `Q_DC`
  - `tiles`
  - slimmed-out measurement lists

Practical interpretation:
- `mosaic_graph_canonical.gpickle` is best treated as a pre-final canonical PIV graph
- it already has canonicalized per-tile measurements and normalized arm geometry
- it still needs a conversion/packaging pass to promote and aggregate those raw `*_piv` / `measurements_piv` fields into the cleaner solver-facing schema used by the normalized and harmonized descendants

What the repo does **not** clearly preserve:

- a single conversion script in `gnn_clean` that builds `datasets/mosaic_graph_canonical.gpickle` from an earlier raw source

So for this family, the honest workflow status is:

### 3a. Canonical source graph

Input:
- not cleanly captured in this repo as a scripted conversion step

Output:
- `datasets/mosaic_graph_canonical.gpickle`

Status:
- provenance is present only implicitly through dataset contents and downstream workflow usage
- there is currently no `scripts/conversion/*.py` builder here that reproduces this file from a more raw upstream artifact
- however, the graph contents make the next conversion step fairly clear

### 3b. Normalized canonical derivative

Repo documentation:
- `README_norm_canonical.md`

That README states:
- `mosaic_graph_norm_canonical.gpickle` is a normalized canonical graph for external sharing
- it is a slimmed derivative of `mosaic_graph_canonical_2_normalized.gpickle`
- it preserves the same physics and normalization while stripping intermediate caches and per-tile measurement lists

From the graph contents, that normalization/packaging step appears to do roughly this:

1. start from `mosaic_graph_canonical.gpickle`
2. promote canonical `*_piv` fields to stable top-level solver fields
3. aggregate or expose harmonic phasors as top-level `Q_H1`, `Q_H2`, `Q_H3`
4. retain solver-ready boundary harmonics
5. strip bulky per-tile lists like `measurements_piv` for the slim external-sharing graph

### 3c. Harmonized canonical workflow graph

Workflow consumer:
- `scripts/canonical_rerun/run_python_workflow.sh`
- `scripts/canonical_rerun/README.md`

That workflow uses:

- `datasets/mosaic_graph_canonical_harmonized/mosaic_graph_canonical_harmonized.gpickle`

as the canonical harmonized graph for the DC/AC rerun pipeline.

Practical conclusion for now:
- the repo has a complete runnable workflow **using** the harmonized canonical graph
- the repo has documentation for the normalized canonical derivative
- the repo does **not** yet have a clean conversion script documenting how `mosaic_graph_canonical.gpickle` itself was originally created
- but the repo now has enough evidence to define a future conversion script for the next step:
  `mosaic_graph_canonical.gpickle -> normalized/slim canonical graph`

If we want a full reproducible conversion workflow for the canonical source graph, that will need to be reconstructed from upstream notebooks/scripts or external source data and then added explicitly.

## Summary map

- `harmonize_emb1_mosaic_dataset.py`
  input: `datasets/emb1_mosaic_graph_analyzed.gpickle`
  output: `datasets/harmonized_scaled_dataset.gpickle`

- `prepare_somite21_cut_for_pipeline.py`
  input: `datasets/quail-flow-share/data/somite21_mosaic.gpickle`
  output: `datasets/somite21_mosaic_cut_pipeline_ready.gpickle`

- `repair_somite21_cut_geometry_compat.py`
  input/output: `datasets/somite21_mosaic_cut_pipeline_ready.gpickle` in place

- `backfill_somite21_cut_boundary_phasors.py`
  input/output: `datasets/somite21_mosaic_cut_pipeline_ready.gpickle` in place

- `export_somite21_demo_conservative_graph.py`
  input: `datasets/quail-flow-share/data/somite21_mosaic.gpickle`
  output: `datasets/somite21_mosaic_ml_conservative.gpickle`

- canonical family
  current repo status: documented consumers and derivatives exist, but the original builder for `datasets/mosaic_graph_canonical.gpickle` is not yet captured as a clean conversion script in `scripts/conversion`
