"""PyTorch datasets for temporal projections and pseudo-masks."""

from __future__ import annotations

from pathlib import Path


class ProjectionMaskDataset(__import__("torch").utils.data.Dataset):
    def __init__(self, projections_dir: str | Path, masks_dir: str | Path | None = None, tile_ids: list[int] | None = None):
        self.projections_dir = Path(projections_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.tile_ids = tile_ids or sorted(
            int(p.stem.split("_")[1]) for p in self.projections_dir.glob("tile_*_channels.npy")
        )

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int):
        np = __import__("numpy")
        torch = __import__("torch")
        tile_id = self.tile_ids[idx]
        x = np.load(self.projections_dir / f"tile_{tile_id:04d}_channels.npy").astype("float32")
        x_t = torch.from_numpy(x)
        if self.masks_dir is None:
            return {"image": x_t, "tile_id": tile_id}
        y = np.load(self.masks_dir / f"tile_{tile_id:04d}_mask.npy").astype("float32")[None, ...]
        return {"image": x_t, "mask": torch.from_numpy(y), "tile_id": tile_id}
