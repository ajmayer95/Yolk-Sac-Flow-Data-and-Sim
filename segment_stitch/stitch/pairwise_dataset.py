"""Scaffold for future learned pairwise tile compatibility datasets.

Future versions should create tile pairs from manual positions, crop overlapping
regions, and return labels for overlap probability plus relative translation,
scale, or affine parameters. This is intentionally not wired into the baseline
workflow yet.
"""


class PairwiseTileDataset:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Pairwise learned stitching dataset is a scaffold.")
