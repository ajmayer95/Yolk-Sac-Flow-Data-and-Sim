"""Scaffold for future global pose graph optimization.

Future versions should solve globally consistent tile poses from pairwise
translation, scale, or affine predictions and validate them against the manual
tile positions and stitched TIFF reference.
"""


def solve_pose_graph(*args, **kwargs):
    raise NotImplementedError("Pose graph optimization is a scaffold.")
