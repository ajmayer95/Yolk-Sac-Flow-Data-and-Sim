"""Scaffold for future pairwise deep stitching models.

Future models can predict overlap probability and relative transform parameters
for tile pairs, then feed those constraints into pose graph optimization.
"""


class PairwiseStitchingModel:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Pairwise learned stitching model is a scaffold.")
