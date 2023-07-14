import numpy as np

from .base_feature import *

class SpatialSphericalFeature(BaseFeature):
    """
    spherical harmonious coeffs || (y, x, z)
    """
    def __init__(self, config: dict):
        super().__init__(config)

    def make_spatial_features(self, data_dict: dict, key: str):
        sh_features = data_dict["spherical_coeffs"]  # (H, W, D, num_features)
        H, W, D, _ = sh_features.shape
        # ijk_grid = np.meshgrid(np.arange(H), np.arange(W), np.arange(D))
        ref = min([H, W, D])
        ijk_grid = np.meshgrid(np.linspace(0, H / ref, H), np.linspace(0, W / ref, W), np.linspace(0, D / ref, D))
        ijk_grid = np.stack(ijk_grid, axis=-1)  # (H, W, D, 3)

        return ijk_grid
