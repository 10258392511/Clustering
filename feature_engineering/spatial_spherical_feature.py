import numpy as np

from .base_feature import BaseFeature
from monai.transforms import LoadImaged
from typing import Dict, Tuple


class SpatialSphericalFeature(BaseFeature):
    """
    (y, x, z) || spherical harmonious coefficients
    """
    def __init__(self, path_dict: Dict[str, str], params: dict):
        """
        path_dict: keys: thalamus_mask (H, W, D), spherical_coeff (H, W, D, N), (dwi (H, W, D))
        params: spherical_scale, spatial_weight, if_dwi
        """
        super().__init__()
        assert 0 <= params["spatial_weight"] <= 1
        self.params = params
        self.image_loader = LoadImaged(["thalamus_mask", "spherical_coeff"], image_only=True)
        self.images = self.image_loader(path_dict)
        self.H, self.W, self.D = self.images["thalamus_mask"].shape
        ijk_grid = np.meshgrid(np.arange(self.H), np.arange(self.W), np.arange(self.D), indexing="ij")  # tuple[(H, W, D)]
        self.ijk_grid = np.stack(ijk_grid, axis=-1)  # (H, W, D, 3)

    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        i_inds, j_inds, k_inds = np.nonzero(self.images["thalamus_mask"])  # each: (N_all,)
        ijk_inds = np.stack([i_inds, j_inds, k_inds], axis=-1)  # (N_all, 3)
        spatial_feats = self.ijk_grid[i_inds, j_inds, k_inds] * np.sqrt(self.params["spatial_weight"])
        spherical_feats = self.images["spherical_coeff"][i_inds, j_inds, k_inds] * self.params["spherical_scale"] * np.sqrt(1 - self.params["spatial_weight"])
        feats = np.concatenate([spatial_feats, spherical_feats], axis=-1)  # (N_all, D_all)

        # (N, D), (N, 3)
        return feats, ijk_inds
    