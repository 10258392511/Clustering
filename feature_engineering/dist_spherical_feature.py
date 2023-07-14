from .base_feature import *


class DistSphericalFeature(BaseFeature):
    """
    spherical harmonious coeffs || dist_maps
    """
    def __init__(self, config: dict):
        super().__init__(config)
    
    def make_spatial_features(self, data_dict: dict, key: str):
        dist_maps = data_dict[key]["dist_maps"]
        keys = sorted(dist_maps.keys())
        spatial_features = [dist_maps[key].get_fdata() for key in keys]
        spatial_features = np.stack(spatial_features, axis=-1)  # (H, W, D, num_dist_maps)

        return spatial_features
    