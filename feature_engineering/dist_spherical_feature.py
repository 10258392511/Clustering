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

        # for key, dist_map_iter in dist_maps.items():
        #     if np.any(np.isnan(dist_map_iter.get_fdata())):
        #         print(key)

        spatial_features = np.stack(spatial_features, axis=-1)  # (H, W, D, num_dist_maps)
        # assert not np.any(np.isnan(spatial_features))

        return spatial_features
    