import abc
import numpy as np

from typing import Tuple


class BaseFeature(abc.ABC):
    def __init__(self, config: dict):
        """
        config: see test_retest_kmeans.yml
        """
        self.config = config

    def __process_thalamus(self, data_dict: dict, key: str, spatial_features: np.ndarray):
        """
        data_dict: the same as in .__call__(.)
        key: "left" or "right"
        spatial_features: (H, W, D, num_spatial_features)

        Returns
        --------
        {
            "features": (N_all, num_features),
            "coords": (N_all, 3),
            "shape": (H, W, D)
        }
        """
        sh_features = data_dict["spherical_coeffs"].get_fdata()  # (H, W, D, num_sh_features)
        H, W, D, _ = sh_features.shape
        thalamus_mask = data_dict[key]["thalamus_atlas_mask"].get_fdata()  # (H, W, D)
        i_inds, j_inds, k_inds = np.nonzero(thalamus_mask)
        ijk_inds = np.stack([i_inds, j_inds, k_inds], axis=-1)  # (N_all, 3)
        sh_features = sh_features * self.config["features"]["spherical_scale"] * np.sqrt(1 - self.config["features"]["spatial_weight"])
        spatial_features = spatial_features * np.sqrt(self.config["features"]["spatial_weight"])
        all_features = np.concatenate([sh_features[i_inds, j_inds, k_inds], spatial_features[i_inds, j_inds, k_inds]], axis=-1)  # (N_all, num_features)
        out_dict = {
            "features": all_features,  # （N_all, num_features）
            "coords": ijk_inds,  # （N_all, 3)
            "shape": (H, W, D)
        }

        return out_dict

    @abc.abstractmethod
    def make_spatial_features(self, data_dict: dict, key: str):
        """
        data_dict: the same as in .__call__(.)
        key: "left" or "right"
        """    
        raise NotImplementedError

    def __call__(self, data_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        data_dict: 
            See utils.utils.read_data(.); using a sub-structure from one run:
            {
                "left": ...,
                "right": ...,
                "spherical_coeffs": ...,
                **other_key_val_pairs
            }
        
        Returns
        -------
        {
            "left":
                {
                    "features": (N_all, num_features),
                    "coords": (N_all, 3),
                    "shape": (H, W, D)
                }
            "right": ...
        }
        """
        left_spatial_features = self.make_spatial_features(data_dict, "left")
        left_thalamus_dict = self.__process_thalamus(data_dict, "left", left_spatial_features)
        right_spatial_features = self.make_spatial_features(data_dict, "right")
        right_thalamus_dict = self.__process_thalamus(data_dict, "right", right_spatial_features)
        out_dict = {
            "left": left_thalamus_dict,
            "right": right_thalamus_dict
        }

        return out_dict
