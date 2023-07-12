import numpy as np
import pandas as pd
import abc

from monai.metrics import compute_hausdorff_distance
from Clustering.configs import load_clustering_model
from Clustering.utils.utils import align_labels
from warnings import warn
from typing import Sequence


# class BaseCluster(abc.ABC):
#     def __init__(self, in_shape: Sequence[int], features: np.ndarray, ijk_inds: np.ndarray, params: dict):
#         self.in_shape = in_shape  # (D, H, W)
#         self.features = features  # (N, D_feature)
#         self.inds = ijk_inds  # (N, 3)
#         self.params = params  # for sklearn's clustering model; "clustering" node from config tree

#     @abc.abstractmethod
#     def fit_transform(self) -> np.ndarray:
#         """
#         Returns the volume of clustering labels, of shape .in_shape 
#         """
#         raise NotImplementedError

#     @abc.abstractmethod
#     def get_centroids(self):
#         raise NotImplementedError


class BaseCluster(abc.ABC):
    def __init__(self, config: dict):
        """
        See test_retest_kmeans.yml
        """
        self.config = config
        if config["init"] != "histology_atlas":
            self.models = {
                "left": load_clustering_model(config),
                "right": load_clustering_model(config)
            }
        else:
            self.models = {
                "left": None,
                "right": None
            }
    
    def _compute_means_from_histology_atlas(self, data_dict: dict, feature_dict: dict):
        """
        data_dict, feature_dict: the same as .fit_transform(.)

        Returns
        -------
        {
            left: (n_clusters, n_features)
            right: (n_clusters, n_features)
        }
        """
        out_dict = {}
        for key in ["left", "right"]:
            standard_atlas = data_dict[key]["nucleigroups"].get_fdata().astype(int)  # (H, W, D)
            coords = feature_dict[key]["coords"]  # (N_all, 3)
            features = feature_dict[key]["features"]  # (N_all, n_features)
            standard_thalamus_labels = standard_atlas[coords[:, 0], coords[:, 1], coords[:, 2]] - 1  # (N_all,)
            if np.any(standard_thalamus_labels < 0):
                warn("Selected histology atlas contains background")
                standard_thalamus_labels = standard_thalamus_labels[standard_thalamus_labels >= 0]
            if "n_clusters" in self.config["clustering"]["params"]:
                num_clusters = self.config["clustering"]["params"]["n_clusters"]
            else:
                num_clusters = self.config["clustering"]["params"]["n_components"]
            centroids = np.zeros((num_clusters, features.shape[-1]))
            for label_iter in range(num_clusters):
                indices = np.argwhere(standard_thalamus_labels == label_iter)  # (N',)
                centroids[label_iter, :] = features[indices, :].mean(axis=0)
            
            out_dict[key] = centroids

        return out_dict
    
    def init_means(self, data_dict: dict, feature_dict: dict):
        """
        Initialize labels with histology atlas centroids if specified.
        """
        pass
    
    def __process_thalamus(self, data_dict: dict, feature_dict: dict, key: str):
        """
        data_dict, feature_dict: the same as .fit_transform(.)
        key: "left" or "right"

        Returns
        -------
        {
            "atlas": (H, W, D),
            "atlas_not_remapped": (H, W, D),
            "model",
            "hausdorff_dist_mat": (num_clusters, num_clusters_standard)
        }
        """
        model = self.models[key]
        feature_thalamus_dict = feature_dict[key]
        labels = model.fit_predict(feature_thalamus_dict["features"])  # (N_all,)
        labels += 1  # 0-based -> 1-based; 0: bg
        atlas = np.zeros(feature_thalamus_dict["shape"], dtype=int)
        coords = feature_thalamus_dict["coords"]
        atlas[coords[:, 0], coords[:, 1], coords[:, 2]] = labels  # (H, W, D)
        out_dict = {}
        out_dict["atlas_not_remapped"] = atlas.copy()
        atlas, dist_df = self.__align_labels_for_one_thalamus(data_dict, key, atlas)
        out_dict.update({
            "atlas": atlas,
            "model": model,
            "hausdorff_dist_df": dist_df
        })

        return out_dict

    def __align_labels_for_one_thalamus(self, data_dict: dict, key: str, atlas: np.ndarray):
        """
        data_dict, feature_dict: the same as .fit_transform(.)
        key: "left" or "right"

        Returns
        -------
        aligned_atlas: (H, W, D)
        hausdorff_dist_df: (num_clusters, num_clusters_standard)
        """
        standard_atlas = data_dict[key]["nucleigroups"].get_fdata()  # (H, W, D)
        percentile = self.config["alignment"]["hausdorff_percent"]
        atlas_out, hausdorff_dist_df = align_labels(atlas, standard_atlas, percentile)

        return atlas_out, hausdorff_dist_df

    @abc.abstractmethod
    def create_probabilistic_maps(self, feature_dict: dict, key: str, hausdorff_dist_df: pd.DataFrame):
        """
        Uses .models to create probablistic map of shape (H, W, D, num_clusters).

        data_dict: the same as .fit_transform(.)
        key: "left" or "right

        Returns
        -------
        prob_maps: (H, W, D, num_clusters + 1)
        """
        raise NotImplemented

    def fit_transform(self, data_dict: dict, feature_dict: dict):
        """
        Create probabilistic atlases with aligned labels with standard atlases.

        Parameters
        ----------
        data_dict: 
        See utils.utils.read_data(.); using a sub-structure from one run:
            {
                "left": ...,
                "right": ...,
                "spherical_coeffs": ...,
                **other_key_val_pairs
            }
        feature_dict:
            {
                "left":
                    {
                        "features": (N_all, num_features),
                        "coords": (N_all, 3),
                        "shape": (H, W, D)
                    }
                "right": ...
            }
        
        Returns
        -------
        {
            "left":
                {
                    "atlas": (H, W, D),
                    "atlas_not_remapped": (H, W, D),
                    "prob_maps": (H, W, D, num_clusters + 1),  # channel is aligned with the standard atlas; 0: bg
                    "model",
                    "hausdorff_dist_df": (num_clusters, num_clusters_standard)
                }
            "right": ...
        }
        """
        self.init_means(data_dict, feature_dict)
        out_dict = {}
        for key in ["left", "right"]:
            out_dict_iter = self.__process_thalamus(data_dict, feature_dict, key)
            out_dict_iter["atlas"] = out_dict_iter["atlas"] * data_dict[key]["thalamus_atlas_mask"].get_fdata()
            out_dict_iter["prob_maps"] = self.create_probabilistic_maps(feature_dict, key, out_dict_iter["hausdorff_dist_df"])
            out_dict[key] = out_dict_iter
        
        return out_dict
