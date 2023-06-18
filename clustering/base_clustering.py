import numpy as np
import pandas as pd
import abc

from monai.metrics import compute_hausdorff_distance
from Clustering.configs import load_clustering_model
from typing import Sequence

# # TODO: run clustering on left & right thalamus, and then combine to one clustering result 
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
        self.models = {
            "left": load_clustering_model(config),
            "right": load_clustering_model(config)
        }
    
    def __process_thalamus(self, data_dict: dict, feature_dict: dict, key: str):
        """
        data_dict, feature_dict: the same as .fit_transform(.)
        key: "left" or "right"

        Returns
        -------
        {
            "atlas": (H, W, D),
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
        atlas, dist_mat = self.__align_labels_for_one_thalamus(data_dict, key, atlas)
        out_dict = {
            "atlas": atlas,
            "model": model,
            "hausdorff_dist_mat": dist_mat
        }

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
        atlas = atlas.astype(int)
        standard_atlas = data_dict[key]["nucleigroups"].get_fdata()  # (H, W, D)
        standard_atlas = standard_atlas.astype(int)
        percentile = self.config["alignment"]["hausdorff_percent"]
        cluster_inds = np.unique(atlas)
        cluster_inds.sort()
        cluster_inds = cluster_inds[1:]
        standard_inds = np.unique(standard_atlas)
        standard_inds.sort()
        standard_inds = standard_inds[1:]

        hausdorff_dist_mat = np.zeros((len(cluster_inds), len(standard_inds)))
        hausdorff_dist_df = pd.DataFrame(data=hausdorff_dist_mat, index=cluster_inds, columns=standard_inds)
        for i in hausdorff_dist_df.shape[0]:
            for j in hausdorff_dist_df.shape[1]:
                cluster_label = int(hausdorff_dist_df.index[i])
                standard_label = int(hausdorff_dist_df.columns[j])
                cluster_atlas_iter = atlas.copy()
                cluster_atlas_iter[cluster_atlas_iter == cluster_label] = 1
                cluster_atlas_iter[cluster_atlas_iter != cluster_label] = 0  # (H, W, D)

                standard_atlas_iter = standard_atlas.copy()
                standard_atlas_iter[standard_atlas == standard_label] = 1
                standard_atlas_iter[standard_atlas != standard_label] = 0  # (H, W, D)
                
                hausdorff_dist_df.iloc[i, j] = compute_hausdorff_distance(cluster_atlas_iter[None, ...], standard_atlas_iter[None, ...], percentile=percentile)
        
        atlas2standard = hausdorff_dist_df.idxmin(axis=1)
        for label_iter in cluster_inds:
            label_iter = int(label_iter)
            atlas[atlas == label_iter] = atlas2standard[label_iter]

        return atlas, hausdorff_dist_df

    @abc.abstractmethod
    def create_probabilistic_maps(self, data_dict: dict, key: str):
        """
        Uses .models to create probablistic map of shape (H, W, D, num_clusters).

        data_dict: the same as .fit_transform(.)
        key: "left" or "right
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
                    "prob_maps": (H, W, D, num_clusters),
                    "model",
                    "hausdorff_dist_mat": (num_clusters, num_clusters_standard)
                }
            "right": ...
        }
        """
        out_dict = {}
        for key in ["left", "right"]:
            out_dict_iter = self.__process_thalamus(data_dict, feature_dict, key)
            out_dict_iter["prob_maps"] = self.create_probabilistic_maps(data_dict, key)
            out_dict[key] = out_dict_iter
        
        return out_dict_iter
