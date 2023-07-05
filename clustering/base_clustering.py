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
        atlas, dist_df = self.__align_labels_for_one_thalamus(data_dict, key, atlas)
        out_dict = {
            "atlas": atlas,
            "model": model,
            "hausdorff_dist_df": dist_df
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
        # print(f"atlas: {(atlas > 0).sum()}")
        # print(f"standard_atlas: {(standard_atlas > 0).sum()}")

        percentile = self.config["alignment"]["hausdorff_percent"]
        cluster_inds = np.unique(atlas)
        cluster_inds.sort()
        cluster_inds = cluster_inds[1:]
        standard_inds = np.unique(standard_atlas)
        standard_inds.sort()
        standard_inds = standard_inds[1:]
        
        # for label in cluster_inds:
        #     print(f"atlas {label}: {(atlas == label).sum()}")
        
        # for label in standard_inds:
        #     print(f"standard {label}: {(standard_atlas == label).sum()}")

        hausdorff_dist_mat = np.zeros((len(cluster_inds), len(standard_inds)))
        hausdorff_dist_df = pd.DataFrame(data=hausdorff_dist_mat, index=cluster_inds, columns=standard_inds)
        for i in range(hausdorff_dist_df.shape[0]):
            for j in range(hausdorff_dist_df.shape[1]):
                cluster_label = int(hausdorff_dist_df.index[i])
                standard_label = int(hausdorff_dist_df.columns[j])
                cluster_atlas_iter = atlas.copy()
                mask = (cluster_atlas_iter == cluster_label)
                cluster_atlas_iter[mask] = 1
                cluster_atlas_iter[~mask] = 0  # (H, W, D)

                standard_atlas_iter = standard_atlas.copy()
                mask = (standard_atlas_iter == standard_label)
                standard_atlas_iter[mask] = 1
                standard_atlas_iter[~mask] = 0  # (H, W, D)
                
                # print(f"{(i, j)}")
                # print(f"{(cluster_label, standard_label)}")
                # print(f"cluster_atlas_iter: {(cluster_atlas_iter == 1).sum()}")
                # print(f"standard_atlas_iter: {(standard_atlas_iter == 1).sum()}")
                dist = compute_hausdorff_distance(cluster_atlas_iter[None, None, ...], standard_atlas_iter[None, None, ...], percentile=percentile)
                hausdorff_dist_df.iloc[i, j] = dist[0, 0].item()
        
        atlas2standard = hausdorff_dist_df.idxmin(axis=1)
        atlas_out = atlas.copy()
        for label_iter in cluster_inds:
            label_iter = int(label_iter)
            atlas_out[atlas == label_iter] = atlas2standard[label_iter]

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
                    "prob_maps": (H, W, D, num_clusters + 1),  # channel is aligned with the standard atlas; 0: bg
                    "model",
                    "hausdorff_dist_df": (num_clusters, num_clusters_standard)
                }
            "right": ...
        }
        """
        out_dict = {}
        for key in ["left", "right"]:
            out_dict_iter = self.__process_thalamus(data_dict, feature_dict, key)
            out_dict_iter["atlas"] = out_dict_iter["atlas"] * data_dict[key]["thalamus_atlas_mask"].get_fdata()
            out_dict_iter["prob_maps"] = self.create_probabilistic_maps(feature_dict, key, out_dict_iter["hausdorff_dist_df"])
            out_dict[key] = out_dict_iter
        
        return out_dict
