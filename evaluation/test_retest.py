import numpy as np
import nibabel as nib
import pandas as pd
import torch
import abc
import os
import Clustering.utils.pytorch_utils as ptu

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from scipy.spatial.distance import cdist
from Clustering.utils.utils import apply_affine, align_labels, save_image
from typing import Sequence

# legacy code
class TestRetestBaseEvaluator(abc.ABC):
    def __init__(self, test_labels: Sequence[np.ndarray], test_centroids: Sequence[np.ndarray], 
                 retest_labels: Sequence[np.ndarray], retest_centroids: Sequence[np.ndarray], params: dict):
        """
        test_labels, retest_labels: List[(H, W, D)]
        test_centroids, retest_centroids: List[(num_clusters, D_feature)]
        params: hausdorff_dist_pct = 95, num_clusters = 7, centroid_dist_order = 2, standard_space_atlases: List[str]
        .metric_df: colnames: dsc, hausdorff_dist, centroid_dist
        """
        self.params = params
        self.test_centroids = test_centroids
        self.retest_centroids = retest_centroids
        
        self.post_processor = AsDiscrete(to_onehot=self.params["num_clusters"] + 1)  # including bg
        self.test_label2retest_label = [np.zeros((self.params["num_clusters"] + 1,), dtype=int) for _ in range(len(test_labels))]  # to be computed; np.ndarray: e.g. 0 -> 0, 1 -> 2, ...
        self.compute_label_mapping()
        self.__align_centroids()

        self.test_labels = []
        self.retest_labels = []
        for test_label, retest_label, mapping in zip(test_labels, retest_labels, self.test_label2retest_label):
            test_label = self.align_labels(test_label, mapping)
            self.test_labels.append(self.post_processor(test_label[None, ...]))  # (H, W, D) -> (num_clusters + 1, H, W, D)
            self.retest_labels.append(self.post_processor(retest_label[None, ...]))
        self.test_labels = torch.tensor(np.stack(self.test_labels, axis=0))  # (N, num_clusters + 1, H, W, D)
        self.retest_labels = torch.tensor(np.stack(self.retest_labels, axis=0))  # (N, num_clusters + 1, H, W, D)
        self.metric_df = pd.DataFrame()
        self.dice_metric = DiceMetric(include_background=False, reduction="none")
        self.hausdorff_dist_metric = HausdorffDistanceMetric(include_background=False, percentile=self.params["hausdorff_dist_pct"], reduction="none")
    
    @abc.abstractmethod
    def compute_label_mapping(self):
        raise NotImplementedError        
    
    def __align_centroids(self):
        for i, test_centroids in enumerate(self.test_centroids):
            mapping = self.test_label2retest_label[i][1:] - 1  # converted to 0-based for all fg
            self.test_centroids[i] = test_centroids[mapping, :]

    def align_labels(self, test_label: np.ndarray, mapping: np.ndarray) -> np.ndarray:
        """
        test_label: (H, W, D)
        """
        test_label_aligned = mapping[test_label.flatten()].reshape(test_label.shape)

        return test_label_aligned

    def __call__(self) -> pd.DataFrame:
        self.compute_dsc()
        self.compute_hausdorff_dist()
        self.compute_centroid_dist()

        return self.metric_df
    
    def compute_dsc(self):
        metric_vals = self.dice_metric(self.test_labels, self.retest_labels).mean(dim=-1)  # (B, num_clusters) -> (B,)
        self.metric_df["dsc"] = ptu.to_numpy(metric_vals)

    def compute_hausdorff_dist(self):
        metric_vals = self.hausdorff_dist_metric(self.test_labels, self.retest_labels).mean(dim=-1)  # (B, num_clusters) -> (B,)
        self.metric_df["hausdorff_dist"] = ptu.to_numpy(metric_vals)

    def compute_centroid_dist(self):
        # avg on number of features
        test_centroids = np.stack(self.test_centroids, axis=0)  # (N, num_clusters, D_feats)
        retest_centroids = np.stack(self.retest_centroids, axis=0)
        dist = np.linalg.norm(test_centroids - retest_centroids, ord=self.params["centroid_dist_order"], axis=-1).mean(axis=1)  # (N,)
        self.metric_df["centroid_dist"] = dist / test_centroids.shape[-1]


# legacy code
class TestRetestPairwiseEvaluator(TestRetestBaseEvaluator):
    def compute_label_mapping(self):
        for i, (test_centroids, retest_centroids) in enumerate(zip(self.test_centroids, self.retest_centroids)):
            dist_mat = cdist(test_centroids, retest_centroids)
            self.test_label2retest_label[i][1:] = dist_mat.argmin(axis=1) + 1


# # Using the remapped atlas in "run_B/"
# def compute_dsc_two_scans(data_dict: dict, cluster_dict_A: dict, cluster_dict_B: dict, out_dir: str, num_classes=7) -> dict:
#     """
#     Compute DSC between two runs for whole thalamus.

#     Parameters
#     ----------
#     data_dict: dict
#         {
#             "run_A":
#             {
#                 "B2A": {"left": (4, 4), "right": (4, 4)},
#                 "left": 
#                 {
#                     "thalamus_mask": (H, W, D)
#                     "nucleigroups": (H, W, D)
#                 },
#                 "right": ...
#             },
#             "run_B": ...,
#         }
    
#     cluster_dict_A, cluster_dict_B: dict
#         {
#             "left":
#             {
#                 "atlas": (H, W, D)
#             }
#             "right":
#             {
#                 "atlas": (H, W, D)
#             }
#         }
    
#     out_dir: str
#         Temporary output directory for "flirt" command
    
#     Returns
#     -------
#     dict:
#         DSC for the whole thalamus between two scans
#         {   
#             "left_all": (num_classes,),
#             "left": float,
#             "right_all": (num_classes,),
#             "right": float,
#             "whole_all": (num_classes,),
#             "whole": 1 / 2 * (left + right)
#         }
#     """
#     dice_metric = DiceMetric(include_background=False, reduction="none")
#     to_onehot = AsDiscrete(to_onehot=num_classes + 1)

#     atlas_B2A = {}  # {"left": (H, W, D), "right": ...}
#     system_affine_A = data_dict["run_A"]["left"]["thalamus_mask"].affine
#     system_affine_B = data_dict["run_B"]["left"]["thalamus_mask"].affine

#     for key in ["left", "right"]:
#         atlas_iter_B = cluster_dict_B[key]["atlas"]
#         atlas_iter_A = cluster_dict_A[key]["atlas"]
#         atlas_iter_B = nib.Nifti1Image(atlas_iter_B, system_affine_B)
#         atlas_iter_A = nib.Nifti1Image(atlas_iter_A, system_affine_A)

#         affine_mat = data_dict["run_A"]["B2A"][key]
#         out_filename = os.path.join(out_dir, f"{key}.nii.gz")
#         atlas_iter_B2A = apply_affine(atlas_iter_B, atlas_iter_A, out_filename, affine_mat)
#         atlas_B2A[key] = atlas_iter_B2A.get_fdata()
    
#     atlas_B2A["whole"] = atlas_B2A["left"] + atlas_B2A["right"]  # (H, W, D)
#     cluster_dict_A["whole"] = {}
#     cluster_dict_A["whole"]["atlas"] = cluster_dict_A["left"]["atlas"] + cluster_dict_A["right"]["atlas"]  # (H, W, D)
#     out_dict = {}
#     for key in ["left", "right", "whole"]:
#         try:
#             atlas_B2A_iter = torch.tensor(atlas_B2A[key].astype(int)).unsqueeze(0)   # (1, H, W, D)
#             print(torch.unique(atlas_B2A_iter))
#             atlas_B2A_iter = to_onehot(atlas_B2A_iter).unsqueeze(0)  #(1, C + 1, H, W, D)
#             atlas_A_iter = torch.tensor(cluster_dict_A[key]["atlas"].astype("int")).unsqueeze(0)  # (1, H, W, D)
#             print(torch.unique(atlas_A_iter))
#             atlas_A_iter = to_onehot(atlas_A_iter).unsqueeze(0)  # (1, C + 1, H, W, D)
#             dsc_iter = dice_metric(atlas_B2A_iter, atlas_A_iter)  # (1, 1, C)
#             assert dsc_iter.shape == (1, num_classes)
#             out_dict[f"{key}_all"] = dsc_iter
#             out_dict[key] = dsc_iter.nanmean().item()
#         except Exception:
#             if key != "whole":
#                 raise ValueError(f"Error in computing DSC for {key} thalamus.")
#             else:
#                 print("Using average DSC of left and right thalamuses")
#                 out_dict[f"{key}_all"] = 0.5 * (out_dict["left_all"] + out_dict["right_all"])
#                 out_dict[key] = 0.5 * (out_dict["left"] + out_dict["right"])
    
#     return out_dict


def compute_dsc_two_scans(data_dict: dict, cluster_dict_A: dict, cluster_dict_B: dict, out_dir: str, num_classes=7, percentile=95.) -> dict:
    """
    Compute DSC between two runs for whole thalamus.

    Parameters
    ----------
    data_dict: dict
        {
            "run_A":
            {
                "B2A": {"left": (4, 4), "right": (4, 4), [to add: "left_B2A": (H, W, D), "right_B2A": (H, W, D)]},
                "left": 
                {
                    "thalamus_mask": (H, W, D),
                    "nucleigroups": (H, W, D)
                },
                "right": ...
            },
            "run_B": ...,
        }
    
    cluster_dict_A, cluster_dict_B: dict
        {
            "left":
            {
                "atlas_not_remapped": (H, W, D)
            }
            "right":
            {
                "atlas_not_remapped": (H, W, D)
            }
        }
    
    out_dir: str
        Temporary output directory for "flirt" command
    
    Returns
    -------
    dict
        DSC for the whole thalamus between two scans
        {   
            "left_all": (num_classes,),
            "left": float,
            "right_all": (num_classes,),
            "right": float,
            "whole_all": (num_classes,),
            "whole": float
        }
    """
    dice_metric = DiceMetric(include_background=False, reduction="none")
    to_onehot = AsDiscrete(to_onehot=num_classes + 1)

    atlas_B2A = {}  # {"left": (H, W, D), "right": ...}
    system_affine_A = data_dict["run_A"]["left"]["thalamus_mask"].affine
    system_affine_B = data_dict["run_B"]["left"]["thalamus_mask"].affine

    for key in ["left", "right"]:
        atlas_iter_B = cluster_dict_B[key]["atlas_not_remapped"]
        atlas_iter_A = cluster_dict_A[key]["atlas_not_remapped"]
        atlas_iter_B = nib.Nifti1Image(atlas_iter_B, system_affine_B)
        atlas_iter_A = nib.Nifti1Image(atlas_iter_A, system_affine_A)

        affine_mat = data_dict["run_A"]["B2A"][key]
        out_filename = os.path.join(out_dir, f"{key}.nii.gz")
        atlas_iter_B2A = apply_affine(atlas_iter_B, atlas_iter_A, out_filename, affine_mat)
        atlas_B2A[key], dist_df = align_labels(atlas_iter_B2A.get_fdata(), atlas_iter_A.get_fdata(), percentile)
        # print(dist_df.idxmin(axis=1))
        # save_image(atlas_B2A[key], os.path.join(out_dir, f"B2A_{key}.nii.gz"), system_affine_A)
        data_dict["run_A"]["B2A"][f"{key}_B2A"] = atlas_B2A[key]
    
    atlas_B2A["whole"] = atlas_B2A["left"] + atlas_B2A["right"]  # (H, W, D)
    cluster_dict_A["whole"] = {}
    cluster_dict_A["whole"]["atlas"] = cluster_dict_A["left"]["atlas"] + cluster_dict_A["right"]["atlas"]  # (H, W, D)
    out_dict = {}
    for key in ["left", "right", "whole"]:
        try:
            atlas_B2A_iter = torch.tensor(atlas_B2A[key].astype(int)).unsqueeze(0)   # (1, H, W, D)
            # print(torch.unique(atlas_B2A_iter))
            atlas_B2A_iter = to_onehot(atlas_B2A_iter).unsqueeze(0)  #(1, C + 1, H, W, D)
            atlas_A_iter = torch.tensor(cluster_dict_A[key]["atlas_not_remapped"].astype("int")).unsqueeze(0)  # (1, H, W, D)
            # print(torch.unique(atlas_A_iter))
            atlas_A_iter = to_onehot(atlas_A_iter).unsqueeze(0)  # (1, C + 1, H, W, D)
            dsc_iter = dice_metric(atlas_B2A_iter, atlas_A_iter)  # (1, C)
            assert dsc_iter.shape == (1, num_classes)
            out_dict[f"{key}_all"] = dsc_iter
            out_dict[key] = dsc_iter.nanmean().item()
        except Exception:
            if key != "whole":
                raise ValueError(f"Error in computing DSC for {key} thalamus.")
            else:
                print("Using average DSC of left and right thalamuses")
                out_dict[f"{key}_all"] = 0.5 * (out_dict["left_all"] + out_dict["right_all"])
                out_dict[key] = 0.5 * (out_dict["left"] + out_dict["right"])
    
    return out_dict


def compute_dsc_cluster_and_histology(data_dict: dict, cluster_dict: dict):
    """
    Works for one run.

    Parameters
    ----------
    data_dict: dict (sub-dict for one run)
        {
            
            "left":
            {
                "nucleigroups": (H, W, D)
            },
            "right": ...   
        }
    
    cluster_dict: dict
        {
            "left":
            {
                "atlas": (H, W, D)
            },
            "right": ...
        }
    
    Returns
    -------
    dict
        DSC between the whole thalamus and histology atlas
        {   
            "left_all": (num_classes,),
            "left": float,
            "right_all": (num_classes,),
            "right": float,
            "whole_all": (num_classes,),
            "whole": float
        }
    """
    dice_metric = DiceMetric(include_background=False, reduction="none")
    all_classes = np.unique(data_dict["left"]["nucleigroups"].get_fdata())
    num_classes = all_classes.shape[0]  # including bg
    to_onehot = AsDiscrete(to_onehot=num_classes)

    data_dict["whole"] = {}
    data_dict["whole"]["nucleigroups"] = data_dict["left"]["nucleigroups"].get_fdata() + data_dict["right"]["nucleigroups"].get_fdata()
    cluster_dict["whole"] = {}
    cluster_dict["whole"]["atlas"] = cluster_dict["left"]["atlas"] + cluster_dict["right"]["atlas"]

    out_dict = {}
    for key in ["left", "right", "whole"]:
        try:
            atlas_iter = torch.tensor(cluster_dict[key]["atlas"].astype(int)).unsqueeze(0)  # (1, H, W, D)
            standard_iter = torch.tensor(data_dict[key]["nucleigroups"].get_fdata().astype(int)).unsqueeze(0)  # (1, H, W, D)
            atlas_iter = to_onehot(atlas_iter).unsqueeze(0)  # (1, C + 1, H, W, D)
            standard_iter = to_onehot(standard_iter).unsqueeze(0)  # (1, C + 1, H, W, D)
            dsc_iter = dice_metric(atlas_iter, standard_iter)  # (1, C)
            assert dsc_iter.shape == (1, num_classes - 1)
            out_dict[f"{key}_all"] = dsc_iter
            out_dict[key] = dsc_iter.nanmean().item()
        except Exception:
            if key != "whole":
                raise ValueError(f"Error in computing DSC for {key} thalamus.")
            else:
                print("Using average DSC of left and right thalamuses")
                out_dict[f"{key}_all"] = 0.5 * (out_dict["left_all"] + out_dict["right_all"])
                out_dict[key] = 0.5 * (out_dict["left"] + out_dict["right"])
    
    return out_dict
