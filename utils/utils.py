import numpy as np
import pandas as pd
import nibabel as nib
import os
import glob
import scipy.io as sio
import re

from fsl.wrappers import applyxfm
from monai.transforms import Affine
from monai.metrics import compute_hausdorff_distance
from jax.tree_util import tree_map
from typing import Dict, Union
from pprint import pprint


def print_data_dict_shape(data_dict: Dict[str, np.ndarray], return_type=False):
    def func(x):
        x_out = None
        try:
            x_out = x.shape
        except AttributeError:
            x_out = x
        if return_type:
            x_out = (x_out, type(x))
        
        return x_out
    
    shape_dict = tree_map(func, data_dict)
    pprint(shape_dict)


def binarize_mask(mask: np.ndarray, th: float):
    mask_binary = (mask > th).astype(float)

    return mask_binary


def save_image(img: np.ndarray, filename: str, affine: np.ndarray = np.eye(4)):
    assert ".nii.gz" in filename
    output_dir = os.path.dirname(filename)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    img_nib = nib.Nifti1Image(img.astype(float), affine)
    nib.save(img_nib, filename)


def read_data(dirname: str, if_read_tfm=True) -> dict:
    """
    dirname: */pp*

    out_dict:
    {
        'run_A': [{'B2A': {'left': (4, 4), 'right': (4, 4)},]
           'left': {'dist_maps': {'1': (112, 112, 60)...},
                    'nucleigroups': (112, 112, 60),
                    'thalamus_mask': (112, 112, 60),
                    'thalamus_atlas_mask': (112, 112, 60)},
           'right': {'dist_maps': {'1': (112, 112, 60)...},
                     'nucleigroups': (112, 112, 60),
                     'thalamus_mask': (112, 112, 60),
                     'thalamus_atlas_mask': (112, 112, 60)},
           'spherical_coeffs': (112, 112, 60, 45)},
        ['run_B': ...,]
        'spherical_coeffs': (112, 112, 60, 45)}
    }
    """
    all_runs = glob.glob(os.path.join(dirname, "run_*"))
    out_dict = {}
    for run_iter in all_runs:
        run_dict_iter = {}
        run_dict_iter["spherical_coeffs"] = nib.load(os.path.join(run_iter, "spherical_coeffs.nii.gz"))
        if if_read_tfm:
            try:
                run_dict_iter["B2A"] = {
                    "left": sio.loadmat(os.path.join(dirname, "run_A", f"B_to_A_left.mat")),
                    "right": sio.loadmat(os.path.join(dirname, "run_A", f"B_to_A_right.mat"))
                }
            except ValueError:
                run_dict_iter["B2A"] = {
                    "left": np.loadtxt(os.path.join(dirname, "run_A", f"B_to_A_left.mat")),
                    "right": np.loadtxt(os.path.join(dirname, "run_A", f"B_to_A_right.mat"))
                }
        for key_iter in ["left", "right"]:
            individual_thalamus_dict = {}
            thalamus_subdir_name = glob.glob(os.path.join(run_iter, "thalamus*"))[0]
            individual_thalamus_dict["thalamus_mask"] = nib.load(os.path.join(run_iter, f"thalamus_mask_{key_iter}.nii.gz"))
            individual_thalamus_dict["thalamus_atlas_mask"] = nib.load(os.path.join(thalamus_subdir_name, f"{key_iter}_thalamus_atlasmask.nii.gz"))
            individual_thalamus_dict["nucleigroups"] = nib.load(os.path.join(thalamus_subdir_name, f"{key_iter}_thalamus_nucleigroups_nonlinear.nii.gz"))
            individual_thalamus_dict["dist_maps"] = {}
            all_paths = glob.glob(os.path.join(run_iter, "thalamus*/*.nii.gz"))
            pattern = key_iter + r"_.*(\d+).*distance_feature.*nii.gz"
            for filename_iter in all_paths:
                idx = re.search(pattern, filename_iter)
                if idx is None:
                    continue
                individual_thalamus_dict["dist_maps"][idx.group(1)] = nib.load(filename_iter)

            run_dict_iter[key_iter] = individual_thalamus_dict
    
        out_dict[os.path.basename(run_iter)] = run_dict_iter

    return out_dict


def apply_affine(src: nib.Nifti1Image, ref: nib.Nifti1Image, out: str, mat: np.ndarray, interp="nearestneighbour") -> nib.Nifti1Image:
    out_dir = os.path.dirname(out)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    applyxfm(src, ref, mat, out, interp)
    img_tfm = nib.load(out)

    return img_tfm


def save_prob_maps(save_dir: str, left_prob_maps: Union[np.ndarray, None] = None, right_prob_maps: Union[np.ndarray, None] = None, affine: np.ndarray = np.eye(4)):
    """
    left/right_prob_maps: (H, W, D, num_clusters + 1); discarding bg -> (H, W, D, num_clusters)
    """
    assert left_prob_maps is not None or right_prob_maps is not None
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    def save_thalamus(prob_maps: np.ndarray, key: str, parent_dir: str, affine: np.ndarray = np.eye(4)):
        """ 
        key: "left", "right" or "whole"
        """
        save_dir = os.path.join(parent_dir, key)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_image(prob_maps, os.path.join(save_dir, "all_clusters.nii.gz"), affine)
        for channel_iter in range(1, prob_maps.shape[-1]):
            save_image(prob_maps[..., channel_iter], os.path.join(save_dir, f"cluster_{channel_iter}.nii.gz"), affine)
    
    if left_prob_maps is not None:
        save_thalamus(left_prob_maps, "left", save_dir)
    if right_prob_maps is not None:
        save_thalamus(right_prob_maps, "right", save_dir)
    if left_prob_maps is not None and right_prob_maps is not None:
        save_thalamus(left_prob_maps + right_prob_maps, "whole", save_dir)


def align_labels(atlas: np.ndarray, standard_atlas: np.ndarray, percentile: float):
    """
    Parameters
    ----------
    atlas1, atlas2: np.ndarray
    percentile: float
        Hausdorff distance percentile

    Returns
    -------
    aligned_atlas: (H, W, D)
    hausdorff_dist_df: (num_clusters, num_clusters_standard)
    """
    atlas = atlas.astype(int)
    standard_atlas = standard_atlas.astype(int)

    cluster_inds = np.unique(atlas)
    cluster_inds.sort()
    cluster_inds = cluster_inds[1:]
    standard_inds = np.unique(standard_atlas)
    standard_inds.sort()
    standard_inds = standard_inds[1:]

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
            
            dist = compute_hausdorff_distance(cluster_atlas_iter[None, None, ...], standard_atlas_iter[None, None, ...], percentile=percentile)
            hausdorff_dist_df.iloc[i, j] = dist[0, 0].item()
    
    atlas2standard = hausdorff_dist_df.idxmin(axis=1)
    atlas_out = atlas.copy()
    for label_iter in cluster_inds:
        label_iter = int(label_iter)
        atlas_out[atlas == label_iter] = atlas2standard[label_iter]

    return atlas_out, hausdorff_dist_df
