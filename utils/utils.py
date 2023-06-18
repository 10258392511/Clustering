import numpy as np
import nibabel as nib
import os
import glob
import scipy.io as sio
import re

from monai.transforms import Affine
from jax.tree_util import tree_map
from typing import Dict
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
    img_nib = nib.Nifti1Image(img, affine)
    nib.save(img_nib, filename)


def read_data(dirname: str, if_read_tfm=True) -> dict:
    """
    dirname: */pp*

    out_dict:
    {
        'run_A': {'B2A': {'left': (4, 4), 'right': (4, 4)},
           'left': {'dist_maps': {'1': (112, 112, 60)...},
                    'nucleigroups': (112, 112, 60),
                    'thalamus_mask': (112, 112, 60),
                    'thalamus_atlas_mask': (112, 112, 60)},
           'right': {'dist_maps': {'1': (112, 112, 60)...},
                     'nucleigroups': (112, 112, 60),
                     'thalamus_mask': (112, 112, 60),
                     'thalamus_atlas_mask': (112, 112, 60)},
           'spherical_coeffs': (112, 112, 60, 45)},
        'run_B': ...
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


def apply_affine(img: nib.Nifti1Image, tfm_mat: np.ndarray):
    data = img.get_fdata()
    affine = img.affine
    new_affine = tfm_mat
    affine_tfm = Affine(mode="nearest", affine=new_affine)
    data_tfm, _ = affine_tfm(data[None, ...])
    img_tfm = nib.Nifti1Image(data_tfm[0, ...], affine)

    return img_tfm
