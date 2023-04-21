import numpy as np
import nibabel as nib

from jax.tree_util import tree_map
from typing import Dict


def print_data_dict_shape(data_dict: Dict[str, np.ndarray]):
    shape_dict = tree_map(lambda x : x.shape, data_dict)
    print(shape_dict)


def binarize_mask(mask: np.ndarray, th: float):
    mask_binary = (mask > th).astype(float)

    return mask_binary


def save_image(img: np.ndarray, filename: str, affine: np.ndarray = np.eye(4)):
    assert ".nii.gz" in filename
    img_nib = nib.Nifti1Image(img, affine)
    nib.save(img_nib, filename)
