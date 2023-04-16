import numpy as np

from jax.tree_util import tree_map
from typing import Dict


def print_data_dict_shape(data_dict: Dict[str, np.ndarray]):
    shape_dict = tree_map(lambda x : x.shape, data_dict)
    print(shape_dict)


def binarize_mask(mask: np.ndarray, th: float):
    mask_binary = (mask > th).astype(float)

    return mask_binary
