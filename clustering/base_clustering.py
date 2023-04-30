import numpy as np
import abc

from typing import Sequence


class BaseCluster(abc.ABC):
    def __init__(self, in_shape: Sequence[int], features: np.ndarray, ijk_inds: np.ndarray, params: dict):
        self.in_shape = in_shape  # (D, H, W)
        self.features = features  # (N, D_feature)
        self.inds = ijk_inds  # (N, 3)
        self.params = params  # for sklearn's clustering model; "clustering" node

    @abc.abstractmethod
    def fit_transform(self) -> np.ndarray:
        """
        Returns the volume of clustering labels, of shape .in_shape 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_centroids(self):
        raise NotImplementedError
