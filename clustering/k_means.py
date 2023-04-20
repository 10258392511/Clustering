from .base_clustering import *
from sklearn.cluster import KMeans


class KMeansCluster(BaseCluster):
    def __init__(self, in_shape: Sequence[int], features: np.ndarray, ijk_inds: np.ndarray, config: dict):
        super().__init__(in_shape, features, ijk_inds, config)
        self.model = KMeans(**self.params["params"])
    
    def fit_transform(self) -> np.ndarray:
        self.model.fit(self.features)
        labels = self.model.labels_  # (N,)
        img_out = np.zeros(self.in_shape, dtype=int)  # bg: 0
        img_out[self.inds[:, 0], self.inds[:, 1], self.inds[:, 2]] = labels.astype(int) + 1

        return img_out    
