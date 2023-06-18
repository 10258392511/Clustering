from .base_clustering import *
# from sklearn.cluster import KMeans


# class KMeansCluster(BaseCluster):
#     def __init__(self, in_shape: Sequence[int], features: np.ndarray, ijk_inds: np.ndarray, config: dict):
#         super().__init__(in_shape, features, ijk_inds, config)
#         self.model = KMeans(**self.params["params"])
    
#     def fit_transform(self) -> np.ndarray:
#         self.model.fit(self.features)
#         labels = self.model.labels_  # (N,)
#         img_out = np.zeros(self.in_shape, dtype=int)  # bg: 0
#         img_out[self.inds[:, 0], self.inds[:, 1], self.inds[:, 2]] = labels.astype(int) + 1

#         return img_out    
    
#     def get_centroids(self):
        
#         return self.model.cluster_centers_


class KMeansCluster(BaseCluster):
    def __init__(self, config: dict):
        super().__init__(config)
    
    def create_probabilistic_maps(self, feature_dict: dict, key: str, hausdorff_dist_df: pd.DataFrame):
        model = self.models[key]
        thalamus_feature_dict = feature_dict[key]
        coords = thalamus_feature_dict["coords"]  # (N_all,)
        atlas2standard = hausdorff_dist_df.idxmin(axis=1)
        shape = thalamus_feature_dict["shape"]
        num_clusters = model.cluster_centers_.shape[0]
        prob_maps = np.zeros((*shape, num_clusters + 1))
        feature_probs = model.predict(thalamus_feature_dict["features"])  # (N_all,)
        feature_probs += 1

        for label in range(num_clusters):
            label += 1
            tgt_channel = atlas2standard[label]
            mask = (feature_probs == label)
            coords_iter = coords[mask, ...]  # (N_all',)
            prob_maps[coords_iter[:, 0], coords_iter[:, 1], coords_iter[:, 2], tgt_channel] += 1
        
        assert np.all(prob_maps <= 1)
        return prob_maps
    