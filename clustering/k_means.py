from .base_clustering import *
from sklearn.cluster import KMeans


class KMeansCluster(BaseCluster):
    def __init__(self, config: dict):
        super().__init__(config)
    
    def init_means(self, data_dict: dict, feature_dict: dict):
        if self.config["init"] != "histology_atlas":
            return
        init_centroids_dict = self._compute_means_from_histology_atlas(data_dict, feature_dict)
        for key in ["left", "right"]:
            params = self.config["clustering"]["params"]
            params["init"] = init_centroids_dict[key]
            self.models[key] = KMeans(**params)

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
            if num_clusters != 7:
                tgt_channel = label
            else:
                tgt_channel = atlas2standard[label]
            mask = (feature_probs == label)
            coords_iter = coords[mask, ...]  # (N_all',)
            prob_maps[coords_iter[:, 0], coords_iter[:, 1], coords_iter[:, 2], tgt_channel] += 1
        
        assert np.all(prob_maps <= 1)
        return prob_maps
    
